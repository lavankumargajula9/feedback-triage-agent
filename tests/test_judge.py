"""Tests for the LLM draft judge (U8) — R13, R14, R22, R27, AE6.

Covers:
- Schema shape: critique fields come BEFORE their scores (critique-then-score,
  R13), scores are bounded 1-5, and the four dimensions are exactly R13's.
- Prompt assembly: anchored scale examples, injected anchor-stock few-shots,
  and length-neutral instructions (R13); judge model + schema reach the call.
- Scoring runs (mocked judge): both arms scored, escalated threads included,
  per-dimension means + mean draft score + binary send verdicts; unscorable
  drafts counted and disclosed, never dropped from the denominator (R27, AE6).
- Agreement stats (R14): hand-computed exact / within-one / quadratic-weighted
  kappa, score distributions, subsample size; anchor-stock items are excluded;
  kappa edge cases (perfect agreement = 1.0, all-same-score degenerate case).

Every test is fully mocked: zero network.

Hand-computed agreement example (single pooled dimension):
  judge  a:5 b:4 c:3 d:2 e:1
  human  a:5 b:3 c:3 d:2 e:2
  exact 3/5, within-one 5/5.
  Quadratic weighted kappa, w(i,j) = (i-j)^2/16:
    observed = (0 + 1 + 0 + 0 + 1)/16 = 0.125
    expected = sum_ij r_i*c_j*w(i,j)/n with judge marginals all 1 and human
      marginals c2=2, c3=2, c5=1 = (26 + 11 + 6 + 11 + 26)/(16*5) = 1.0
    kappa = 1 - 0.125/1.0 = 0.875
"""

import json

import pytest
from pydantic import ValidationError
from test_tools import ExplodingClient, FakeClient, ok, validation_error_for

from triage.evals.judge import (
    AGREEMENT_PROTOCOL,
    DIMENSIONS,
    JUDGE_CALLS_PER_THREAD,
    JUDGE_STEP,
    AnchorExample,
    DimensionScore,
    JudgeResult,
    compute_agreement,
    judge_checkpoint_path,
    judge_cost_per_call,
    judge_draft,
    judge_system,
    judge_user_text,
    load_anchor_stock,
    score_run,
)
from triage.evals.runner import EvalError, GoldLabel

JUDGE_MODEL = "claude-fable-5"

ANCHOR = AnchorExample(
    item_id="anchor-1",
    category="Billing/Payments",
    draft="Hi! We've refunded the duplicate charge; you'll see it in 3-5 days.",
    scores={"correctness": 5, "tone": 5, "grounding": 4, "actionability": 5},
    critiques={
        "correctness": "Addresses the exact billing issue raised.",
        "tone": "Warm and professional.",
        "grounding": "The 3-5 day window is a plausible but unverified claim.",
        "actionability": "Names the concrete next step and its timing.",
    },
    send_as_is=True,
)


def jr(c, t, g, a, send=True):
    return JudgeResult(
        correctness=DimensionScore(critique="c", score=c),
        tone=DimensionScore(critique="t", score=t),
        grounding=DimensionScore(critique="g", score=g),
        actionability=DimensionScore(critique="a", score=a),
        send_critique="overall take",
        send_as_is=send,
    )


def make_gold(spec):
    return {
        tid: GoldLabel(
            thread_id=tid, category=cat, queue="Technical Support", escalate=esc
        )
        for tid, (cat, esc) in spec.items()
    }


def make_entry(tid, pipeline_draft, baseline_draft):
    """Checkpoint-shaped entry; a draft of None is a typed step failure."""
    if pipeline_draft is None:
        draft_step = {"failure": {"step": "draft", "kind": "malformed", "attempts": 3,
                                  "detail": "x"}}
    else:
        draft_step = {"draft": pipeline_draft, "status": "never_sent"}
    if baseline_draft is None:
        baseline = {"failure": {"step": "baseline", "kind": "malformed", "attempts": 3,
                                "detail": "x"}}
    else:
        baseline = {
            "category": "Technical/Product",
            "queue": "Technical Support",
            "escalate": False,
            "escalate_reason": "r",
            "draft": baseline_draft,
            "status": "never_sent",
        }
    return {
        "thread_id": tid,
        "pipeline": {
            "steps": {
                "categorize": {"label": "Technical/Product", "rationale": "r"},
                "route": {"queue": "Technical Support", "rationale": "r"},
                "escalate": {"escalate": False, "reason": "r"},
                "draft": draft_step,
            },
            "failed_steps": [] if pipeline_draft is not None else ["draft"],
            "ok": pipeline_draft is not None,
        },
        "baseline": baseline,
        "ok": pipeline_draft is not None and baseline_draft is not None,
    }


def make_results(entries):
    return {"run_id": "", "models": {}, "prompt_hashes": {}, "eval_set_hash": "",
            "threads": entries}


def score_args(tmp_path, out_name="judge-run"):
    """Checkpoint dir + shared call cache every score_run needs (R32, KTD5)."""
    return {"out_dir": tmp_path / out_name, "cache_path": tmp_path / "cache.db"}


class TestSchema:
    def test_dimensions_are_exactly_r13s(self):
        assert DIMENSIONS == ("correctness", "tone", "grounding", "actionability")

    def test_critique_comes_before_score(self):
        # R13: the judge outputs its critique before each score; with a
        # generated schema, field order is generation order.
        assert list(DimensionScore.model_fields) == ["critique", "score"]
        fields = list(JudgeResult.model_fields)
        assert fields.index("send_critique") < fields.index("send_as_is")
        for dim in DIMENSIONS:
            assert fields.index(dim) < fields.index("send_as_is")

    def test_scores_are_bounded_one_to_five(self):
        with pytest.raises(ValidationError):
            DimensionScore(critique="x", score=6)
        with pytest.raises(ValidationError):
            DimensionScore(critique="x", score=0)

    def test_send_verdict_is_binary(self):
        assert jr(3, 3, 3, 3, send=False).send_as_is is False


class TestPrompt:
    def test_system_prompt_is_length_neutral_and_anchored(self):
        system = judge_system([ANCHOR])
        assert "length" in system.lower()  # length-neutral instruction (R13)
        assert "1" in system and "5" in system  # anchored scale
        # Anchor-stock few-shot example is injected verbatim.
        assert ANCHOR.draft in system
        assert ANCHOR.critiques["grounding"] in system
        assert "critique" in system.lower()  # critique-before-score

    def test_user_text_carries_draft_category_and_escalation(self):
        text = judge_user_text(
            "We're on it!", category="Billing/Payments", escalated=True
        )
        assert "We're on it!" in text
        assert "Billing/Payments" in text
        assert "escalat" in text.lower()

    def test_judge_call_uses_judge_model_and_schema(self):
        client = FakeClient([ok(jr(4, 4, 4, 4))])
        result = judge_draft(
            "Sorry about that!",
            category="Technical/Product",
            escalated=False,
            model=JUDGE_MODEL,
            anchors=[ANCHOR],
            client=client,
        )
        assert isinstance(result, JudgeResult)
        call = client.messages.calls[0]
        assert call["model"] == JUDGE_MODEL
        assert call["output_format"] is JudgeResult
        assert call["system"] == judge_system([ANCHOR])


class TestScoreRun:
    def test_scores_both_arms_including_escalated_threads(self, tmp_path):
        # Thread 2 is escalated (gold) and still judged for both arms (R13).
        gold = make_gold({1: ("Technical/Product", False), 2: ("Billing/Payments", True)})
        results = make_results(
            [make_entry(1, "p-draft-1", "b-draft-1"), make_entry(2, "p-draft-2", "b-draft-2")]
        )
        # Call order: per thread, pipeline then baseline.
        client = FakeClient(
            [ok(jr(5, 4, 4, 5, send=True)), ok(jr(2, 2, 2, 2, send=False)),
             ok(jr(3, 3, 3, 3, send=False)), ok(jr(4, 4, 4, 4, send=True))]
        )
        run = score_run(results, gold, model=JUDGE_MODEL, anchors=[ANCHOR], client=client,
                        **score_args(tmp_path))
        pipeline = run["systems"]["pipeline"]
        # Hand-computed: thread means 4.5 and 3.0 -> mean draft score 3.75.
        assert pipeline["mean_draft_score"] == pytest.approx(3.75)
        assert pipeline["mean_scores"]["correctness"] == pytest.approx(4.0)
        assert pipeline["send_rate"] == pytest.approx(0.5)
        assert set(pipeline["per_thread"]) == {1, 2}  # escalated thread included
        baseline = run["systems"]["baseline"]
        assert baseline["mean_draft_score"] == pytest.approx(3.0)
        assert baseline["send_rate"] == pytest.approx(0.5)
        assert len(client.messages.calls) == 4
        # The escalated thread's judge prompt says so.
        escalated_call = client.messages.calls[2]
        assert "escalat" in escalated_call["messages"][0]["content"].lower()

    def test_missing_draft_is_unscorable_counted_not_dropped(self, tmp_path):
        # AE6/R27: thread 2's pipeline draft failed -> scored at the floor,
        # kept in the denominator, and disclosed as unscorable.
        gold = make_gold({1: ("Technical/Product", False), 2: ("Technical/Product", False)})
        results = make_results(
            [make_entry(1, "p-draft-1", "b-draft-1"), make_entry(2, None, "b-draft-2")]
        )
        client = FakeClient(
            [ok(jr(5, 5, 5, 5)), ok(jr(3, 3, 3, 3)), ok(jr(3, 3, 3, 3))]
        )
        run = score_run(results, gold, model=JUDGE_MODEL, anchors=[ANCHOR], client=client,
                        **score_args(tmp_path))
        pipeline = run["systems"]["pipeline"]
        assert pipeline["unscorable"] == {"count": 1, "total": 2, "rate": pytest.approx(0.5)}
        assert pipeline["mean_draft_score"] == pytest.approx(3.0)  # (5 + 1)/2
        assert pipeline["per_thread"][2]["unscorable"] is True
        assert pipeline["per_thread"][2]["send_as_is"] is False
        assert run["systems"]["baseline"]["unscorable"]["count"] == 0
        assert len(client.messages.calls) == 3  # no judge call for the missing draft

    def test_judge_output_failure_is_unscorable_counted_not_dropped(self, tmp_path):
        gold = make_gold({1: ("Technical/Product", False)})
        results = make_results([make_entry(1, "p-draft-1", "b-draft-1")])
        bad = [validation_error_for(JudgeResult, "{not json") for _ in range(3)]
        client = FakeClient(bad + [ok(jr(4, 4, 4, 4))])
        run = score_run(results, gold, model=JUDGE_MODEL, anchors=[ANCHOR], client=client,
                        **score_args(tmp_path))
        pipeline = run["systems"]["pipeline"]
        assert pipeline["unscorable"] == {"count": 1, "total": 1, "rate": 1.0}
        assert pipeline["mean_draft_score"] == pytest.approx(1.0)  # floor, in denominator
        assert run["systems"]["baseline"]["unscorable"]["count"] == 0
        assert run["systems"]["baseline"]["mean_draft_score"] == pytest.approx(4.0)


TWO_THREADS = make_results(
    [make_entry(1, "p-draft-1", "b-draft-1"), make_entry(2, "p-draft-2", "b-draft-2")]
)
TWO_GOLD = make_gold({1: ("Technical/Product", False), 2: ("Billing/Payments", True)})
FOUR_SCORES = [ok(jr(5, 4, 4, 5)), ok(jr(2, 2, 2, 2)), ok(jr(3, 3, 3, 3)), ok(jr(4, 4, 4, 4))]


def score(results, gold, client, args, **extra):
    return score_run(
        results, gold, model=JUDGE_MODEL, anchors=[ANCHOR], client=client, **args, **extra
    )


class TestJudgeCheckpointResume:
    """R32/KTD5 applied to the judge — the priciest calls in the system."""

    def test_mid_run_crash_leaves_a_resumable_checkpoint(self, tmp_path):
        # The crash lands on thread 1's baseline draft, after its pipeline
        # draft was already paid for and checkpointed.
        args = score_args(tmp_path)
        failing = FakeClient([ok(jr(5, 4, 4, 5)), RuntimeError("simulated transport failure")])
        with pytest.raises(RuntimeError, match="transport"):
            score(TWO_THREADS, TWO_GOLD, failing, args)
        assert judge_checkpoint_path(args["out_dir"], "pipeline", 1).is_file()
        assert not judge_checkpoint_path(args["out_dir"], "baseline", 1).is_file()

        resumed = FakeClient(list(FOUR_SCORES))
        run = score(TWO_THREADS, TWO_GOLD, resumed, args)
        # The money assertion: the already-scored item is NOT re-called.
        assert len(resumed.messages.calls) == 3
        assert run["systems"]["pipeline"]["per_thread"][1]["mean"] == pytest.approx(4.5)

    def test_resumed_scores_match_an_uninterrupted_run(self, tmp_path):
        args = score_args(tmp_path)
        failing = FakeClient([ok(jr(5, 4, 4, 5)), RuntimeError("boom")])
        with pytest.raises(RuntimeError):
            score(TWO_THREADS, TWO_GOLD, failing, args)
        resumed = score(TWO_THREADS, TWO_GOLD, FakeClient(list(FOUR_SCORES)), args)
        fresh = score(
            TWO_THREADS,
            TWO_GOLD,
            FakeClient(list(FOUR_SCORES)),
            score_args(tmp_path, out_name="fresh"),
        )
        assert resumed["systems"] == fresh["systems"]

    def test_checkpoint_from_a_different_judge_model_is_refused(self, tmp_path):
        args = score_args(tmp_path)
        score(TWO_THREADS, TWO_GOLD, FakeClient(list(FOUR_SCORES)), args)
        with pytest.raises(EvalError, match="judge_model"):
            score_run(
                TWO_THREADS, TWO_GOLD, model="claude-opus-5", anchors=[ANCHOR],
                client=FakeClient(list(FOUR_SCORES)), **args,
            )

    def test_checkpoint_from_a_different_judge_prompt_is_refused(self, tmp_path):
        # R14: the anchor stock is part of the judge; swapping it changes what
        # produced the scores, so the old checkpoints cannot be mixed in.
        args = score_args(tmp_path)
        score(TWO_THREADS, TWO_GOLD, FakeClient(list(FOUR_SCORES)), args)
        other = AnchorExample(
            item_id="anchor-2", category="Billing/Payments", draft="A different anchor draft.",
            scores=dict(ANCHOR.scores), critiques=dict(ANCHOR.critiques), send_as_is=False,
        )
        with pytest.raises(EvalError, match="judge_prompt_hash"):
            score_run(
                TWO_THREADS, TWO_GOLD, model=JUDGE_MODEL, anchors=[other],
                client=FakeClient(list(FOUR_SCORES)), **args,
            )

    def test_checkpoint_scored_from_a_different_draft_is_refused(self, tmp_path):
        args = score_args(tmp_path)
        score(TWO_THREADS, TWO_GOLD, FakeClient(list(FOUR_SCORES)), args)
        redrafted = make_results(
            [make_entry(1, "a completely rewritten draft", "b-draft-1"),
             make_entry(2, "p-draft-2", "b-draft-2")]
        )
        with pytest.raises(EvalError, match="thread 1"):
            score(redrafted, TWO_GOLD, FakeClient(list(FOUR_SCORES)), args)

    def test_checkpoint_records_what_produced_it(self, tmp_path):
        args = score_args(tmp_path)
        score(TWO_THREADS, TWO_GOLD, FakeClient(list(FOUR_SCORES)), args)
        entry = json.loads(
            judge_checkpoint_path(args["out_dir"], "pipeline", 1).read_text(encoding="utf-8")
        )
        assert entry["identity"]["judge_model"] == JUDGE_MODEL
        assert entry["identity"]["judge_prompt_hash"]
        assert entry["identity"]["run_id"] == ""
        assert entry["draft_fingerprint"]
        assert entry["record"]["mean"] == pytest.approx(4.5)

    def test_replay_through_the_cache_makes_zero_live_calls(self, tmp_path):
        # KTD5: a second run with fresh checkpoints but the same cache replays
        # every judge call for free — the client is never touched.
        args = score_args(tmp_path)
        score(TWO_THREADS, TWO_GOLD, FakeClient(list(FOUR_SCORES)), args)
        replay_args = {**args, "out_dir": tmp_path / "replay"}
        replayed = score(TWO_THREADS, TWO_GOLD, ExplodingClient(), replay_args)
        assert replayed["systems"]["pipeline"]["mean_draft_score"] == pytest.approx(3.75)

    def test_run_id_salt_bypasses_prior_cache_entries(self, tmp_path):
        # R21 variance passes: a salted run must not replay the unsalted one.
        args = score_args(tmp_path)
        score(TWO_THREADS, TWO_GOLD, FakeClient(list(FOUR_SCORES)), args)
        salted = FakeClient(list(FOUR_SCORES))
        score(TWO_THREADS, TWO_GOLD, salted, {**args, "out_dir": tmp_path / "v2"},
              run_id="variance-1")
        assert len(salted.messages.calls) == 4


class TestJudgeCostPlan:
    def test_judge_calls_are_per_thread_per_arm(self):
        assert JUDGE_CALLS_PER_THREAD == 2

    def test_judge_model_is_priced(self):
        # The operator decides to spend based on this number, so the judge
        # model must not price as "unknown" (plan: ~$0.05/call on Fable 5).
        assert judge_cost_per_call(JUDGE_MODEL) == pytest.approx(0.05)


class TestAnchorStock:
    def test_load_anchor_stock_round_trips(self, tmp_path):
        path = tmp_path / "anchors.json"
        path.write_text(
            json.dumps([
                {
                    "item_id": ANCHOR.item_id,
                    "category": ANCHOR.category,
                    "draft": ANCHOR.draft,
                    "scores": dict(ANCHOR.scores),
                    "critiques": dict(ANCHOR.critiques),
                    "send_as_is": ANCHOR.send_as_is,
                }
            ]),
            encoding="utf-8",
        )
        anchors = load_anchor_stock(path)
        assert [a.item_id for a in anchors] == ["anchor-1"]
        assert judge_system(anchors) == judge_system([ANCHOR])

    def test_missing_anchor_file_names_the_labeling_pass(self, tmp_path):
        with pytest.raises(EvalError, match="anchor stock"):
            load_anchor_stock(tmp_path / "absent.json")

    def test_anchor_missing_a_dimension_score_is_rejected(self, tmp_path):
        path = tmp_path / "anchors.json"
        path.write_text(
            json.dumps([{
                "item_id": "a", "category": "Billing/Payments", "draft": "d",
                "scores": {"correctness": 5}, "critiques": dict(ANCHOR.critiques),
                "send_as_is": True,
            }]),
            encoding="utf-8",
        )
        with pytest.raises(EvalError, match="tone"):
            load_anchor_stock(path)


class TestAgreement:
    def test_hand_computed_agreement_stats(self):
        judge = {"a": 5, "b": 4, "c": 3, "d": 2, "e": 1}
        human = {"a": 5, "b": 3, "c": 3, "d": 2, "e": 2}
        stats = compute_agreement(judge, human)
        assert stats["n_items"] == 5
        assert stats["n_scores"] == 5
        assert stats["exact"] == pytest.approx(0.6)
        assert stats["within_one"] == pytest.approx(1.0)
        assert stats["weighted_kappa"] == pytest.approx(0.875)
        assert stats["judge_distribution"] == {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
        assert stats["human_distribution"] == {2: 2, 3: 2, 5: 1}

    def test_anchor_items_are_excluded_from_agreement(self):
        # R14: agreement only over items the judge never saw as examples.
        judge = {"a": 5, "b": 4, "c": 3, "d": 2, "e": 1}
        human = {"a": 5, "b": 3, "c": 3, "d": 2, "e": 2}
        stats = compute_agreement(judge, human, anchor_ids={"a"})
        assert stats["n_items"] == 4
        assert stats["excluded_anchor_items"] == 1
        # Remaining pairs: (4,3), (3,3), (2,2), (1,2) -> exact 2/4.
        assert stats["exact"] == pytest.approx(0.5)
        assert stats["within_one"] == pytest.approx(1.0)

    def test_per_dimension_scores_are_pooled(self):
        judge = {"a": {"correctness": 5, "tone": 3}, "b": {"correctness": 2, "tone": 2}}
        human = {"a": {"correctness": 5, "tone": 4}, "b": {"correctness": 2, "tone": 2}}
        stats = compute_agreement(judge, human)
        assert stats["n_items"] == 2
        assert stats["n_scores"] == 4
        assert stats["exact"] == pytest.approx(0.75)

    def test_perfect_agreement_kappa_is_one(self):
        judge = {"a": 5, "b": 3, "c": 1}
        stats = compute_agreement(judge, dict(judge))
        assert stats["weighted_kappa"] == pytest.approx(1.0)
        assert stats["exact"] == 1.0

    def test_all_same_score_degenerate_kappa_is_one_by_convention(self):
        # Both raters give every item the same constant score: expected
        # disagreement is 0; the frozen protocol defines kappa = 1.0 here.
        judge = {"a": 3, "b": 3, "c": 3}
        stats = compute_agreement(judge, dict(judge))
        assert stats["weighted_kappa"] == 1.0

    def test_constant_but_offset_raters_get_zero_kappa(self):
        judge = {"a": 3, "b": 3, "c": 3}
        human = {"a": 4, "b": 4, "c": 4}
        stats = compute_agreement(judge, human)
        assert stats["weighted_kappa"] == pytest.approx(0.0)
        assert stats["exact"] == 0.0
        assert stats["within_one"] == 1.0

    def test_empty_overlap_is_an_error(self):
        with pytest.raises(ValueError, match="no items"):
            compute_agreement({"a": 3}, {"b": 3})

    def test_protocol_is_frozen_in_code(self):
        # R14: the agreement protocol is code-frozen before any judge output
        # is inspected; it names the reported stats and the anchor exclusion.
        for phrase in ("anchor", "within-one", "kappa", "exact"):
            assert phrase in AGREEMENT_PROTOCOL.lower()
        assert JUDGE_STEP == "judge"

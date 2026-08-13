"""Tests for the eval runner and `triage eval` CLI (U7) — R8, R30, R32, KTD5.

Covers:
- Gold-labels loader (U6 data integrity enforced here): CSV schema, label-set
  membership against the single-sourced taxonomy, no duplicate thread ids.
- Baseline arm (R8): ONE call producing all four outputs, its prompt assembled
  from the SAME shared fragments the pipeline steps use (equal information);
  degenerate threads short-circuit with zero LLM calls; persistent bad output
  becomes a typed OutputFailure that is recorded, never raised.
- Runner (R32, KTD5): per-thread checkpoints, a mid-run failure leaves a
  resumable run, resume never re-calls completed threads, results files are
  written only for 100%-complete runs and record model ids + prompt hashes +
  eval-set hash.
- Thread-content integrity (R32): a checkpoint whose thread text has since
  changed is refused on resume and by write_results, so no results file mixes
  two generations of thread text.
- Per-arm usage capture (KTD6): tokens/cost/latency land in every checkpoint,
  with zeros for calls served from the cache.
- CLI `triage eval`: cost preview printed BEFORE executing, --dry-run executes
  nothing, incomplete run exits nonzero with a resume instruction.

Every test is fully mocked and in-process: zero network, no real client.
"""

import json
from types import SimpleNamespace

import pytest
from test_tools import (
    CATEGORIZED,
    DRAFTED,
    ESCALATED,
    FIXTURE_CSV,
    ROUTED,
    ExplodingClient,
    FakeClient,
    make_thread,
    ok,
    validation_error_for,
)

from triage import cli
from triage.evals.judge import judge_plan
from triage.evals.runner import (
    CALLS_PER_THREAD,
    BaselineResult,
    EvalError,
    baseline_system,
    baseline_user_text,
    checkpoint_path,
    completed_thread_ids,
    format_preview,
    load_gold_labels,
    run_baseline,
    run_eval,
    thread_fingerprint,
    threads_digest,
    write_results,
)
from triage.ingest.store import ingest_csv, open_store
from triage.prompts import fragments
from triage.tools import OutputFailure
from triage.tools.retrieval import build_user_text, search_threads

MODEL = "claude-haiku-4-5"

BASE = BaselineResult(
    category="Technical/Product",
    queue="Technical Support",
    escalate=False,
    escalate_reason="Routine technical issue.",
    draft="So sorry about that — we're on it!",
)

# One thread's worth of outcomes in runner call order:
# pipeline (categorize, route, escalate, draft) then baseline.
PER_THREAD_OUTCOMES = [ok(CATEGORIZED), ok(ROUTED), ok(ESCALATED), ok(DRAFTED), ok(BASE)]

LABEL_HEADER = "thread_id,category,queue,escalate\n"


def ok_with_usage(parsed, tokens_in, tokens_out):
    """A scripted response carrying token usage, like the real message object."""
    return SimpleNamespace(
        parsed_output=parsed,
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=tokens_in, output_tokens=tokens_out),
    )


# One thread's worth of usage-bearing outcomes, in runner call order.
USAGE_PER_THREAD = [
    ok_with_usage(CATEGORIZED, 100, 10),
    ok_with_usage(ROUTED, 110, 11),
    ok_with_usage(ESCALATED, 120, 12),
    ok_with_usage(DRAFTED, 130, 13),
    ok_with_usage(BASE, 200, 40),
]
# claude-haiku-4-5 at $1/MTok in, $5/MTok out, hand-computed from the tokens above.
PIPELINE_COST = (460 * 1.0 + 46 * 5.0) / 1_000_000
BASELINE_COST = (200 * 1.0 + 40 * 5.0) / 1_000_000


def make_threads(count):
    """Distinct, diagnosable synthetic threads keyed by thread id."""
    return {
        i: make_thread(
            [f"@brand my order number {1000 + i} arrived broken and shows error code {i}"]
        )
        for i in range(1, count + 1)
    }


def write_labels(path, rows):
    path.write_text(LABEL_HEADER + "".join(f"{r}\n" for r in rows), encoding="utf-8")
    return path


def happy_client(thread_count):
    return FakeClient(list(PER_THREAD_OUTCOMES) * thread_count)


def run_args(tmp_path):
    return {
        "out_dir": tmp_path / "run",
        "cache_path": tmp_path / "cache.db",
        "pipeline_model": MODEL,
        "baseline_model": MODEL,
    }


class TestGoldLabelsLoader:
    def test_shipped_smoke_fixture_passes_the_loader(self):
        # CI's eval dry-run reads this fixture; nothing else did, so it drifted
        # unnoticed through the taxonomy split and only CI went red.
        fixture = FIXTURE_CSV.parent / "smoke_labels.csv"
        labels = load_gold_labels(fixture)
        assert labels

    def test_valid_labels_load(self, tmp_path):
        path = write_labels(
            tmp_path / "gold.csv",
            [
                "1,Technical/Product,Technical Support,true",
                "2,Billing/Payments,Billing Ops,false",
            ],
        )
        labels = load_gold_labels(path)
        assert set(labels) == {1, 2}
        assert labels[1].escalate is True
        assert labels[2].category == "Billing/Payments"

    def test_off_taxonomy_category_is_rejected(self, tmp_path):
        path = write_labels(
            tmp_path / "gold.csv", ["1,Weather,Technical Support,false"]
        )
        with pytest.raises(EvalError, match="Weather"):
            load_gold_labels(path)

    def test_off_taxonomy_queue_is_rejected(self, tmp_path):
        path = write_labels(
            tmp_path / "gold.csv", ["1,Technical/Product,Weather,false"]
        )
        with pytest.raises(EvalError, match="Weather"):
            load_gold_labels(path)

    def test_duplicate_thread_id_is_rejected(self, tmp_path):
        path = write_labels(
            tmp_path / "gold.csv",
            [
                "1,Technical/Product,Technical Support,false",
                "1,Billing/Payments,Billing Ops,true",
            ],
        )
        with pytest.raises(EvalError, match="[Dd]uplicate"):
            load_gold_labels(path)

    def test_missing_column_is_rejected(self, tmp_path):
        path = tmp_path / "gold.csv"
        path.write_text("thread_id,category\n1,Technical/Product\n", encoding="utf-8")
        with pytest.raises(EvalError, match="queue"):
            load_gold_labels(path)

    def test_bad_escalate_value_is_rejected(self, tmp_path):
        path = write_labels(
            tmp_path / "gold.csv", ["1,Technical/Product,Technical Support,maybe"]
        )
        with pytest.raises(EvalError, match="escalate"):
            load_gold_labels(path)

    def test_missing_file_gives_instructive_error(self, tmp_path):
        with pytest.raises(EvalError, match="thread_id"):
            load_gold_labels(tmp_path / "absent.csv")

    def test_utf8_bom_is_tolerated(self, tmp_path):
        # Labeling tools on Windows (Excel, PowerShell) write a UTF-8 BOM.
        path = tmp_path / "gold.csv"
        path.write_text(
            LABEL_HEADER + "1,Technical/Product,Technical Support,false\n",
            encoding="utf-8-sig",
        )
        assert set(load_gold_labels(path)) == {1}


class TestBaselineArm:
    def test_baseline_prompt_reuses_the_shared_fragments(self):
        # R8: equal information — the same preamble, the same four step
        # instructions, and BOTH label spaces, since the baseline emits a
        # category and a queue in one call.
        system = baseline_system()
        assert fragments.SYSTEM_PREAMBLE in system
        assert fragments.category_block() in system
        assert fragments.queue_block() in system
        for instructions in (
            fragments.CATEGORIZE_INSTRUCTIONS,
            fragments.ROUTE_INSTRUCTIONS,
            fragments.ESCALATE_INSTRUCTIONS,
            fragments.DRAFT_INSTRUCTIONS,
        ):
            assert instructions in system


class TestPromptHashIdentity:
    """KTD5 drift detection is only honest if the hashed prompt is the prompt
    the step actually sends; prompt_hashes() rebuilds them independently."""

    def test_hashed_step_prompts_match_what_the_tools_send(self):
        from triage.evals.runner import prompt_hashes, sha256_text
        from triage.tools import categorize, draft, escalate, route

        hashes = prompt_hashes()
        thread = make_thread(["@brand my data plan stopped working with error 105"])

        invocations = {
            "categorize": (CATEGORIZED, lambda c: categorize(thread, model="m", client=c)),
            "route": (ROUTED, lambda c: route(thread, CATEGORIZED, model="m", client=c)),
            "escalate": (
                ESCALATED,
                lambda c: escalate(thread, CATEGORIZED, ROUTED, model="m", client=c),
            ),
            "draft": (
                DRAFTED,
                lambda c: draft(thread, CATEGORIZED, ROUTED, model="m", client=c),
            ),
        }

        for step, (response, call) in invocations.items():
            client = FakeClient([ok(response)])
            call(client)
            system = client.messages.calls[0]["system"]
            assert hashes[step] == sha256_text(system), (
                f"prompt_hashes()[{step!r}] does not hash the prompt the step sends"
            )

    def test_baseline_user_text_matches_pipeline_first_step(self):
        # R8: the baseline sees exactly the thread text the pipeline sees.
        thread = make_threads(1)[1]
        assert baseline_user_text(thread) == build_user_text(thread)

    def test_baseline_is_one_call_with_all_four_outputs(self):
        thread = make_threads(1)[1]
        client = FakeClient([ok(BASE)])
        result = run_baseline(thread, model=MODEL, client=client)
        assert isinstance(result, BaselineResult)
        assert result.category in fragments.CATEGORY_LABELS
        assert result.queue in fragments.QUEUE_LABELS
        assert result.escalate is False
        assert result.escalate_reason
        assert result.draft
        assert result.status == "never_sent"
        assert len(client.messages.calls) == 1  # ONE call for all four outputs
        call = client.messages.calls[0]
        assert call["system"] == baseline_system()
        assert call["output_format"] is BaselineResult

    def test_degenerate_thread_short_circuits_without_llm(self):
        thread = make_thread(["@brand"])
        result = run_baseline(thread, model=MODEL, client=ExplodingClient())
        assert result.category == fragments.GENERAL_INQUIRY
        assert result.escalate is True
        assert "insufficient content" in result.escalate_reason.lower()
        assert result.draft

    def test_persistent_bad_output_becomes_typed_failure(self):
        thread = make_threads(1)[1]
        bad = [validation_error_for(BaselineResult, "{not json") for _ in range(3)]
        result = run_baseline(thread, model=MODEL, client=FakeClient(bad))
        assert isinstance(result, OutputFailure)
        assert result.step == "baseline"


class TestRunnerCheckpointsAndResume:
    def test_mid_run_failure_leaves_resumable_checkpoint(self, tmp_path):
        # R32: thread 1 completes and checkpoints; thread 2 dies on a
        # transport error; the resumed run finishes WITHOUT re-calling
        # thread 1.
        threads = make_threads(3)
        args = run_args(tmp_path)
        failing = FakeClient(
            list(PER_THREAD_OUTCOMES) + [RuntimeError("simulated transport failure")]
        )
        with pytest.raises(RuntimeError, match="transport"):
            run_eval(threads, client=failing, **args)
        assert completed_thread_ids(args["out_dir"]) == {1}

        resumed_client = FakeClient(list(PER_THREAD_OUTCOMES) * 2)
        summary = run_eval(threads, client=resumed_client, **args)
        assert summary == {"completed": 3, "total": 3}
        assert completed_thread_ids(args["out_dir"]) == {1, 2, 3}
        # Only threads 2 and 3 were executed on resume (R32).
        assert len(resumed_client.messages.calls) == 2 * CALLS_PER_THREAD

    def test_output_failure_is_recorded_not_raised(self, tmp_path):
        threads = make_threads(1)
        args = run_args(tmp_path)
        bad = [validation_error_for(CATEGORIZED.__class__, "{oops") for _ in range(3)]
        client = FakeClient(bad + [ok(ROUTED), ok(ESCALATED), ok(DRAFTED), ok(BASE)])
        summary = run_eval(threads, client=client, **args)
        assert summary == {"completed": 1, "total": 1}
        entry = json.loads(
            checkpoint_path(args["out_dir"], 1).read_text(encoding="utf-8")
        )
        assert entry["ok"] is False
        assert entry["pipeline"]["failed_steps"] == ["categorize"]
        assert entry["baseline"]["category"]  # baseline arm still recorded

    def test_resume_refuses_when_the_model_changed(self, tmp_path):
        # A results file is only honest if every checkpoint in it came from the
        # same models and prompts. Resuming a run into an out_dir whose
        # checkpoints were produced by a different model must refuse, not
        # silently mix arms and stamp `complete: true` with the new model ids.
        threads = make_threads(3)
        args = run_args(tmp_path)
        failing = FakeClient(
            list(PER_THREAD_OUTCOMES) + [RuntimeError("simulated transport failure")]
        )
        with pytest.raises(RuntimeError, match="transport"):
            run_eval(threads, client=failing, **args)
        assert completed_thread_ids(args["out_dir"]) == {1}

        measurement_args = {**args, "pipeline_model": "claude-opus-5"}
        with pytest.raises(EvalError, match="pipeline_model"):
            run_eval(threads, client=happy_client(3), **measurement_args)

    def test_resume_refuses_when_a_prompt_changed(self, tmp_path):
        # Same hazard via frozen prompts rather than model ids (KTD5).
        threads = make_threads(1)
        args = run_args(tmp_path)
        run_eval(threads, client=happy_client(1), **args)
        entry = json.loads(checkpoint_path(args["out_dir"], 1).read_text(encoding="utf-8"))
        entry["identity"]["prompt_hashes"]["categorize"] = "stale-hash"
        checkpoint_path(args["out_dir"], 1).write_text(json.dumps(entry), encoding="utf-8")
        with pytest.raises(EvalError, match="prompt_hashes"):
            run_eval(threads, client=happy_client(1), **args)

    def test_write_results_refuses_identity_mismatched_checkpoints(self, tmp_path):
        # Defense in depth: even if a mismatched checkpoint reaches disk, the
        # results file must not be stamped with models that do not describe it.
        threads = make_threads(1)
        args = run_args(tmp_path)
        run_eval(threads, client=happy_client(1), **args)
        labels_path = write_labels(
            tmp_path / "gold.csv", ["1,Technical/Product,Technical Support,false"]
        )
        with pytest.raises(EvalError, match="pipeline_model"):
            write_results(
                args["out_dir"],
                list(threads),
                labels_path=labels_path,
                profile="measurement",
                pipeline_model="claude-opus-5",
                baseline_model=MODEL,
            )
        assert not (args["out_dir"] / "results.json").exists()

    def test_resume_proceeds_when_identity_matches(self, tmp_path):
        # The guard must not break the ordinary resume path.
        threads = make_threads(3)
        args = run_args(tmp_path)
        failing = FakeClient(
            list(PER_THREAD_OUTCOMES) + [RuntimeError("simulated transport failure")]
        )
        with pytest.raises(RuntimeError):
            run_eval(threads, client=failing, **args)
        summary = run_eval(threads, client=FakeClient(list(PER_THREAD_OUTCOMES) * 2), **args)
        assert summary == {"completed": 3, "total": 3}

    def test_write_results_refuses_incomplete_run(self, tmp_path):
        threads = make_threads(3)
        args = run_args(tmp_path)
        failing = FakeClient(list(PER_THREAD_OUTCOMES) + [RuntimeError("boom")])
        with pytest.raises(RuntimeError):
            run_eval(threads, client=failing, **args)
        # R32: metrics input (the results file) only from 100%-complete runs.
        with pytest.raises(EvalError, match="resume"):
            write_results(
                args["out_dir"],
                list(threads),
                labels_path=write_labels(
                    tmp_path / "gold.csv",
                    [f"{i},Technical/Product,Technical Support,false" for i in threads],
                ),
                profile="dev",
                pipeline_model=MODEL,
                baseline_model=MODEL,
            )
        assert not (args["out_dir"] / "results.json").exists()

    def test_full_mocked_eval_over_ten_threads(self, tmp_path):
        # Integration: a complete per-thread results file for BOTH arms.
        threads = make_threads(10)
        args = run_args(tmp_path)
        summary = run_eval(threads, client=happy_client(10), **args)
        assert summary == {"completed": 10, "total": 10}
        labels_path = write_labels(
            tmp_path / "gold.csv",
            [f"{i},Technical/Product,Technical Support,false" for i in threads],
        )
        results_path = write_results(
            args["out_dir"],
            list(threads),
            labels_path=labels_path,
            profile="dev",
            pipeline_model=MODEL,
            baseline_model=MODEL,
            run_id="",
        )
        results = json.loads(results_path.read_text(encoding="utf-8"))
        # KTD5: every results file records the determinism controls.
        assert results["models"] == {"pipeline": MODEL, "baseline": MODEL}
        assert set(results["prompt_hashes"]) == {
            "categorize", "route", "escalate", "draft", "baseline",
        }
        assert results["eval_set_hash"]
        assert results["complete"] is True
        assert len(results["threads"]) == 10
        for entry in results["threads"]:
            assert entry["pipeline"]["steps"]["draft"]["status"] == "never_sent"
            assert entry["baseline"]["status"] == "never_sent"
            assert entry["ok"] is True


class TestThreadContentIntegrity:
    def test_resume_refuses_when_thread_content_changed(self, tmp_path):
        # Ingest gains a previously-missing tweet, then an interrupted run is
        # resumed: mixing two generations of thread text would stamp
        # `complete: true` on outputs the loaded threads do not describe.
        threads = make_threads(2)
        args = run_args(tmp_path)
        failing = FakeClient(
            list(PER_THREAD_OUTCOMES) + [RuntimeError("simulated transport failure")]
        )
        with pytest.raises(RuntimeError, match="transport"):
            run_eval(threads, client=failing, **args)
        assert completed_thread_ids(args["out_dir"]) == {1}

        drifted = {
            **threads,
            1: make_thread(
                [
                    "@brand hi, following up on my earlier complaint",
                    "@brand my order number 1001 arrived broken and shows error code 1",
                ]
            ),
        }
        with pytest.raises(EvalError, match="thread 1"):
            run_eval(drifted, client=happy_client(2), **args)

    def test_write_results_refuses_content_drifted_checkpoints(self, tmp_path):
        threads = make_threads(1)
        args = run_args(tmp_path)
        run_eval(threads, client=happy_client(1), **args)
        labels_path = write_labels(
            tmp_path / "gold.csv", ["1,Technical/Product,Technical Support,false"]
        )
        drifted = {1: make_thread(["@brand a completely different problem, error 42 now"])}
        with pytest.raises(EvalError, match="thread 1"):
            write_results(
                args["out_dir"],
                list(threads),
                labels_path=labels_path,
                profile="measurement",
                pipeline_model=MODEL,
                baseline_model=MODEL,
                threads=drifted,
            )
        assert not (args["out_dir"] / "results.json").exists()

    def test_results_file_surfaces_thread_fingerprints(self, tmp_path):
        threads = make_threads(2)
        args = run_args(tmp_path)
        run_eval(threads, client=happy_client(2), **args)
        labels_path = write_labels(
            tmp_path / "gold.csv",
            [f"{i},Technical/Product,Technical Support,false" for i in threads],
        )
        results_path = write_results(
            args["out_dir"],
            list(threads),
            labels_path=labels_path,
            profile="dev",
            pipeline_model=MODEL,
            baseline_model=MODEL,
            threads=threads,
        )
        results = json.loads(results_path.read_text(encoding="utf-8"))
        assert results["thread_digest"] == threads_digest(
            {tid: thread_fingerprint(thread) for tid, thread in threads.items()}
        )
        for entry in results["threads"]:
            assert entry["thread_fingerprint"] == thread_fingerprint(
                threads[entry["thread_id"]]
            )


class TestUsageCapture:
    def test_usage_is_recorded_per_arm(self, tmp_path):
        threads = make_threads(1)
        args = run_args(tmp_path)
        run_eval(threads, client=FakeClient(list(USAGE_PER_THREAD)), **args)
        usage = json.loads(
            checkpoint_path(args["out_dir"], 1).read_text(encoding="utf-8")
        )["usage"]
        assert usage["pipeline"]["calls"] == 4
        assert usage["baseline"]["calls"] == 1
        assert usage["pipeline"]["tokens_in"] == 460
        assert usage["pipeline"]["tokens_out"] == 46
        assert usage["baseline"]["tokens_in"] == 200
        assert usage["baseline"]["tokens_out"] == 40
        assert usage["pipeline"]["cost"] == pytest.approx(PIPELINE_COST)
        assert usage["baseline"]["cost"] == pytest.approx(BASELINE_COST)
        assert usage["pipeline"]["cached_calls"] == 0
        # Wall-clock timings: relationships only, never a measured duration.
        assert usage["pipeline"]["latency_s"] >= 0.0
        assert usage["baseline"]["latency_s"] >= 0.0

    def test_cache_hits_record_zero_cost(self, tmp_path):
        # A replayed call spends nothing, so the totals must not bill for it.
        threads = make_threads(1)
        args = run_args(tmp_path)
        run_eval(threads, client=FakeClient(list(USAGE_PER_THREAD)), **args)
        replay_args = {**args, "out_dir": tmp_path / "replay"}
        run_eval(threads, client=ExplodingClient(), **replay_args)
        usage = json.loads(
            checkpoint_path(replay_args["out_dir"], 1).read_text(encoding="utf-8")
        )["usage"]
        assert usage["pipeline"]["cached_calls"] == 4
        assert usage["baseline"]["cached_calls"] == 1
        for arm in ("pipeline", "baseline"):
            assert usage[arm]["cost"] == 0
            assert usage[arm]["tokens_in"] == 0
            assert usage[arm]["tokens_out"] == 0


class TestCostPreview:
    def test_judge_calls_are_counted_and_priced_in_the_plan(self):
        # The operator decides to spend on this number (R32), so the judge's
        # calls must be counted and priced, not merely mentioned.
        # Hand-computed at measurement prices: 10 threads x 5 arm calls x
        # ~$0.015 = $0.75; 10 threads x 2 judge calls x ~$0.05 = $1.00.
        text = format_preview(
            10, 10,
            pipeline_model="claude-opus-5",
            baseline_model="claude-opus-5",
            profile="measurement",
            judge=judge_plan("claude-fable-5"),
        )
        assert "= 50 calls" in text
        assert "= 20 calls on claude-fable-5" in text
        assert "~= $0.75" in text
        assert "~= $1.00" in text
        assert "Rough total (both arms + judge): ~$1.75" in text

    def test_judge_is_absent_from_the_plan_when_not_judging(self):
        text = format_preview(
            10, 10, pipeline_model=MODEL, baseline_model=MODEL, profile="dev"
        )
        assert "judge" not in text.lower()


@pytest.fixture
def sample_db(tmp_path):
    db_path = tmp_path / "sample.db"
    ingest_csv(FIXTURE_CSV, db_path)
    return db_path


@pytest.fixture
def eval_setup(sample_db, tmp_path):
    """A store, a two-thread gold-labels file, and eval out/cache paths."""
    conn = open_store(sample_db)
    try:
        ids = [
            search_threads(conn, "data plan")[0]["thread_id"],
            search_threads(conn, "package never arrived")[0]["thread_id"],
        ]
    finally:
        conn.close()
    labels = write_labels(
        tmp_path / "gold.csv",
        [f"{i},Technical/Product,Technical Support,false" for i in ids],
    )
    return {
        "db": sample_db,
        "labels": labels,
        "out": tmp_path / "eval-out",
        "cache": tmp_path / "cache.db",
        "thread_ids": ids,
    }


def eval_argv(setup, *extra):
    return [
        "eval",
        "--labels", str(setup["labels"]),
        "--db", str(setup["db"]),
        "--out", str(setup["out"]),
        "--cache", str(setup["cache"]),
        *extra,
    ]


def run_cli(argv, capsys, client=None):
    code = cli.main(argv, client=client)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class TestEvalCli:
    def test_dry_run_prints_plan_and_executes_nothing(self, eval_setup, capsys):
        # R32: planned call count + rough cost, then exit 0 with zero calls.
        code, out, _ = run_cli(
            eval_argv(eval_setup, "--dry-run"), capsys, client=ExplodingClient()
        )
        assert code == 0
        assert f"= {2 * CALLS_PER_THREAD} calls" in out
        assert "$" in out
        assert MODEL in out  # --profile defaults to dev (R30)
        assert not eval_setup["out"].exists()  # nothing executed

    def test_preview_prints_before_execution(self, eval_setup, capsys):
        code, out, _ = run_cli(
            eval_argv(eval_setup), capsys, client=happy_client(2)
        )
        assert code == 0
        assert out.index("Planned calls") < out.index("Results:")
        results = json.loads(
            (eval_setup["out"] / "results.json").read_text(encoding="utf-8")
        )
        assert len(results["threads"]) == 2
        assert results["complete"] is True

    def test_missing_labels_is_an_instructive_input_error(self, capsys):
        code, _, err = run_cli(["eval", "--dry-run"], capsys, client=ExplodingClient())
        assert code == 2
        assert "--labels" in err

    def test_labeled_thread_missing_from_store_is_an_input_error(
        self, eval_setup, tmp_path, capsys
    ):
        labels = write_labels(
            tmp_path / "bad-gold.csv",
            ["9999,Technical/Product,Technical Support,false"],
        )
        setup = {**eval_setup, "labels": labels}
        code, _, err = run_cli(eval_argv(setup), capsys, client=ExplodingClient())
        assert code == 2
        assert "9999" in err

    def test_ctrl_c_mid_run_still_prints_the_resume_instruction(
        self, eval_setup, capsys
    ):
        # KeyboardInterrupt is not an Exception, so a Ctrl-C during a paid run
        # would otherwise escape the handler as a bare traceback (R32).
        interrupted = FakeClient(list(PER_THREAD_OUTCOMES) + [KeyboardInterrupt()])
        code, _, err = run_cli(eval_argv(eval_setup), capsys, client=interrupted)
        assert code == 1
        assert "resume" in err.lower()
        assert "1/2 threads checkpointed" in err
        assert not (eval_setup["out"] / "results.json").exists()

    def test_eval_arms_the_stores_gold_label_guard(self, eval_setup, capsys):
        # Labels live in a CSV, but ingest can only refuse to rewrite a labeled
        # thread it knows about, so running an eval records them (R11).
        code, _, _ = run_cli(eval_argv(eval_setup), capsys, client=happy_client(2))
        assert code == 0
        conn = open_store(eval_setup["db"])
        try:
            labeled = {row["thread_id"] for row in conn.execute("SELECT thread_id FROM eval_items")}
        finally:
            conn.close()
        assert labeled == set(eval_setup["thread_ids"])

    def test_interrupted_run_exits_nonzero_with_resume_instruction(
        self, eval_setup, capsys
    ):
        # R32 error path: partial run -> nonzero + resume instruction, never a
        # results file; the resumed run completes without re-calling thread 1.
        failing = FakeClient(
            list(PER_THREAD_OUTCOMES) + [RuntimeError("simulated transport failure")]
        )
        code, _, err = run_cli(eval_argv(eval_setup), capsys, client=failing)
        assert code == 1
        assert "resume" in err.lower()
        assert "1/2 threads checkpointed" in err
        assert not (eval_setup["out"] / "results.json").exists()

        resumed = FakeClient(list(PER_THREAD_OUTCOMES))
        code, _out, _ = run_cli(eval_argv(eval_setup), capsys, client=resumed)
        assert code == 0
        assert len(resumed.messages.calls) == CALLS_PER_THREAD  # thread 1 skipped
        assert (eval_setup["out"] / "results.json").exists()

    def test_run_id_salt_is_accepted_and_recorded(self, eval_setup, capsys):
        code, _out, _ = run_cli(
            eval_argv(eval_setup, "--run-id", "variance-1"),
            capsys,
            client=happy_client(2),
        )
        assert code == 0
        results = json.loads(
            (eval_setup["out"] / "results.json").read_text(encoding="utf-8")
        )
        assert results["run_id"] == "variance-1"

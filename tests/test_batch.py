"""Tests for batch-mode eval execution (Messages Batch API, 50% list price).

Wave protocol: threads execute against a collecting client that replays the
shared call cache and records every miss instead of calling the API; each
wave submits the recorded requests as ONE batch, validates results into the
same cache, and re-executes — completed calls replay for free, the next
dependent step becomes the next wave.

Covers:
- Collector: cache hits replay with zero pending; misses collect, dedupe, and
  abort the caller with PendingBatchCall.
- run_eval_batch: a two-thread run completes in the expected number of waves,
  checkpoints match run_eval's shape, and per-arm cost carries the 50% batch
  discount.
- R27 in batch mode: persistently invalid batch output becomes a typed
  OutputFailure with the true failure kind after the same three attempts.
- Transport failures: an entry that keeps erroring raises EvalError; calls
  already answered stay cached, so the re-run is resumable.
- A fully cached run never constructs a batch client (R30).
- score_run_batch: judge scoring of both arms completes in one wave.

Every test is fully mocked and in-process: zero network, no real client.
"""

import json
from types import SimpleNamespace

import pytest
from test_tools import CATEGORIZED, DRAFTED, ESCALATED, ROUTED, make_thread, ok

from triage.evals.batch import (
    BATCH_DISCOUNT,
    BatchCollectingClient,
    PendingBatchCall,
    WaveLedger,
)
from triage.evals.cache import CachingClient, CallCache, call_cost
from triage.evals.judge import DimensionScore, JudgeResult, score_run_batch
from triage.evals.runner import (
    BaselineResult,
    EvalError,
    GoldLabel,
    checkpoint_path,
    run_eval_batch,
    write_results,
)
from triage.tools.llm import call_with_schema
from triage.tools.schemas import FAILURE_MALFORMED, CategorizeResult

MODEL = "claude-haiku-4-5"

BASE = BaselineResult(
    category="Technical/Product",
    queue="Technical Support",
    escalate=False,
    escalate_reason="Routine technical issue.",
    draft="So sorry about that — we're on it!",
)

_DIM = DimensionScore(critique="Solid on this dimension.", score=4)
JUDGED = JudgeResult(
    correctness=_DIM,
    tone=_DIM,
    grounding=_DIM,
    actionability=_DIM,
    send_critique="Sendable with minor polish.",
    send_as_is=True,
)

ANSWERS_BY_TITLE = {
    "CategorizeResult": CATEGORIZED.model_dump_json(),
    "RouteResult": ROUTED.model_dump_json(),
    "EscalateResult": ESCALATED.model_dump_json(),
    "DraftResult": DRAFTED.model_dump_json(),
    "BaselineResult": BASE.model_dump_json(),
    "JudgeResult": JUDGED.model_dump_json(),
}

TOKENS_IN, TOKENS_OUT = 100, 10

ERRORED = object()


class FakeBatches:
    """Scripted messages.batches: answers each request by its schema title."""

    def __init__(self, answer_for=None):
        self.answer_for = answer_for or (lambda title: ANSWERS_BY_TITLE[title])
        self.created = []
        self._results = {}

    def create(self, *, requests):
        self.created.append(list(requests))
        batch_id = f"batch_{len(self.created)}"
        self._results[batch_id] = [self._result_for(r) for r in requests]
        return SimpleNamespace(id=batch_id, processing_status="in_progress")

    def retrieve(self, batch_id):
        return SimpleNamespace(id=batch_id, processing_status="ended")

    def results(self, batch_id):
        return iter(self._results[batch_id])

    def _result_for(self, request):
        title = request["params"]["output_config"]["format"]["schema"]["title"]
        answer = self.answer_for(title)
        if answer is ERRORED:
            return SimpleNamespace(
                custom_id=request["custom_id"],
                result=SimpleNamespace(type="errored", message=None),
            )
        message = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=answer)],
            usage=SimpleNamespace(input_tokens=TOKENS_IN, output_tokens=TOKENS_OUT),
            stop_reason="end_turn",
        )
        return SimpleNamespace(
            custom_id=request["custom_id"],
            result=SimpleNamespace(type="succeeded", message=message),
        )


class FakeBatchClient:
    def __init__(self, answer_for=None):
        self.batches = FakeBatches(answer_for)
        self.messages = SimpleNamespace(batches=self.batches)


class ExplodingBatchClient:
    """Any use is a test failure (asserts a fully cached run stays offline)."""

    @property
    def messages(self):
        raise AssertionError("batch client must not be touched for this run")


def make_threads(count):
    return {
        i: make_thread(
            [f"@brand my order number {1000 + i} arrived broken and shows error code {i}"]
        )
        for i in range(1, count + 1)
    }


def parse_kwargs(user_text="please categorize this"):
    return {
        "model": MODEL,
        "max_tokens": 1024,
        "system": "categorize the thread",
        "messages": [{"role": "user", "content": user_text}],
        "output_format": CategorizeResult,
    }


class TestCollector:
    def test_cache_hit_replays_without_pending(self, tmp_path):
        cache = CallCache(tmp_path / "cache.db")
        kwargs = parse_kwargs()
        seeded = CachingClient(cache, inner=SimpleNamespace(
            messages=SimpleNamespace(parse=lambda **kw: ok(CATEGORIZED))
        ))
        seeded.messages.parse(**kwargs)

        ledger = WaveLedger()
        collector = BatchCollectingClient(cache, ledger)
        message = collector.messages.parse(**kwargs)
        assert message.parsed_output == CATEGORIZED
        assert not ledger.pending

    def test_miss_collects_dedupes_and_raises(self, tmp_path):
        cache = CallCache(tmp_path / "cache.db")
        ledger = WaveLedger()
        collector = BatchCollectingClient(cache, ledger)
        with pytest.raises(PendingBatchCall):
            collector.messages.parse(**parse_kwargs())
        with pytest.raises(PendingBatchCall):
            collector.messages.parse(**parse_kwargs())
        with pytest.raises(PendingBatchCall):
            collector.messages.parse(**parse_kwargs("a different thread"))
        assert len(ledger.pending) == 2


class TestRunEvalBatch:
    def test_happy_run_completes_in_step_waves(self, tmp_path):
        threads = make_threads(2)
        fake = FakeBatchClient()
        summary = run_eval_batch(
            threads,
            out_dir=tmp_path / "out",
            cache_path=tmp_path / "cache.db",
            pipeline_model=MODEL,
            baseline_model=MODEL,
            batch_client=fake,
            poll_seconds=0,
        )
        assert summary == {"completed": 2, "total": 2}
        # Wave 1 collects categorize + baseline for both threads; route,
        # escalate, and draft each need the prior step: four submissions.
        assert len(fake.batches.created) == 4
        assert len(fake.batches.created[0]) == 4  # 2x categorize + 2x baseline

        entry = json.loads(checkpoint_path(tmp_path / "out", 1).read_text(encoding="utf-8"))
        assert entry["ok"] is True
        per_call = call_cost(MODEL, TOKENS_IN, TOKENS_OUT)
        assert entry["usage"]["pipeline"]["cost"] == pytest.approx(
            4 * per_call * BATCH_DISCOUNT
        )
        assert entry["usage"]["baseline"]["cost"] == pytest.approx(
            per_call * BATCH_DISCOUNT
        )
        assert entry["usage"]["pipeline"]["tokens_in"] == 4 * TOKENS_IN

    def test_results_file_assembles_from_batch_checkpoints(self, tmp_path):
        threads = make_threads(1)
        labels = tmp_path / "gold.csv"
        labels.write_text(
            "thread_id,category,queue,escalate\n1,Technical/Product,Technical Support,false\n",
            encoding="utf-8",
        )
        run_eval_batch(
            threads,
            out_dir=tmp_path / "out",
            cache_path=tmp_path / "cache.db",
            pipeline_model=MODEL,
            baseline_model=MODEL,
            batch_client=FakeBatchClient(),
            poll_seconds=0,
        )
        path = write_results(
            tmp_path / "out",
            [1],
            labels_path=labels,
            profile="measurement",
            pipeline_model=MODEL,
            baseline_model=MODEL,
            threads=threads,
        )
        assert json.loads(path.read_text(encoding="utf-8"))["complete"] is True

    def test_persistent_bad_output_records_true_failure_kind(self, tmp_path):
        def answer_for(title):
            if title == "CategorizeResult":
                return "this is not json"
            return ANSWERS_BY_TITLE[title]

        threads = make_threads(1)
        fake = FakeBatchClient(answer_for)
        summary = run_eval_batch(
            threads,
            out_dir=tmp_path / "out",
            cache_path=tmp_path / "cache.db",
            pipeline_model=MODEL,
            baseline_model=MODEL,
            batch_client=fake,
            poll_seconds=0,
        )
        assert summary["completed"] == 1
        entry = json.loads(checkpoint_path(tmp_path / "out", 1).read_text(encoding="utf-8"))
        assert entry["ok"] is False
        failure = entry["pipeline"]["steps"]["categorize"]["failure"]
        assert failure["kind"] == FAILURE_MALFORMED
        assert failure["attempts"] == 3
        # Three submissions of the failing call, no fourth.
        categorize_submissions = sum(
            1
            for wave in fake.batches.created
            for request in wave
            if request["params"]["output_config"]["format"]["schema"]["title"]
            == "CategorizeResult"
        )
        assert categorize_submissions == 3

    def test_erroring_entry_raises_but_leaves_run_resumable(self, tmp_path):
        def broken(title):
            if title == "BaselineResult":
                return ERRORED
            return ANSWERS_BY_TITLE[title]

        threads = make_threads(1)
        with pytest.raises(EvalError):
            run_eval_batch(
                threads,
                out_dir=tmp_path / "out",
                cache_path=tmp_path / "cache.db",
                pipeline_model=MODEL,
                baseline_model=MODEL,
                batch_client=FakeBatchClient(broken),
                poll_seconds=0,
            )
        # The failing run got through escalate before its third baseline
        # error; those answers are cached, so the healthy re-run submits only
        # the never-attempted draft call plus the baseline.
        healthy = FakeBatchClient()
        summary = run_eval_batch(
            threads,
            out_dir=tmp_path / "out",
            cache_path=tmp_path / "cache.db",
            pipeline_model=MODEL,
            baseline_model=MODEL,
            batch_client=healthy,
            poll_seconds=0,
        )
        assert summary["completed"] == 1
        assert len(healthy.batches.created) == 1
        assert sorted(
            request["params"]["output_config"]["format"]["schema"]["title"]
            for request in healthy.batches.created[0]
        ) == ["BaselineResult", "DraftResult"]

    def test_fully_cached_run_never_touches_batch_client(self, tmp_path):
        threads = make_threads(1)
        common = {
            "cache_path": tmp_path / "cache.db",
            "pipeline_model": MODEL,
            "baseline_model": MODEL,
            "poll_seconds": 0,
        }
        run_eval_batch(
            threads, out_dir=tmp_path / "first", batch_client=FakeBatchClient(), **common
        )
        summary = run_eval_batch(
            threads, out_dir=tmp_path / "second", batch_client=ExplodingBatchClient(), **common
        )
        assert summary == {"completed": 1, "total": 1}


class TestScoreRunBatch:
    def test_scores_both_arms_in_one_wave(self, tmp_path):
        results = {
            "threads": [
                {
                    "thread_id": 1,
                    "pipeline": {"steps": {"draft": {"draft": "Pipeline draft text."}}},
                    "baseline": {"draft": "Baseline draft text."},
                }
            ]
        }
        gold = {
            1: GoldLabel(
                thread_id=1,
                category="Technical/Product",
                queue="Technical Support",
                escalate=False,
            )
        }
        fake = FakeBatchClient()
        report = score_run_batch(
            results,
            gold,
            model=MODEL,
            anchors=[],
            out_dir=tmp_path / "judged",
            cache_path=tmp_path / "cache.db",
            batch_client=fake,
            poll_seconds=0,
        )
        assert len(fake.batches.created) == 1
        assert len(fake.batches.created[0]) == 2
        for system in ("pipeline", "baseline"):
            summary = report["systems"][system]
            assert summary["mean_draft_score"] == pytest.approx(4.0)
            assert summary["unscorable"]["count"] == 0


class TestBatchPreview:
    def test_preview_prices_at_half_of_list(self):
        from triage.evals.judge import judge_plan
        from triage.evals.runner import format_preview

        text = format_preview(
            10, 10,
            pipeline_model=MODEL,
            baseline_model=MODEL,
            profile="dev",
            judge=judge_plan(MODEL),
            batch=True,
        )
        # haiku 4.5: arm call ~$0.0030, judge call ~$0.0050 at list price.
        assert "x ~$0.0015" in text
        assert "x ~$0.0025" in text
        assert "Batch API mode" in text

    def test_sync_preview_is_unchanged(self):
        from triage.evals.runner import format_preview

        text = format_preview(
            10, 10, pipeline_model=MODEL, baseline_model=MODEL, profile="dev"
        )
        assert "x ~$0.0030" in text
        assert "Batch API mode" not in text


class TestCollectorFailureReplay:
    def test_exhausted_key_yields_output_failure_via_wrapper(self, tmp_path):
        cache = CallCache(tmp_path / "cache.db")
        ledger = WaveLedger()
        kwargs = parse_kwargs()
        collector = BatchCollectingClient(cache, ledger)
        with pytest.raises(PendingBatchCall) as excinfo:
            collector.messages.parse(**kwargs)
        key = excinfo.value.key
        ledger.attempts[key] = 3
        ledger.bad_payloads[key] = "not json at all"

        result = call_with_schema(
            "categorize",
            model=MODEL,
            system=kwargs["system"],
            user_text="please categorize this",
            schema=CategorizeResult,
            client=collector,
        )
        assert result.kind == FAILURE_MALFORMED
        assert result.attempts == 3

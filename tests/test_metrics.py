"""Tests for eval metrics (U8) — R12, R27, KTD6, AE6.

Every expected accuracy / macro-precision / macro-recall / confusion number in
this file is HAND-COMPUTED from the small synthetic fixture below (per the
plan's execution note), never derived by running the code under test.

Fixture (categorize step), T=Technical/Product B=Billing/Payments S=Shipping/Delivery:

    thread   gold   pipeline   baseline
    1        T      T          T
    2        T      T          B
    3        B      B          B
    4        B      T          B
    5        S      S          T
    6        T      FAIL       B

Pipeline: accuracy 4/6; per class P/R: T 2/3, 2/3; B 1/1, 1/2; S 1/1, 1/1
  -> macro P (2/3+1+1)/3 = 8/9, macro R (2/3+1/2+1)/3 = 13/18; failure rate 1/6.
Baseline: accuracy 3/6; per class P/R: T 1/2, 1/3; B 2/4, 2/2; S 0 (never
  predicted -> precision 0 by convention), 0/1 -> macro P 1/3, macro R 4/9.

Escalate step, gold True on threads 3 and 6:

    thread   gold    pipeline   baseline
    1        False   False      False
    2        False   False      False
    3        True    False      False
    4        False   False      False
    5        False   False      False
    6        True    True       False

Pipeline: accuracy 5/6; P/R: true 1/1, 1/2; false 4/5, 4/4 -> macro P 9/10,
  macro R 3/4. Baseline: accuracy 4/6; P/R: true 0, 0; false 4/6, 4/4
  -> macro P 1/3, macro R 1/2.

Route step: both arms predict every gold queue exactly -> all metrics 1.0.
"""

import json

import pytest
from test_runner import MODEL, USAGE_PER_THREAD, make_threads, run_args, write_labels
from test_tools import FakeClient

from triage.evals.metrics import (
    OUTPUT_FAILURE,
    STEPS,
    accuracy,
    aggregate_usage,
    compute_report,
    paired_bootstrap_delta,
)
from triage.evals.runner import EvalError, GoldLabel, load_gold_labels, run_eval, write_results

T = "Technical/Product"
B = "Billing/Payments"
S = "Shipping/Delivery"

FAIL = None  # in fixture specs: this step's output is a typed OutputFailure

GOLD_SPEC = {
    1: (T, T, False),
    2: (T, T, False),
    3: (B, B, True),
    4: (B, B, False),
    5: (S, S, False),
    6: (T, T, True),
}
PIPELINE_SPEC = {
    1: (T, T, False),
    2: (T, T, False),
    3: (B, B, False),
    4: (T, B, False),
    5: (S, S, False),
    6: (FAIL, T, True),
}
BASELINE_SPEC = {
    1: (T, T, False),
    2: (B, T, False),
    3: (B, B, False),
    4: (B, B, False),
    5: (T, S, False),
    6: (B, T, False),
}


def make_gold(spec=None):
    spec = spec or GOLD_SPEC
    return {
        tid: GoldLabel(thread_id=tid, category=cat, queue=queue, escalate=esc)
        for tid, (cat, queue, esc) in spec.items()
    }


def _failure(step):
    return {"failure": {"step": step, "kind": "malformed", "attempts": 3, "detail": "x"}}


def pipeline_payload(cat, queue, esc):
    steps = {
        "categorize": _failure("categorize") if cat is FAIL else {"label": cat, "rationale": "r"},
        "route": _failure("route") if queue is FAIL else {"queue": queue, "rationale": "r"},
        "escalate": _failure("escalate") if esc is FAIL else {"escalate": esc, "reason": "r"},
        "draft": {"draft": "Sorry about that!", "status": "never_sent"},
    }
    failed = [s for s in ("categorize", "route", "escalate") if "failure" in steps[s]]
    return {"steps": steps, "failed_steps": failed, "ok": not failed}


def baseline_payload(spec):
    if spec is FAIL:
        return _failure("baseline")
    cat, queue, esc = spec
    return {
        "category": cat,
        "queue": queue,
        "escalate": esc,
        "escalate_reason": "r",
        "draft": "Thanks for reaching out!",
        "status": "never_sent",
    }


def make_entry(tid, pipeline_spec, baseline_spec, usage=None):
    entry = {
        "thread_id": tid,
        "run_id": "",
        "pipeline": pipeline_payload(*pipeline_spec),
        "baseline": baseline_payload(baseline_spec),
    }
    entry["ok"] = entry["pipeline"]["ok"] and "failure" not in entry["baseline"]
    if usage is not None:
        entry["usage"] = usage
    return entry


def make_results(pipeline_spec=None, baseline_spec=None, usage_by_tid=None):
    pipeline_spec = pipeline_spec or PIPELINE_SPEC
    baseline_spec = baseline_spec or BASELINE_SPEC
    usage_by_tid = usage_by_tid or {}
    return {
        "run_id": "",
        "profile": "dev",
        "models": {"pipeline": "m", "baseline": "m"},
        "prompt_hashes": {},
        "eval_set_hash": "abc",
        "complete": True,
        "threads": [
            make_entry(tid, pipeline_spec[tid], baseline_spec[tid], usage_by_tid.get(tid))
            for tid in sorted(pipeline_spec)
        ],
    }


@pytest.fixture
def report():
    return compute_report(make_results(), make_gold())


class TestHandComputedClassificationMetrics:
    def test_pipeline_categorize(self, report):
        step = report["systems"]["pipeline"]["steps"]["categorize"]
        assert step["accuracy"] == pytest.approx(4 / 6)
        assert step["macro_precision"] == pytest.approx(8 / 9)
        assert step["macro_recall"] == pytest.approx(13 / 18)

    def test_baseline_categorize(self, report):
        step = report["systems"]["baseline"]["steps"]["categorize"]
        assert step["accuracy"] == pytest.approx(0.5)
        assert step["macro_precision"] == pytest.approx(1 / 3)
        assert step["macro_recall"] == pytest.approx(4 / 9)

    def test_route_is_perfect_for_both_arms(self, report):
        for system in ("pipeline", "baseline"):
            step = report["systems"][system]["steps"]["route"]
            assert step["accuracy"] == 1.0
            assert step["macro_precision"] == 1.0
            assert step["macro_recall"] == 1.0

    def test_escalate_metrics(self, report):
        pipeline = report["systems"]["pipeline"]["steps"]["escalate"]
        assert pipeline["accuracy"] == pytest.approx(5 / 6)
        assert pipeline["macro_precision"] == pytest.approx(9 / 10)
        assert pipeline["macro_recall"] == pytest.approx(3 / 4)
        baseline = report["systems"]["baseline"]["steps"]["escalate"]
        assert baseline["accuracy"] == pytest.approx(2 / 3)
        assert baseline["macro_precision"] == pytest.approx(1 / 3)
        assert baseline["macro_recall"] == pytest.approx(1 / 2)

    def test_per_class_table_with_support(self, report):
        # KTD6: full per-class tables with support, baseline categorize.
        table = report["systems"]["baseline"]["steps"]["categorize"]["per_class"]
        assert table[T] == {
            "precision": pytest.approx(1 / 2),
            "recall": pytest.approx(1 / 3),
            "support": 3,
        }
        assert table[B] == {
            "precision": pytest.approx(1 / 2),
            "recall": pytest.approx(1.0),
            "support": 2,
        }
        # S is never predicted: precision 0.0 by the documented convention.
        assert table[S] == {"precision": 0.0, "recall": 0.0, "support": 1}

    def test_confusion_matrix_for_categorization(self, report):
        # KTD6: confusion matrix, gold -> predicted counts, pipeline arm.
        assert report["confusion"]["pipeline"] == {
            T: {T: 2, OUTPUT_FAILURE: 1},
            B: {B: 1, T: 1},
            S: {S: 1},
        }


class TestOutputFailures:
    def test_failure_counts_as_incorrect_never_excluded(self, report):
        # AE6/R27: thread 6's failed categorize stays in the denominator (6),
        # scored incorrect: accuracy 4/6 (not 4/5).
        step = report["systems"]["pipeline"]["steps"]["categorize"]
        assert step["accuracy"] == pytest.approx(4 / 6)
        assert step["output_failures"] == {
            "failures": 1,
            "total": 6,
            "rate": pytest.approx(1 / 6),
        }

    def test_per_system_failure_rate_is_its_own_line(self, report):
        # R27: each system's output-failure rate reported per system.
        assert report["systems"]["pipeline"]["output_failures"] == {
            "failures": 1,
            "total": 18,
            "rate": pytest.approx(1 / 18),
        }
        assert report["systems"]["baseline"]["output_failures"]["failures"] == 0

    def test_baseline_failure_hits_all_three_steps(self):
        # A failed baseline call has no usable output for any step.
        gold = make_gold({1: (T, T, False), 2: (B, B, True)})
        results = make_results(
            pipeline_spec={1: (T, T, False), 2: (B, B, True)},
            baseline_spec={1: (T, T, False), 2: FAIL},
        )
        report = compute_report(results, gold)
        for step in STEPS:
            stats = report["systems"]["baseline"]["steps"][step]["output_failures"]
            assert stats == {"failures": 1, "total": 2, "rate": pytest.approx(0.5)}
        # Thread 2 is scored incorrect on every step, never dropped (AE6).
        assert report["systems"]["baseline"]["steps"]["escalate"]["accuracy"] == pytest.approx(0.5)


class TestBootstrap:
    def test_ci_contains_observed_delta(self, report):
        delta = report["deltas"]["categorize"]
        assert delta["delta"] == pytest.approx(1 / 6)
        assert delta["ci_low"] <= delta["delta"] <= delta["ci_high"]
        assert -1.0 <= delta["ci_low"] <= delta["ci_high"] <= 1.0

    def test_seeded_bootstrap_is_deterministic(self):
        gold = {tid: cat for tid, (cat, _, _) in GOLD_SPEC.items()}
        pred_a = {1: T, 2: T, 3: B, 4: T, 5: S, 6: OUTPUT_FAILURE}
        pred_b = {1: T, 2: B, 3: B, 4: B, 5: B, 6: T}
        first = paired_bootstrap_delta(gold, pred_a, pred_b, seed=7)
        second = paired_bootstrap_delta(gold, pred_a, pred_b, seed=7)
        assert first == second
        assert first["resamples"] == 1000

    def test_identical_arms_have_zero_delta_ci(self):
        gold = {1: T, 2: B}
        pred = {1: T, 2: T}
        result = paired_bootstrap_delta(gold, pred, dict(pred))
        assert result["delta"] == 0.0
        assert result["ci_low"] == 0.0
        assert result["ci_high"] == 0.0


class TestFlipsAndUsage:
    def test_variance_flip_candidates(self, report):
        flips = report["flips"]["categorize"]
        # Threads 2 and 5: pipeline right, baseline wrong; thread 4 the reverse;
        # thread 6: both wrong -> in neither list.
        assert flips["pipeline_only_correct"] == [2, 5]
        assert flips["baseline_only_correct"] == [4]

    def test_usage_aggregated_when_present(self):
        usage = {
            1: {
                "pipeline": {"tokens_in": 100, "tokens_out": 40, "cost": 0.01, "latency_s": 2.0},
                "baseline": {"tokens_in": 90, "tokens_out": 50, "cost": 0.02, "latency_s": 1.0},
            },
            2: {
                "pipeline": {"tokens_in": 200, "tokens_out": 60, "cost": 0.03, "latency_s": 4.0},
                "baseline": {"tokens_in": 110, "tokens_out": 50, "cost": 0.02, "latency_s": 3.0},
            },
        }
        gold = make_gold({1: (T, T, False), 2: (B, B, False)})
        results = make_results(
            pipeline_spec={1: (T, T, False), 2: (B, B, False)},
            baseline_spec={1: (T, T, False), 2: (B, B, False)},
            usage_by_tid=usage,
        )
        report = compute_report(results, gold)
        assert report["usage"]["pipeline"] == {
            "threads": 2,
            "totals": {"cost": pytest.approx(0.04), "latency_s": 6.0, "tokens_in": 300,
                       "tokens_out": 100},
            "means": {"cost": pytest.approx(0.02), "latency_s": 3.0, "tokens_in": 150.0,
                      "tokens_out": 50.0},
        }

    def test_usage_absent_reports_none(self, report):
        assert report["usage"] == {"pipeline": None, "baseline": None}

    def test_aggregate_usage_ignores_threads_without_usage(self):
        entries = [
            {"thread_id": 1, "usage": {"pipeline": {"cost": 0.5}}},
            {"thread_id": 2},
        ]
        assert aggregate_usage(entries, "pipeline") == {
            "threads": 1,
            "totals": {"cost": 0.5},
            "means": {"cost": 0.5},
        }


class TestUsageOverRealCheckpoints:
    def test_totals_come_from_checkpoints_the_runner_wrote(self, tmp_path):
        # The cost/latency table must be reachable from a real run, not only
        # from hand-built usage blocks: two threads x (4 pipeline + 1 baseline)
        # calls, all token counts hand-summed from USAGE_PER_THREAD.
        threads = make_threads(2)
        args = run_args(tmp_path)
        run_eval(threads, client=FakeClient(list(USAGE_PER_THREAD) * 2), **args)
        labels_path = write_labels(
            tmp_path / "gold.csv",
            [f"{tid},Technical/Product,Technical/Product,false" for tid in threads],
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
        report = compute_report(results, load_gold_labels(labels_path))
        pipeline = report["usage"]["pipeline"]
        assert pipeline["threads"] == 2
        assert pipeline["totals"]["calls"] == 8
        assert pipeline["totals"]["tokens_in"] == 920
        assert pipeline["totals"]["tokens_out"] == 92
        assert pipeline["means"]["tokens_in"] == 460
        assert pipeline["totals"]["cost"] == pytest.approx(2 * (460 + 46 * 5) / 1_000_000)
        baseline = report["usage"]["baseline"]
        assert baseline["totals"]["calls"] == 2
        assert baseline["totals"]["tokens_in"] == 400
        assert baseline["totals"]["cost"] == pytest.approx(2 * (200 + 40 * 5) / 1_000_000)


class TestInputValidation:
    def test_thread_set_mismatch_is_an_error(self):
        gold = make_gold({1: (T, T, False)})
        results = make_results()  # six threads
        with pytest.raises(EvalError, match="gold"):
            compute_report(results, gold)

    def test_accuracy_over_missing_prediction_counts_incorrect(self):
        assert accuracy({1: T, 2: B}, {1: T}) == pytest.approx(0.5)

"""Tests for the regression gate (U8) — R15, KTD10, AE4.

Covers:
- Tolerances per R15: percentage-scale metrics use max(2x spread, 2pp); mean
  draft score uses 2x its own spread with NO fixed floor.
- Failures name the metric; improvements and within-tolerance drops pass.
- Per-class numbers are reported but never gated (AE4).
- Environment drift (model ids, prompt hashes, eval-set hash) emits a hard
  warning distinct from a metric failure.
- Reference artifacts pin model ids, prompt hashes, eval-set hash, metrics,
  spreads, and per-thread outputs; they are recorded ONLY via the explicit
  record-reference command (KTD10).
- CLI: `triage eval --record-reference` / `--regression` wiring, fully mocked.
"""

import json

import pytest
from test_judge import ANCHOR, jr
from test_runner import PER_THREAD_OUTCOMES, write_labels
from test_tools import FIXTURE_CSV, FakeClient, ok

from triage import cli
from triage.evals.metrics import compute_report
from triage.evals.regression import (
    DRAFT_SCORE_METRIC,
    RegressionError,
    check_regression,
    environment_of,
    gated_metrics,
    load_reference,
    record_reference,
    tolerance_for,
)
from triage.ingest.store import ingest_csv, open_store
from triage.tools.retrieval import search_threads

ENV = {
    "models": {"pipeline": "claude-haiku-4-5", "baseline": "claude-haiku-4-5"},
    "prompt_hashes": {"categorize": "h1", "draft": "h2"},
    "eval_set_hash": "e1",
    "thread_digest": "t1",
}


def make_reference(metrics, spreads=None, env=None):
    env = env or ENV
    return {
        "recorded_at": "2026-08-12T00:00:00+00:00",
        "run_id": "",
        "profile": "dev",
        "models": env["models"],
        "prompt_hashes": env["prompt_hashes"],
        "eval_set_hash": env["eval_set_hash"],
        "thread_digest": env["thread_digest"],
        "metrics": dict(metrics),
        "spreads": dict(spreads or {}),
        "per_thread": [],
    }


def failure_metrics(check):
    return [f["metric"] for f in check["failures"]]


class TestTolerances:
    def test_percentage_metric_floor_is_two_points(self):
        # R15: max(2x spread, 2pp) -> spread 0 still tolerates a 2pp drop.
        assert tolerance_for("categorize_accuracy", 0.0) == pytest.approx(0.02)
        assert tolerance_for("route_macro_recall", 0.005) == pytest.approx(0.02)

    def test_percentage_metric_spread_widens_tolerance(self):
        assert tolerance_for("categorize_accuracy", 0.03) == pytest.approx(0.06)

    def test_draft_score_has_no_floor(self):
        # R15: 2x its own observed spread on the 1-5 scale — no fixed floor.
        assert tolerance_for(DRAFT_SCORE_METRIC, 0.0) == 0.0
        assert tolerance_for(DRAFT_SCORE_METRIC, 0.1) == pytest.approx(0.2)


class TestCheckRegression:
    def test_within_tolerance_passes(self):
        reference = make_reference({"categorize_accuracy": 0.90})
        check = check_regression({"categorize_accuracy": 0.89}, reference, current_env=ENV)
        assert check["ok"] is True
        assert check["failures"] == []
        assert check["drift"] == []

    def test_drop_beyond_tolerance_fails_naming_the_metric(self):
        reference = make_reference({"categorize_accuracy": 0.90})
        check = check_regression({"categorize_accuracy": 0.87}, reference, current_env=ENV)
        assert check["ok"] is False
        assert failure_metrics(check) == ["categorize_accuracy"]
        assert "categorize_accuracy" in check["failures"][0]["message"]
        assert check["drift"] == []  # a metric failure is not drift

    def test_recorded_spread_widens_the_band(self):
        reference = make_reference(
            {"categorize_accuracy": 0.90}, spreads={"categorize_accuracy": 0.03}
        )
        check = check_regression({"categorize_accuracy": 0.85}, reference, current_env=ENV)
        assert check["ok"] is True  # drop 0.05 <= tolerance 0.06

    def test_draft_score_gated_with_no_floor(self):
        reference = make_reference({DRAFT_SCORE_METRIC: 4.0})
        check = check_regression({DRAFT_SCORE_METRIC: 3.99}, reference, current_env=ENV)
        assert failure_metrics(check) == [DRAFT_SCORE_METRIC]
        reference = make_reference({DRAFT_SCORE_METRIC: 4.0}, spreads={DRAFT_SCORE_METRIC: 0.1})
        check = check_regression({DRAFT_SCORE_METRIC: 3.85}, reference, current_env=ENV)
        assert check["ok"] is True  # drop 0.15 <= 2 x 0.1

    def test_improvement_passes(self):
        reference = make_reference({"escalate_accuracy": 0.80})
        check = check_regression({"escalate_accuracy": 0.95}, reference, current_env=ENV)
        assert check["ok"] is True

    def test_missing_gated_metric_fails_naming_it(self):
        reference = make_reference({DRAFT_SCORE_METRIC: 4.0})
        check = check_regression({}, reference, current_env=ENV)
        assert failure_metrics(check) == [DRAFT_SCORE_METRIC]

    def test_drift_is_a_warning_distinct_from_metric_failure(self):
        # A drifted prompt hash triggers the environment warning, NOT a
        # metric failure (R15, KTD10).
        reference = make_reference({"categorize_accuracy": 0.90})
        drifted_env = {**ENV, "prompt_hashes": {"categorize": "CHANGED", "draft": "h2"}}
        check = check_regression({"categorize_accuracy": 0.90}, reference,
                                 current_env=drifted_env)
        assert check["failures"] == []
        assert len(check["drift"]) == 1
        assert "prompt_hashes" in check["drift"][0]
        assert check["ok"] is False

    def test_all_drift_axes_are_checked(self):
        # thread_digest is an axis because ingest can change the thread text
        # under a run without touching models, prompts, or the labels file.
        reference = make_reference({"categorize_accuracy": 0.90})
        drifted_env = {
            "models": {"pipeline": "other-model", "baseline": "claude-haiku-4-5"},
            "prompt_hashes": {"categorize": "CHANGED", "draft": "h2"},
            "eval_set_hash": "e2",
            "thread_digest": "CHANGED",
        }
        check = check_regression({"categorize_accuracy": 0.90}, reference,
                                 current_env=drifted_env)
        drift_text = "\n".join(check["drift"])
        for axis in ("models", "prompt_hashes", "eval_set_hash", "thread_digest"):
            assert axis in drift_text


class TestGatedSet:
    @pytest.fixture
    def report(self):
        from test_metrics import make_gold, make_results

        return compute_report(make_results(), make_gold())

    def test_gated_set_is_per_step_accuracy_and_macro_prf(self, report):
        gated = gated_metrics(report)
        assert set(gated) == {
            f"{step}_{metric}"
            for step in ("categorize", "route", "escalate")
            for metric in ("accuracy", "macro_precision", "macro_recall")
        }
        assert gated["categorize_accuracy"] == pytest.approx(4 / 6)

    def test_mean_draft_score_joins_the_gate_when_supplied(self, report):
        gated = gated_metrics(report, mean_draft_score=3.75)
        assert gated[DRAFT_SCORE_METRIC] == 3.75

    def test_per_class_numbers_are_never_gated(self, report):
        # AE4: per-class tables are reported in the metrics report but no
        # per-class name enters the gated set, so a per-class dip alone
        # cannot fail the check.
        gated = gated_metrics(report, mean_draft_score=4.0)
        assert not any("per_class" in name or "/" in name for name in gated)
        reference = make_reference(gated)
        check = check_regression(gated, reference, current_env=ENV)
        assert check["ok"] is True


REFERENCE_RESULTS = {
    "run_id": "ref-run",
    "profile": "dev",
    "models": ENV["models"],
    "prompt_hashes": ENV["prompt_hashes"],
    "eval_set_hash": ENV["eval_set_hash"],
    "thread_digest": ENV["thread_digest"],
    "threads": [{"thread_id": 1, "pipeline": {}, "baseline": {}, "ok": True}],
}


REFERENCE_METRICS = {"categorize_accuracy": 0.9, DRAFT_SCORE_METRIC: 4.0}
REFERENCE_SPREADS = {"categorize_accuracy": 0.01, DRAFT_SCORE_METRIC: 0.1}


class TestReferenceArtifact:
    def test_record_reference_pins_everything(self, tmp_path):
        path = record_reference(
            REFERENCE_RESULTS,
            REFERENCE_METRICS,
            spreads=REFERENCE_SPREADS,
            path=tmp_path / "reference.json",
        )
        reference = json.loads(path.read_text(encoding="utf-8"))
        # KTD10: model IDs, prompt hashes, eval-set hash, per-thread outputs.
        assert reference["models"] == ENV["models"]
        assert reference["prompt_hashes"] == ENV["prompt_hashes"]
        assert reference["eval_set_hash"] == "e1"
        assert reference["metrics"] == REFERENCE_METRICS
        assert reference["spreads"] == REFERENCE_SPREADS
        assert reference["per_thread"] == REFERENCE_RESULTS["threads"]
        assert reference["recorded_at"]

    def test_load_reference_roundtrip(self, tmp_path):
        path = record_reference(
            REFERENCE_RESULTS,
            REFERENCE_METRICS,
            spreads=REFERENCE_SPREADS,
            path=tmp_path / "reference.json",
        )
        reference = load_reference(path)
        assert reference["metrics"] == REFERENCE_METRICS

    def test_record_reference_refuses_empty_spreads(self, tmp_path):
        # R15/R21: with no measured spread every percentage tolerance collapses
        # to the 2pp floor and the draft-score band collapses to zero.
        with pytest.raises(RegressionError, match="spread"):
            record_reference(
                REFERENCE_RESULTS, REFERENCE_METRICS, path=tmp_path / "reference.json"
            )
        assert not (tmp_path / "reference.json").exists()

    def test_record_reference_refuses_metrics_without_the_draft_score(self, tmp_path):
        # A reference without mean_draft_score makes the draft gate vacuous.
        with pytest.raises(RegressionError, match=DRAFT_SCORE_METRIC):
            record_reference(
                REFERENCE_RESULTS,
                {"categorize_accuracy": 0.9},
                spreads=REFERENCE_SPREADS,
                path=tmp_path / "reference.json",
            )
        assert not (tmp_path / "reference.json").exists()

    def test_load_reference_refuses_a_reference_that_cannot_gate_drafts(self, tmp_path):
        path = tmp_path / "old-reference.json"
        stale = {
            **make_reference({"categorize_accuracy": 0.9}, spreads={"categorize_accuracy": 0.01})
        }
        path.write_text(json.dumps(stale), encoding="utf-8")
        with pytest.raises(RegressionError, match=DRAFT_SCORE_METRIC):
            load_reference(path)

    def test_load_missing_reference_names_the_explicit_command(self, tmp_path):
        with pytest.raises(RegressionError, match="record-reference"):
            load_reference(tmp_path / "absent.json")

    def test_load_rejects_corrupt_or_incomplete_artifacts(self, tmp_path):
        corrupt = tmp_path / "corrupt.json"
        corrupt.write_text("{not json", encoding="utf-8")
        with pytest.raises(RegressionError, match="not valid JSON"):
            load_reference(corrupt)
        incomplete = tmp_path / "incomplete.json"
        incomplete.write_text(json.dumps({"metrics": {}}), encoding="utf-8")
        with pytest.raises(RegressionError, match="missing"):
            load_reference(incomplete)

    def test_environment_of_extracts_the_drift_axes(self):
        assert environment_of(REFERENCE_RESULTS) == ENV


# ---------------------------------------------------------------------------
# CLI wiring: triage eval --record-reference / --regression (mocked client).
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_setup(tmp_path):
    db = tmp_path / "sample.db"
    ingest_csv(FIXTURE_CSV, db)
    conn = open_store(db)
    try:
        ids = [
            search_threads(conn, "data plan")[0]["thread_id"],
            search_threads(conn, "package never arrived")[0]["thread_id"],
        ]
    finally:
        conn.close()
    # Gold: thread 1 Technical/Product, thread 2 Billing/Payments; the mocked
    # arms always predict Technical/Product -> categorize/route accuracy 0.5.
    labels = write_labels(
        tmp_path / "gold.csv",
        [
            f"{ids[0]},Technical/Product,Technical Support,false",
            f"{ids[1]},Billing/Payments,Billing Ops,false",
        ],
    )
    anchors = tmp_path / "anchors.json"
    anchors.write_text(
        json.dumps([{
            "item_id": ANCHOR.item_id,
            "category": ANCHOR.category,
            "draft": ANCHOR.draft,
            "scores": dict(ANCHOR.scores),
            "critiques": dict(ANCHOR.critiques),
            "send_as_is": ANCHOR.send_as_is,
        }]),
        encoding="utf-8",
    )
    spreads = tmp_path / "spreads.json"
    spreads.write_text(
        json.dumps({"categorize_accuracy": 0.02, DRAFT_SCORE_METRIC: 0.1}), encoding="utf-8"
    )
    return {
        "labels": labels,
        "db": db,
        "out": tmp_path / "eval-out",
        "cache": tmp_path / "cache.db",
        "reference": tmp_path / "reference.json",
        "anchors": anchors,
        "spreads": spreads,
        "tmp": tmp_path,
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


def judged_argv(setup, *extra):
    """The gate as it is actually operated: judging on, spreads supplied."""
    return eval_argv(
        setup,
        "--judge",
        "--anchors", str(setup["anchors"]),
        "--spreads", str(setup["spreads"]),
        *extra,
    )


def run_cli(argv, capsys, client):
    code = cli.main(argv, client=client)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def happy_client(thread_count):
    return FakeClient(list(PER_THREAD_OUTCOMES) * thread_count)


def judged_client(thread_count):
    """Both eval arms, then one judge call per (thread, arm)."""
    return FakeClient(
        list(PER_THREAD_OUTCOMES) * thread_count + [ok(jr(4, 4, 4, 4))] * (2 * thread_count)
    )


class TestEvalCliRegression:
    def test_plain_eval_never_records_a_reference(self, cli_setup, capsys):
        # KTD10: a reference is recorded ONLY via the explicit command.
        code, _, _ = run_cli(eval_argv(cli_setup), capsys, happy_client(2))
        assert code == 0
        assert not cli_setup["reference"].exists()

    def test_record_then_regression_passes(self, cli_setup, capsys):
        code, out, _ = run_cli(
            judged_argv(cli_setup, "--record-reference", str(cli_setup["reference"])),
            capsys,
            judged_client(2),
        )
        assert code == 0
        assert "Reference recorded" in out
        reference = json.loads(cli_setup["reference"].read_text(encoding="utf-8"))
        assert reference["metrics"]["categorize_accuracy"] == pytest.approx(0.5)
        assert len(reference["per_thread"]) == 2

        # Re-run against the reference: checkpoints make it call-free.
        code, out, err = run_cli(
            judged_argv(cli_setup, "--regression", str(cli_setup["reference"])),
            capsys,
            FakeClient([]),
        )
        assert code == 0
        assert "passed" in out.lower()
        assert err == ""
        # The pass did not silently re-record the reference (KTD10).
        assert json.loads(cli_setup["reference"].read_text(encoding="utf-8")) == reference

    def test_regression_failure_names_the_metric_and_exits_nonzero(self, cli_setup, capsys):
        code, _, _ = run_cli(
            judged_argv(cli_setup, "--record-reference", str(cli_setup["reference"])),
            capsys,
            judged_client(2),
        )
        assert code == 0
        reference = json.loads(cli_setup["reference"].read_text(encoding="utf-8"))
        reference["metrics"]["categorize_accuracy"] = 0.9  # pretend a better past
        cli_setup["reference"].write_text(json.dumps(reference), encoding="utf-8")

        code, _, err = run_cli(
            judged_argv(cli_setup, "--regression", str(cli_setup["reference"])),
            capsys,
            FakeClient([]),
        )
        assert code == 1
        assert "categorize_accuracy" in err
        assert "drift" not in err.lower()

    def test_drifted_reference_emits_environment_warning_not_metric_failure(
        self, cli_setup, capsys
    ):
        code, _, _ = run_cli(
            judged_argv(cli_setup, "--record-reference", str(cli_setup["reference"])),
            capsys,
            judged_client(2),
        )
        assert code == 0
        reference = json.loads(cli_setup["reference"].read_text(encoding="utf-8"))
        reference["prompt_hashes"]["categorize"] = "0" * 64  # drifted prompt
        cli_setup["reference"].write_text(json.dumps(reference), encoding="utf-8")

        code, _, err = run_cli(
            judged_argv(cli_setup, "--regression", str(cli_setup["reference"])),
            capsys,
            FakeClient([]),
        )
        assert code == 1
        assert "environment drift" in err.lower()
        assert "prompt_hashes" in err
        assert "regression:" not in err  # distinct from a metric failure

    def test_missing_reference_is_an_input_error(self, cli_setup, capsys):
        code, _, err = run_cli(
            eval_argv(cli_setup, "--regression", str(cli_setup["reference"])),
            capsys,
            happy_client(2),
        )
        assert code == 2
        assert "record-reference" in err


class TestEvalCliDraftGate:
    """Finding (f): the draft-quality gate is wired, or it fails loudly."""

    def test_record_reference_without_the_judge_is_refused_before_spending(
        self, cli_setup, capsys
    ):
        # Recording without a judge-derived mean_draft_score would produce a
        # reference whose draft gate silently does not exist.
        code, _, err = run_cli(
            eval_argv(cli_setup, "--record-reference", str(cli_setup["reference"])),
            capsys,
            FakeClient([]),  # refused before any call is made
        )
        assert code == 2
        assert "--judge" in err
        assert DRAFT_SCORE_METRIC in err
        assert not cli_setup["reference"].exists()

    def test_judging_requires_the_held_out_anchor_stock(self, cli_setup, capsys):
        code, _, err = run_cli(
            eval_argv(cli_setup, "--judge"), capsys, FakeClient([])
        )
        assert code == 2
        assert "--anchors" in err

    def test_record_reference_without_spreads_is_refused(self, cli_setup, capsys):
        code, _, err = run_cli(
            eval_argv(
                cli_setup,
                "--judge", "--anchors", str(cli_setup["anchors"]),
                "--record-reference", str(cli_setup["reference"]),
            ),
            capsys,
            FakeClient([]),
        )
        assert code == 2
        assert "--spreads" in err

    def test_regression_against_a_draft_gated_reference_requires_the_judge(
        self, cli_setup, capsys
    ):
        code, _, _ = run_cli(
            judged_argv(cli_setup, "--record-reference", str(cli_setup["reference"])),
            capsys,
            judged_client(2),
        )
        assert code == 0
        # Without --judge the current run has no draft score at all: refuse up
        # front rather than after paying for a whole eval run.
        code, _, err = run_cli(
            eval_argv(cli_setup, "--regression", str(cli_setup["reference"])),
            capsys,
            FakeClient([]),
        )
        assert code == 2
        assert DRAFT_SCORE_METRIC in err
        assert "--judge" in err

    def test_judged_reference_carries_the_draft_score_and_measured_spreads(
        self, cli_setup, capsys
    ):
        code, out, _ = run_cli(
            judged_argv(cli_setup, "--record-reference", str(cli_setup["reference"])),
            capsys,
            judged_client(2),
        )
        assert code == 0
        reference = json.loads(cli_setup["reference"].read_text(encoding="utf-8"))
        # Every mocked judge score is 4 -> mean draft score 4.0 (R13).
        assert reference["metrics"][DRAFT_SCORE_METRIC] == pytest.approx(4.0)
        assert reference["spreads"][DRAFT_SCORE_METRIC] == pytest.approx(0.1)
        assert reference["spreads"]["categorize_accuracy"] == pytest.approx(0.02)
        assert "mean draft score" in out.lower()
        assert "unscorable" in out.lower()  # R27 disclosure

    def test_draft_score_drop_fails_the_gate_naming_the_metric(self, cli_setup, capsys):
        code, _, _ = run_cli(
            judged_argv(cli_setup, "--record-reference", str(cli_setup["reference"])),
            capsys,
            judged_client(2),
        )
        assert code == 0
        reference = json.loads(cli_setup["reference"].read_text(encoding="utf-8"))
        reference["metrics"][DRAFT_SCORE_METRIC] = 5.0  # pretend a better past
        cli_setup["reference"].write_text(json.dumps(reference), encoding="utf-8")

        code, _, err = run_cli(
            judged_argv(cli_setup, "--regression", str(cli_setup["reference"])),
            capsys,
            FakeClient([]),
        )
        # Drop 1.0 > tolerance 2 x 0.1 (R15, no floor for the draft score).
        assert code == 1
        assert DRAFT_SCORE_METRIC in err

    def test_dry_run_counts_and_prices_the_judge_calls(self, cli_setup, capsys):
        plain_code, plain_out, _ = run_cli(
            eval_argv(cli_setup, "--dry-run"), capsys, FakeClient([])
        )
        code, out, _ = run_cli(
            judged_argv(cli_setup, "--dry-run"), capsys, FakeClient([])
        )
        assert plain_code == code == 0
        assert "judge" in out.lower()
        assert "= 4 calls" in out  # 2 threads x 2 arms
        assert "Rough total" in out
        assert "judge" not in plain_out.lower()  # opt-in only: never priced by default

    def test_spreads_are_derived_from_variance_pass_results(self, cli_setup, capsys):
        code, _, _ = run_cli(judged_argv(cli_setup), capsys, judged_client(2))
        assert code == 0
        results = json.loads(
            (cli_setup["out"] / "results.json").read_text(encoding="utf-8")
        )
        passes = []
        for index, category in enumerate(("Technical/Product", "Billing/Payments")):
            directory = cli_setup["tmp"] / f"variance-{index}"
            directory.mkdir()
            pass_results = json.loads(json.dumps(results))
            # Pass 2 gets thread 2's categorization right: accuracy 0.5 vs 1.0.
            pass_results["threads"][1]["pipeline"]["steps"]["categorize"]["label"] = category
            path = directory / "results.json"
            path.write_text(json.dumps(pass_results), encoding="utf-8")
            passes.append(str(path))

        code, _, err = run_cli(
            eval_argv(
                cli_setup,
                "--judge", "--anchors", str(cli_setup["anchors"]),
                "--variance-results", passes[0],
                "--variance-results", passes[1],
                "--record-reference", str(cli_setup["reference"]),
            ),
            capsys,
            FakeClient([]),  # every call replays from checkpoints and the cache
        )
        assert code == 0, err
        reference = json.loads(cli_setup["reference"].read_text(encoding="utf-8"))
        assert reference["spreads"]["categorize_accuracy"] == pytest.approx(0.5)
        assert reference["spreads"][DRAFT_SCORE_METRIC] == pytest.approx(0.0)

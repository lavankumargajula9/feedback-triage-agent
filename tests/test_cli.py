"""Tests for the CLI run surface (U4) — R7, R28, R31, AE1, AE2.

Covers:
- `triage run <tweet_id>`: loads the store, resolves the thread (R31), emits
  the structured JSON result with all four step outputs (AE1); any tweet id in
  a thread resolves to the same thread.
- Exit codes (R7): 0 success, 1 pipeline failure (failing step named),
  2 input error (unknown id, missing store, malformed --input JSON).
- `--input file.json`: builds a Thread in memory from {messages: [...]};
  degenerate input still yields the R28 result with zero LLM calls.
- `--batch`: continues past per-thread failures, emits per-thread results,
  exits nonzero if any failed.
- `triage ingest` keeps working alongside the new subcommand.

Every test is fully mocked and in-process: zero network, no real client.
"""

import json

import pytest
from test_tools import (
    DRAFTED,
    ESCALATED,
    FIXTURE_CSV,
    HAPPY_OUTCOMES,
    ROUTED,
    ExplodingClient,
    FakeClient,
    ok,
    validation_error_for,
)

from triage import cli
from triage.ingest.store import ingest_csv
from triage.tools import CategorizeResult

INLINE_MESSAGES = {
    "messages": [
        {
            "author": "100042",
            "inbound": True,
            "text": "@brand my order arrived broken and support has not replied in a week",
            "created_at": "Tue Oct 31 22:00:00 +0000 2017",
        },
        {
            "author": "brand",
            "inbound": False,
            "text": "@100042 so sorry! Can you share the order number?",
            "created_at": "Tue Oct 31 22:05:00 +0000 2017",
        },
    ]
}


@pytest.fixture
def sample_db(tmp_path):
    db_path = tmp_path / "sample.db"
    ingest_csv(FIXTURE_CSV, db_path)
    return db_path


def run_cli(argv, capsys, client=None):
    code = cli.main(argv, client=client)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class TestRunSingle:
    def test_happy_path_emits_all_four_steps(self, sample_db, capsys):
        # AE1 via the CLI: category, queue, escalation + reason, draft — each
        # attributable to its own step in the JSON output.
        client = FakeClient(list(HAPPY_OUTCOMES))
        code, out, _ = run_cli(["run", "1", "--db", str(sample_db)], capsys, client=client)
        assert code == 0
        result = json.loads(out)
        steps = result["steps"]
        assert steps["categorize"]["label"] == "Technical/Product"
        assert steps["route"]["queue"] == "Technical Support"
        assert steps["escalate"]["escalate"] is False
        assert steps["escalate"]["reason"]
        assert steps["draft"]["status"] == "never_sent"
        assert result["ok"] is True

    def test_any_tweet_id_resolves_to_the_same_thread(self, sample_db, capsys):
        # R31: running on tweet 1 (root) and tweet 3 (mid-thread) triages the
        # same reconstructed thread.
        code_a, out_a, _ = run_cli(
            ["run", "1", "--db", str(sample_db)], capsys, client=FakeClient(list(HAPPY_OUTCOMES))
        )
        code_b, out_b, _ = run_cli(
            ["run", "3", "--db", str(sample_db)], capsys, client=FakeClient(list(HAPPY_OUTCOMES))
        )
        assert code_a == code_b == 0
        assert json.loads(out_a)["thread"]["tweet_ids"] == json.loads(out_b)["thread"]["tweet_ids"]

    def test_unknown_tweet_id_exits_2(self, sample_db, capsys):
        code, _, err = run_cli(
            ["run", "9999", "--db", str(sample_db)], capsys, client=ExplodingClient()
        )
        assert code == 2
        assert "9999" in err

    def test_shared_tweet_resolves_to_its_own_authors_thread(self, sample_db, capsys):
        # Fixture tweet 21 is a brand reply shared by threads 4 (customer
        # 100005) and 5 (customer 100006). Tweet 22 is 100006's own, so it must
        # reach 100006's thread rather than the lowest-id one.
        client = FakeClient(list(HAPPY_OUTCOMES))
        code, out, err = run_cli(
            ["run", "22", "--db", str(sample_db)], capsys, client=client
        )
        assert code == 0, err
        assert json.loads(out)["thread"]["tweet_ids"] == [20, 21, 22]

    def test_shared_tweet_without_an_authoring_customer_is_deterministic(
        self, sample_db, capsys
    ):
        # Tweet 21 itself is brand-authored, so no customer owns it; resolution
        # must still be stable rather than order-dependent.
        first, second = (
            json.loads(
                run_cli(
                    ["run", "21", "--db", str(sample_db)],
                    capsys,
                    client=FakeClient(list(HAPPY_OUTCOMES)),
                )[1]
            )["thread"]["tweet_ids"]
            for _ in range(2)
        )
        assert first == second

    def test_missing_store_exits_2(self, tmp_path, capsys):
        code, _, err = run_cli(
            ["run", "1", "--db", str(tmp_path / "absent.db")], capsys, client=ExplodingClient()
        )
        assert code == 2
        assert "ingest" in err

    def test_pipeline_failure_exits_1_and_names_step(self, sample_db, capsys):
        # R7: node-level OutputFailure -> exit 1; the JSON names the failing step.
        bad = [validation_error_for(CategorizeResult, "{not json") for _ in range(3)]
        client = FakeClient(bad + [ok(ROUTED), ok(ESCALATED), ok(DRAFTED)])
        code, out, _ = run_cli(["run", "1", "--db", str(sample_db)], capsys, client=client)
        assert code == 1
        result = json.loads(out)
        assert result["ok"] is False
        assert result["failed_steps"] == ["categorize"]
        assert result["steps"]["categorize"]["failure"]["step"] == "categorize"

    def test_requires_exactly_one_input_source(self, sample_db, capsys):
        code, _, err = run_cli(["run"], capsys)
        assert code == 2
        assert "exactly one" in err
        code, _, err = run_cli(
            ["run", "1", "--input", "x.json", "--db", str(sample_db)], capsys
        )
        assert code == 2


class TestRunInput:
    def test_input_file_builds_thread_in_memory(self, tmp_path, capsys):
        input_path = tmp_path / "thread.json"
        input_path.write_text(json.dumps(INLINE_MESSAGES), encoding="utf-8")
        client = FakeClient(list(HAPPY_OUTCOMES))
        code, out, _ = run_cli(["run", "--input", str(input_path)], capsys, client=client)
        assert code == 0
        result = json.loads(out)
        assert result["steps"]["categorize"]["label"] == "Technical/Product"
        assert result["steps"]["draft"]["status"] == "never_sent"
        # The in-memory thread text reached the model.
        assert "order arrived broken" in client.messages.calls[0]["messages"][0]["content"]

    def test_malformed_input_json_exits_2(self, tmp_path, capsys):
        input_path = tmp_path / "bad.json"
        input_path.write_text("{not json", encoding="utf-8")
        code, _, err = run_cli(["run", "--input", str(input_path)], capsys)
        assert code == 2
        assert "JSON" in err

    def test_input_missing_message_key_exits_2(self, tmp_path, capsys):
        input_path = tmp_path / "incomplete.json"
        payload = {"messages": [{"author": "1", "inbound": True, "text": "hi there friend"}]}
        input_path.write_text(json.dumps(payload), encoding="utf-8")
        code, _, err = run_cli(["run", "--input", str(input_path)], capsys)
        assert code == 2
        assert "created_at" in err

    def test_degenerate_input_yields_r28_result_with_zero_llm_calls(self, tmp_path, capsys):
        # R28 reachability via the CLI: a degenerate thread still produces the
        # valid General-Inquiry/escalated result without any LLM call.
        input_path = tmp_path / "degenerate.json"
        payload = {
            "messages": [
                {
                    "author": "100042",
                    "inbound": True,
                    "text": "@brand help",
                    "created_at": "Tue Oct 31 22:00:00 +0000 2017",
                }
            ]
        }
        input_path.write_text(json.dumps(payload), encoding="utf-8")
        code, out, _ = run_cli(
            ["run", "--input", str(input_path)], capsys, client=ExplodingClient()
        )
        assert code == 0
        result = json.loads(out)
        assert result["steps"]["categorize"]["label"] == "General Inquiry"
        assert result["steps"]["escalate"]["escalate"] is True
        assert result["steps"]["draft"]["status"] == "never_sent"


class TestBatch:
    def test_batch_file_with_a_single_tweet_id_is_accepted(self, sample_db, tmp_path, capsys):
        # A one-line batch file parses as bare JSON (the int 1), so it must not
        # be rejected as "not a non-empty array".
        batch_path = tmp_path / "batch.txt"
        batch_path.write_text("1\n", encoding="utf-8")
        client = FakeClient(list(HAPPY_OUTCOMES))
        code, out, err = run_cli(
            ["run", "--batch", str(batch_path), "--db", str(sample_db)], capsys, client=client
        )
        assert code == 0, err
        entries = json.loads(out)
        assert len(entries) == 1
        assert entries[0]["input"] == 1
        assert entries[0]["result"]["steps"]["categorize"]["label"] == "Technical/Product"

    def test_batch_continues_past_poisoned_thread_and_exits_nonzero(
        self, sample_db, tmp_path, capsys
    ):
        # R7: the unknown id in the middle does not stop the batch; the run
        # emits per-thread results and exits nonzero because one failed.
        batch_path = tmp_path / "batch.txt"
        batch_path.write_text("1\n9999\n20\n", encoding="utf-8")
        client = FakeClient(list(HAPPY_OUTCOMES) * 2)
        code, out, _ = run_cli(
            ["run", "--batch", str(batch_path), "--db", str(sample_db)], capsys, client=client
        )
        assert code != 0
        items = json.loads(out)
        assert len(items) == 3
        assert items[0]["ok"] is True
        assert items[0]["result"]["steps"]["draft"]["status"] == "never_sent"
        assert items[1]["ok"] is False
        assert "9999" in items[1]["error"]
        assert items[2]["ok"] is True

    def test_batch_json_array_of_ids(self, sample_db, tmp_path, capsys):
        batch_path = tmp_path / "batch.json"
        batch_path.write_text(json.dumps([1, 20]), encoding="utf-8")
        client = FakeClient(list(HAPPY_OUTCOMES) * 2)
        code, out, _ = run_cli(
            ["run", "--batch", str(batch_path), "--db", str(sample_db)], capsys, client=client
        )
        assert code == 0
        items = json.loads(out)
        assert [item["ok"] for item in items] == [True, True]

    def test_unreadable_batch_file_exits_2(self, sample_db, capsys):
        code, _, err = run_cli(
            ["run", "--batch", "no-such-file.txt", "--db", str(sample_db)], capsys
        )
        assert code == 2
        assert "no-such-file.txt" in err


class TestIngestStillWorks:
    def test_ingest_sample_subcommand(self, tmp_path, capsys):
        db_path = tmp_path / "ingested.db"
        code = cli.main(["ingest", "--sample", "--csv", str(FIXTURE_CSV), "--db", str(db_path)])
        assert code == 0
        assert db_path.is_file()
        out = capsys.readouterr().out
        assert "Ingested" in out


class TestLabelCommand:
    def test_requires_a_pass_or_merge(self, capsys):
        code, _out, err = run_cli(["label"], capsys)
        assert code == 2
        assert "--pass" in err

    def test_merge_without_any_pass_files_exits_2(self, tmp_path, capsys):
        code, _out, err = run_cli(
            ["label", "--merge", "--out", str(tmp_path)], capsys
        )
        assert code == 2
        assert "no answers recorded" in err

    def test_merge_writes_the_gold_file(self, tmp_path, capsys):
        from triage.evals.label_helper import (
            CATEGORY_PASS,
            ESCALATE_PASS,
            QUEUE_PASS,
            record_answer,
        )

        for thread_id in (1, 2):
            record_answer(CATEGORY_PASS, tmp_path, thread_id, "Billing/Payments")
            record_answer(QUEUE_PASS, tmp_path, thread_id, "Billing Ops")
            record_answer(ESCALATE_PASS, tmp_path, thread_id, "false")

        dest = tmp_path / "gold_labels.csv"
        code, out, _err = run_cli(
            ["label", "--merge", "--out", str(tmp_path), "--labels", str(dest)], capsys
        )
        assert code == 0
        assert "merged 2 threads" in out
        assert dest.is_file()

    def test_missing_store_exits_2(self, tmp_path, capsys):
        code, _out, err = run_cli(
            [
                "label",
                "--pass",
                "category",
                "--db",
                str(tmp_path / "nope.db"),
                "--out",
                str(tmp_path),
            ],
            capsys,
        )
        assert code == 2
        assert "no store" in err

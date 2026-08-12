"""MCP server smoke tests (U9) — R9, R27, R28, R30, KTD3.

Scripted generic-client sessions over the SDK's in-memory transport (no
subprocess, zero network): tool listing with read-only annotations and
discoverable label sets, discovery -> retrieval -> categorization happy path
with a mocked LLM, tool-level errors (isError results, never protocol
crashes) for unknown thread ids and exhausted LLM retries (R27), the graceful
not-yet-recorded eval report, and fail-fast startup diagnostics naming the
ingest command and the API key variable. The model profile defaults to dev
(R30) so exercising this surface never silently burns measurement models.
"""

import json
from pathlib import Path

import anyio
import pytest
from mcp.client import Client
from test_tools import (
    CATEGORIZED,
    DRAFTED,
    ESCALATED,
    ROUTED,
    FakeClient,
    ok,
    validation_error_for,
)

from triage import mcp_server
from triage.config import API_KEY_ENV_VAR, DEV, MEASUREMENT
from triage.ingest.store import ingest_csv, open_store
from triage.prompts import fragments
from triage.tools import CategorizeResult
from triage.tools.retrieval import search_threads

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "sample_tweets.csv"

# Tokens that would suggest a send/publish/write capability (R9: none exists
# anywhere in the tool layer; nothing writes gold labels or runs evals).
FORBIDDEN_NAME_TOKENS = (
    "send",
    "publish",
    "post",
    "write",
    "update",
    "delete",
    "create",
    "set_",
    "run_eval",
    "label_",
)


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("mcp_smoke") / "triage.db"
    ingest_csv(FIXTURE_CSV, path)
    return path


@pytest.fixture(scope="module")
def diagnosable_thread_id(db_path):
    conn = open_store(db_path)
    try:
        return search_threads(conn, "data plan")[0]["thread_id"]
    finally:
        conn.close()


def list_tools(server):
    async def scenario():
        async with Client(server) as client:
            return (await client.list_tools()).tools

    return anyio.run(scenario)


def call_tool(server, name, arguments=None):
    async def scenario():
        async with Client(server) as client:
            return await client.call_tool(name, arguments or {})

    return anyio.run(scenario)


class TestListing:
    def test_all_seven_tools_carry_read_only_annotations(self, db_path):
        tools_listed = list_tools(mcp_server.build_server(db_path=db_path))
        assert len(tools_listed) == 7
        for tool in tools_listed:
            assert tool.annotations is not None, tool.name
            assert tool.annotations.read_only_hint is True, tool.name

    def test_no_tool_name_suggests_send_publish_or_write(self, db_path):
        for tool in list_tools(mcp_server.build_server(db_path=db_path)):
            for token in FORBIDDEN_NAME_TOKENS:
                assert token not in tool.name, (tool.name, token)

    def test_label_sets_are_discoverable_from_descriptions(self, db_path):
        # R9: a generic client learns the valid label sets from descriptions.
        by_name = {t.name: t for t in list_tools(mcp_server.build_server(db_path=db_path))}
        for label in fragments.CATEGORY_LABELS:
            assert label in by_name["categorize_thread"].description
        for label in fragments.QUEUE_LABELS:
            assert label in by_name["route_thread"].description

    def test_draft_description_states_never_sent(self, db_path):
        by_name = {t.name: t for t in list_tools(mcp_server.build_server(db_path=db_path))}
        assert "never_sent" in by_name["draft_reply"].description


class TestHappyPath:
    def test_discover_retrieve_categorize_in_one_session(self, db_path):
        fake = FakeClient([ok(CATEGORIZED)])
        server = mcp_server.build_server(db_path=db_path, client=fake)

        async def scenario():
            async with Client(server) as client:
                tools_listed = (await client.list_tools()).tools
                found = await client.call_tool("find_threads", {"query": "data plan"})
                thread_id = found.structured_content["threads"][0]["thread_id"]
                fetched = await client.call_tool("get_thread", {"thread_id": thread_id})
                categorized = await client.call_tool(
                    "categorize_thread", {"thread_id": thread_id}
                )
                return tools_listed, fetched, categorized

        tools_listed, fetched, categorized = anyio.run(scenario)
        assert len(tools_listed) == 7
        assert not fetched.is_error
        assert "my data plan stopped working" in fetched.structured_content["rendered"]
        assert not categorized.is_error
        payload = categorized.structured_content
        assert payload["label"] in fragments.CATEGORY_LABELS  # valid taxonomy label
        assert payload["rationale"]  # R9: label plus rationale
        assert len(fake.messages.calls) == 1  # exactly one mocked LLM call

    def test_find_threads_without_query_lists(self, db_path):
        server = mcp_server.build_server(db_path=db_path)
        result = call_tool(server, "find_threads", {"limit": 3})
        assert not result.is_error
        threads = result.structured_content["threads"]
        assert len(threads) == 3
        assert {"thread_id", "root_tweet_id", "customer_author_id", "tweet_count"} <= set(
            threads[0]
        )

    def test_route_thread_returns_taxonomy_queue(self, db_path, diagnosable_thread_id):
        server = mcp_server.build_server(db_path=db_path, client=FakeClient([ok(ROUTED)]))
        result = call_tool(server, "route_thread", {"thread_id": diagnosable_thread_id})
        assert not result.is_error
        assert result.structured_content["queue"] in fragments.QUEUE_LABELS

    def test_draft_reply_carries_explicit_never_sent_status(
        self, db_path, diagnosable_thread_id
    ):
        # R9/R6: the draft result's status field is always 'never_sent'.
        server = mcp_server.build_server(db_path=db_path, client=FakeClient([ok(DRAFTED)]))
        result = call_tool(server, "draft_reply", {"thread_id": diagnosable_thread_id})
        assert not result.is_error
        assert result.structured_content["status"] == "never_sent"
        assert result.structured_content["draft"]

    def test_assess_escalation_returns_decision_and_reason(
        self, db_path, diagnosable_thread_id
    ):
        server = mcp_server.build_server(db_path=db_path, client=FakeClient([ok(ESCALATED)]))
        result = call_tool(server, "assess_escalation", {"thread_id": diagnosable_thread_id})
        assert not result.is_error
        assert result.structured_content["escalate"] is False
        assert result.structured_content["reason"]


class TestToolErrors:
    def test_unknown_thread_id_is_tool_level_error_not_crash(self, db_path):
        server = mcp_server.build_server(db_path=db_path)

        async def scenario():
            async with Client(server) as client:
                bad = await client.call_tool("categorize_thread", {"thread_id": 999_999})
                # The session survives: a follow-up call still succeeds.
                good = await client.call_tool("find_threads", {"limit": 1})
                return bad, good

        bad, good = anyio.run(scenario)
        assert bad.is_error
        assert "999999" in bad.content[0].text
        assert not good.is_error

    def test_llm_output_failure_becomes_tool_level_error(
        self, db_path, diagnosable_thread_id
    ):
        # R27: an OutputFailure after exhausted retries surfaces as an
        # isError tool result, never an exception escaping the protocol.
        bad = [validation_error_for(CategorizeResult, "{not json") for _ in range(3)]
        server = mcp_server.build_server(db_path=db_path, client=FakeClient(bad))
        result = call_tool(server, "categorize_thread", {"thread_id": diagnosable_thread_id})
        assert result.is_error
        text = result.content[0].text
        assert "categorize" in text
        assert "malformed" in text


class TestEvalReport:
    def test_before_reference_exists_returns_graceful_message(self, db_path, tmp_path):
        reference_path = tmp_path / "reference_run.json"
        server = mcp_server.build_server(db_path=db_path, reference_path=reference_path)
        result = call_tool(server, "get_eval_report")
        assert not result.is_error  # graceful message, not an error result
        payload = result.structured_content
        assert payload["status"] == "not_recorded"
        assert "--record-reference" in payload["message"]

    def test_after_reference_exists_returns_summary(self, db_path, tmp_path):
        reference_path = tmp_path / "reference_run.json"
        reference = {
            "recorded_at": "2026-08-11T00:00:00+00:00",
            "run_id": "ref-1",
            "profile": "measurement",
            "models": {"pipeline": "model-a", "baseline": "model-a", "judge": "model-b"},
            "prompt_hashes": {"categorize": "abc123"},
            "eval_set_hash": "deadbeef",
            "thread_digest": "cafe1234",
            # A recorded reference always gates mean draft score (R15).
            "metrics": {"categorize_accuracy": 0.9, "mean_draft_score": 4.2},
            "spreads": {"categorize_accuracy": 0.01, "mean_draft_score": 0.1},
            "per_thread": [{"thread_id": 1}],
        }
        reference_path.write_text(json.dumps(reference), encoding="utf-8")
        server = mcp_server.build_server(db_path=db_path, reference_path=reference_path)
        result = call_tool(server, "get_eval_report")
        assert not result.is_error
        payload = result.structured_content
        assert payload["status"] == "recorded"
        assert payload["metrics"] == {"categorize_accuracy": 0.9, "mean_draft_score": 4.2}
        assert payload["eval_set_hash"] == "deadbeef"
        assert payload["thread_count"] == 1


class TestStartupFailFast:
    def test_missing_store_names_the_ingest_command(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv(mcp_server.DB_ENV_VAR, str(tmp_path / "absent.db"))
        monkeypatch.setenv(API_KEY_ENV_VAR, "sk-test")
        monkeypatch.delenv(mcp_server.PROFILE_ENV_VAR, raising=False)
        assert mcp_server.main() != 0
        err = capsys.readouterr().err
        assert "triage ingest" in err

    def test_missing_api_key_names_the_variable(self, db_path, monkeypatch, capsys):
        monkeypatch.setenv(mcp_server.DB_ENV_VAR, str(db_path))
        monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
        monkeypatch.delenv(mcp_server.PROFILE_ENV_VAR, raising=False)
        assert mcp_server.main() != 0
        assert API_KEY_ENV_VAR in capsys.readouterr().err

    def test_unknown_profile_fails_with_instructive_message(self, monkeypatch, capsys):
        monkeypatch.setenv(mcp_server.PROFILE_ENV_VAR, "bogus")
        assert mcp_server.main() != 0
        assert "bogus" in capsys.readouterr().err


class TestProfileResolution:
    def test_profile_defaults_to_dev(self, monkeypatch):
        # R30: testing through the MCP surface never silently burns
        # measurement models.
        monkeypatch.delenv(mcp_server.PROFILE_ENV_VAR, raising=False)
        assert mcp_server.resolve_profile() == DEV

    def test_profile_env_var_is_honored(self, monkeypatch):
        monkeypatch.setenv(mcp_server.PROFILE_ENV_VAR, MEASUREMENT)
        assert mcp_server.resolve_profile() == MEASUREMENT

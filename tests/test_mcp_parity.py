"""Structural parity: MCP surface == pipeline tool layer (U9) — R9, R10, AE5, KTD2, KTD3.

AE5 is verified STRUCTURALLY (one function, two call sites), not by output
equality alone: the MCP wrappers dispatch through
``triage.mcp_server.STEP_FUNCTIONS``, whose values are asserted BY IDENTITY
(``is``) to be the same module-level tool-layer function objects the pipeline
nodes call — one tool layer, two surfaces (R10) — and the payload a generic
MCP client receives is the ``model_dump`` of the very Pydantic result the
tool-layer function returns: identical schema and label space.

Every test is fully mocked and in-process: the SDK's in-memory transport
(``mcp.client.Client`` wrapping the server instance) replaces a spawned stdio
subprocess. Zero network.
"""

from pathlib import Path

import anyio
import pytest
from mcp.client import Client
from test_tools import CATEGORIZED, DRAFTED, ESCALATED, ROUTED, FakeClient, ok

from triage import mcp_server, tools
from triage.config import DEV, ROLE_PIPELINE, get_model
from triage.ingest.store import ingest_csv, open_store
from triage.pipeline import graph
from triage.tools.retrieval import get_thread_by_id, search_threads

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "sample_tweets.csv"

# R9: the seven-tool surface — discovery is ONE tool (list when no query,
# search when given), keeping the count at exactly seven.
EXPECTED_TOOLS = {
    "find_threads",
    "get_thread",
    "categorize_thread",
    "route_thread",
    "draft_reply",
    "assess_escalation",
    "get_eval_report",
}

# MCP tool name -> the parsed outcome its one mocked LLM call returns.
STEP_OUTCOMES = {
    "categorize_thread": CATEGORIZED,
    "route_thread": ROUTED,
    "draft_reply": DRAFTED,
    "assess_escalation": ESCALATED,
}


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("mcp_parity") / "triage.db"
    ingest_csv(FIXTURE_CSV, path)
    return path


@pytest.fixture(scope="module")
def diagnosable_thread_id(db_path):
    conn = open_store(db_path)
    try:
        return search_threads(conn, "data plan")[0]["thread_id"]
    finally:
        conn.close()


def call_tool(server, name, arguments=None):
    async def scenario():
        async with Client(server) as client:
            return await client.call_tool(name, arguments or {})

    return anyio.run(scenario)


class TestSevenToolListing:
    def test_generic_client_lists_exactly_seven_tools(self, db_path):
        server = mcp_server.build_server(db_path=db_path)

        async def scenario():
            async with Client(server) as client:
                return await client.list_tools()

        listed = anyio.run(scenario)
        assert len(listed.tools) == 7  # R9: seven tools, no more, no fewer
        assert {tool.name for tool in listed.tools} == EXPECTED_TOOLS


class TestStructuralParity:
    def test_mcp_wrappers_dispatch_the_identical_function_objects(self):
        # AE5: one function, two call sites — identity, not output equality.
        assert mcp_server.STEP_FUNCTIONS["categorize_thread"] is tools.categorize
        assert mcp_server.STEP_FUNCTIONS["route_thread"] is tools.route
        assert mcp_server.STEP_FUNCTIONS["draft_reply"] is tools.draft
        assert mcp_server.STEP_FUNCTIONS["assess_escalation"] is tools.escalate

    def test_pipeline_nodes_reference_the_same_function_objects(self):
        # The other call site: the LangGraph nodes call these very objects.
        assert graph.categorize is tools.categorize
        assert graph.route is tools.route
        assert graph.draft is tools.draft
        assert graph.escalate is tools.escalate

    @pytest.mark.parametrize("tool_name", sorted(STEP_OUTCOMES))
    def test_mcp_result_equals_tool_layer_model_dump(
        self, db_path, diagnosable_thread_id, tool_name
    ):
        parsed = STEP_OUTCOMES[tool_name]
        server = mcp_server.build_server(db_path=db_path, client=FakeClient([ok(parsed)]))
        via_mcp = call_tool(server, tool_name, {"thread_id": diagnosable_thread_id})
        assert not via_mcp.is_error

        conn = open_store(db_path)
        try:
            thread = get_thread_by_id(conn, diagnosable_thread_id)
        finally:
            conn.close()
        direct = mcp_server.STEP_FUNCTIONS[tool_name](
            thread,
            model=get_model(ROLE_PIPELINE, DEV),
            client=FakeClient([ok(parsed)]),
        )
        # Identical schema: the MCP payload IS the tool-layer result's dump.
        assert via_mcp.structured_content == direct.model_dump()

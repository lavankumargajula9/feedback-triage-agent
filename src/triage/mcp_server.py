"""Seven-tool read-only stdio MCP server over the shared tool layer (R9, R10, KTD2, KTD3).

R9 tool-count reading: R9 enumerates the surface as "thread discovery
(list/search), thread retrieval, categorize, route, draft, assess-escalation,
and a zero-LLM-cost eval-report reader" — seven tools. Splitting discovery
into separate list and search functions would make eight, so discovery is
deliberately ONE tool: ``find_threads`` lists when ``query`` is empty and
searches when it is given. The seven registered tools are::

    find_threads, get_thread, categorize_thread, route_thread,
    draft_reply, assess_escalation, get_eval_report

Each tool is a thin decorated wrapper delegating to the same tool-layer
function the pipeline nodes call (KTD2, R10): the LLM-backed wrappers dispatch
through :data:`STEP_FUNCTIONS`, whose values ARE the module-level
``triage.tools`` functions, so the one-function-two-call-sites parity (AE5)
is asserted structurally by identity in tests. All tools carry read-only
annotations; the valid label sets are embedded in the categorize/route tool
descriptions; draft results carry the explicit ``never_sent`` status (R9).
No send/publish capability exists anywhere, no tool writes gold labels, and
nothing here can trigger an eval run — ``get_eval_report`` only reads the
recorded reference-run artifact (KTD10).

Tool-level problems — unknown thread ids, LLM output failures after exhausted
retries (R27) — surface as ``isError`` tool results, never protocol crashes.
Startup fails fast with instructive stderr messages when the store is missing
(naming ``triage ingest``) or the API key is absent. The model profile is
resolved from ``TRIAGE_PROFILE`` at startup and DEFAULTS TO DEV (R30), so
exercising the MCP surface never silently burns measurement models. Tool code
never prints to stdout: the stdio transport owns it (KTD3). Scope: stdio tools
only — no HTTP transport, no MCP resources or prompts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from triage.config import DEV, ROLE_PIPELINE, ConfigError, get_api_key, get_model
from triage.evals.regression import RegressionError, load_reference
from triage.ingest import IngestError
from triage.ingest.store import open_store
from triage.prompts import fragments
from triage.tools import OutputFailure, categorize, draft, escalate, route
from triage.tools.retrieval import (
    get_thread_by_id,
    list_threads,
    render_thread,
    search_threads,
)

SERVER_NAME = "feedback-triage"

# Environment knobs, resolved once at startup — never inside tool code.
PROFILE_ENV_VAR = "TRIAGE_PROFILE"
DB_ENV_VAR = "TRIAGE_DB"
REFERENCE_ENV_VAR = "TRIAGE_REFERENCE"

DEFAULT_DB_PATH = Path("data") / "triage.db"
DEFAULT_REFERENCE_PATH = Path("data") / "eval" / "reference_run.json"

# Startup config/input problems (mirrors the CLI's input-error code, R7).
EXIT_CONFIG_ERROR = 2

# AE5/R10: the MCP wrappers dispatch through this mapping, whose values are
# the SAME module-level function objects the pipeline nodes call — one tool
# layer, two surfaces. The parity test asserts the identity with ``is``.
STEP_FUNCTIONS: dict[str, Any] = {
    "categorize_thread": categorize,
    "route_thread": route,
    "draft_reply": draft,
    "assess_escalation": escalate,
}

# Every tool on this surface only reads (R9): no send, publish, or write
# capability exists anywhere in the tool layer.
READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)

_INGEST_HINT = "run `triage ingest` (or `triage ingest --sample` for the bundled fixture)"
_LLM_NOTE = "Makes one schema-enforced LLM call under the server's configured model profile."

FIND_THREADS_DESCRIPTION = (
    "Discover stored support threads (read-only). With no 'query' this LISTS threads in "
    "stable thread_id order (rows: thread_id, root_tweet_id, customer_author_id, "
    "tweet_count; page with 'limit'/'offset'). With a 'query' it SEARCHES for threads "
    "whose tweet text contains the query, case-insensitively (rows: thread_id, tweet_id, "
    "text). One tool covers both discovery modes."
)

GET_THREAD_DESCRIPTION = (
    "Retrieve one stored thread by thread_id (read-only): thread metadata plus the "
    "rendered conversation text — exactly the text every triage surface shows the model."
)

CATEGORIZE_DESCRIPTION = (
    "Categorize a stored thread: returns exactly one category label (what the issue is "
    f"about) plus a rationale grounded in the thread text. {_LLM_NOTE}"
    f"\n\n{fragments.category_block()}"
)

ROUTE_DESCRIPTION = (
    "Route a stored thread: returns exactly one support-queue label (which team should "
    "own it) plus a rationale. The queue label set is a different vocabulary from the "
    f"category label set. {_LLM_NOTE}\n\n{fragments.queue_block()}"
)

DRAFT_DESCRIPTION = (
    "Write a reply draft for a stored thread, for HUMAN REVIEW ONLY: nothing is ever "
    "sent, and the result carries an explicit status field that is always 'never_sent'. "
    f"{_LLM_NOTE}"
)

ESCALATE_DESCRIPTION = (
    "Assess whether a stored thread needs human attention: returns a binary 'escalate' "
    f"decision plus the stated reason. {_LLM_NOTE}"
)

EVAL_REPORT_DESCRIPTION = (
    "Read the recorded reference eval run's report (zero LLM cost: this only reads a "
    "JSON artifact and can never trigger an eval run). Returns the gated metrics, their "
    "observed spreads, and the determinism pins (models, prompt hashes, eval-set hash), "
    "or a graceful not-recorded-yet message while no reference exists."
)


def resolve_profile() -> str:
    """Model profile from ``TRIAGE_PROFILE``, defaulting to dev (R30)."""
    return os.environ.get(PROFILE_ENV_VAR, "").strip() or DEV


def build_server(
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
    profile: str = DEV,
    reference_path: Path | str = DEFAULT_REFERENCE_PATH,
    client: Any = None,
) -> MCPServer:
    """Build the seven-tool server over one store and model profile (KTD3).

    ``client`` is injectable for tests (KTD2); ``None`` lets the tool layer
    construct the real client from the environment on the first LLM call.
    """
    db_path = Path(db_path)
    reference_path = Path(reference_path)
    model = get_model(ROLE_PIPELINE, profile)
    server = MCPServer(
        SERVER_NAME,
        instructions=(
            "Read-only triage tools over reconstructed customer-support threads. "
            "Discover threads with find_threads, fetch one with get_thread, then "
            "categorize/route/draft/assess it by thread_id. Nothing here can send, "
            "publish, or write anything."
        ),
    )

    def _connect():
        """Open the store fresh per call (sync tools run on worker threads)."""
        if not db_path.is_file():
            raise ToolError(f"no store at {db_path}: {_INGEST_HINT} first")
        return open_store(db_path)

    def _load_thread(thread_id: int):
        conn = _connect()
        try:
            try:
                return get_thread_by_id(conn, thread_id)
            except IngestError as exc:
                # Unknown id -> tool-level error result, never a protocol crash.
                raise ToolError(str(exc)) from exc
        finally:
            conn.close()

    def _run_step(tool_name: str, thread_id: int) -> dict[str, Any]:
        """One LLM step via the SAME tool-layer function the pipeline calls (R10)."""
        thread = _load_thread(thread_id)
        result = STEP_FUNCTIONS[tool_name](thread, model=model, client=client)
        if isinstance(result, OutputFailure):
            # R27 typed failure -> isError tool result, not an exception
            # escaping the protocol.
            raise ToolError(
                f"{result.step} failed after {result.attempts} attempt(s) "
                f"({result.kind}): {result.detail}"
            )
        return result.model_dump()

    @server.tool(name="find_threads", description=FIND_THREADS_DESCRIPTION, annotations=READ_ONLY)
    def find_threads(query: str = "", limit: int = 20, offset: int = 0) -> dict[str, Any]:
        conn = _connect()
        try:
            if query.strip():
                rows = search_threads(conn, query, limit=limit)
            else:
                rows = list_threads(conn, limit=limit, offset=offset)
        finally:
            conn.close()
        return {"threads": rows}

    @server.tool(name="get_thread", description=GET_THREAD_DESCRIPTION, annotations=READ_ONLY)
    def get_thread(thread_id: int) -> dict[str, Any]:
        thread = _load_thread(thread_id)
        return {
            "thread_id": thread_id,
            "root_tweet_id": thread.root_tweet_id,
            "customer_author_id": thread.customer_author_id,
            "tweet_count": len(thread.tweets),
            "truncated": thread.truncated,
            "cycle_flagged": thread.cycle_flagged,
            "rendered": render_thread(thread),
        }

    @server.tool(
        name="categorize_thread", description=CATEGORIZE_DESCRIPTION, annotations=READ_ONLY
    )
    def categorize_thread(thread_id: int) -> dict[str, Any]:
        return _run_step("categorize_thread", thread_id)

    @server.tool(name="route_thread", description=ROUTE_DESCRIPTION, annotations=READ_ONLY)
    def route_thread(thread_id: int) -> dict[str, Any]:
        return _run_step("route_thread", thread_id)

    @server.tool(name="draft_reply", description=DRAFT_DESCRIPTION, annotations=READ_ONLY)
    def draft_reply(thread_id: int) -> dict[str, Any]:
        return _run_step("draft_reply", thread_id)

    @server.tool(
        name="assess_escalation", description=ESCALATE_DESCRIPTION, annotations=READ_ONLY
    )
    def assess_escalation(thread_id: int) -> dict[str, Any]:
        return _run_step("assess_escalation", thread_id)

    @server.tool(
        name="get_eval_report", description=EVAL_REPORT_DESCRIPTION, annotations=READ_ONLY
    )
    def get_eval_report() -> dict[str, Any]:
        if not reference_path.is_file():
            # Graceful before the artifact exists (a later unit records it,
            # KTD10) — a normal result, not an error.
            return {
                "status": "not_recorded",
                "message": (
                    f"no reference eval run recorded yet at {reference_path}; record one "
                    f"explicitly with `triage eval --record-reference {reference_path}` "
                    "(a reference is never recorded automatically)"
                ),
            }
        try:
            reference = load_reference(reference_path)
        except RegressionError as exc:
            raise ToolError(str(exc)) from exc
        return {
            "status": "recorded",
            "recorded_at": reference.get("recorded_at", ""),
            "run_id": reference.get("run_id", ""),
            "profile": reference.get("profile", ""),
            "models": reference["models"],
            "prompt_hashes": reference["prompt_hashes"],
            "eval_set_hash": reference["eval_set_hash"],
            "metrics": reference["metrics"],
            "spreads": reference["spreads"],
            "thread_count": len(reference["per_thread"]),
        }

    return server


def main() -> int:
    """Fail-fast stdio entrypoint: validate configuration, then serve.

    Every diagnostic goes to stderr — stdout belongs to the stdio transport
    (KTD3). Exit code 2 for configuration/input problems, matching the CLI's
    input-error convention (R7).
    """
    profile = resolve_profile()
    try:
        get_model(ROLE_PIPELINE, profile)
    except ConfigError as exc:
        print(f"error: {exc} (set {PROFILE_ENV_VAR})", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    db_path = Path(os.environ.get(DB_ENV_VAR, "") or DEFAULT_DB_PATH)
    if not db_path.is_file():
        print(
            f"error: no store at {db_path}: {_INGEST_HINT} first, or point "
            f"{DB_ENV_VAR} at an existing store",
            file=sys.stderr,
        )
        return EXIT_CONFIG_ERROR
    try:
        get_api_key()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    reference_path = Path(os.environ.get(REFERENCE_ENV_VAR, "") or DEFAULT_REFERENCE_PATH)
    server = build_server(db_path=db_path, profile=profile, reference_path=reference_path)
    server.run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())

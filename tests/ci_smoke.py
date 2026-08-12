"""End-to-end CLI smoke over the model-calling path, with a stub client.

CI has no ANTHROPIC_API_KEY, so the console-script smoke can only reach the
degenerate short-circuit on its own. This drives the same `triage run` entry
point through `cli.main`'s injection seam with scripted responses, so the
non-degenerate path — four steps, real labels, a draft — is covered offline.

Run: python tests/ci_smoke.py
"""

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]

from triage import cli
from triage.ingest.store import ingest_csv
from triage.tools.schemas import (
    CategorizeResult,
    DraftResult,
    EscalateResult,
    RouteResult,
)

SCRIPTED = [
    CategorizeResult(label="Technical/Product", rationale="Service outage reported."),
    RouteResult(queue="Technical/Product", rationale="Needs the technical team."),
    EscalateResult(escalate=False, reason="Routine issue, resolvable by the queue."),
    DraftResult(draft="Sorry about that — can you share your account email?"),
]


class StubMessages:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if not self._outcomes:
            raise AssertionError("stub exhausted: the CLI made more calls than expected")
        return SimpleNamespace(
            parsed_output=self._outcomes.pop(0), stop_reason="end_turn", usage=None
        )


class StubClient:
    def __init__(self, outcomes):
        self.messages = StubMessages(outcomes)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "smoke.db"
        ingest_csv(REPO_ROOT / "tests" / "fixtures" / "sample_tweets.csv", db_path)

        client = StubClient(SCRIPTED)
        out_path = Path(tmp) / "result.json"
        original_stdout = sys.stdout
        with out_path.open("w", encoding="utf-8") as handle:
            sys.stdout = handle
            try:
                code = cli.main(["run", "1", "--db", str(db_path)], client=client)
            finally:
                sys.stdout = original_stdout

        if code != 0:
            print(f"smoke FAILED: exit {code}", file=sys.stderr)
            return 1
        result = json.loads(out_path.read_text(encoding="utf-8"))
        steps = result["steps"]
        checks = {
            "categorize label": steps["categorize"]["label"] == "Technical/Product",
            "route queue": steps["route"]["queue"] == "Technical/Product",
            "escalation reason": bool(steps["escalate"]["reason"]),
            "draft never sent": steps["draft"]["status"] == "never_sent",
            "draft text": bool(steps["draft"]["draft"]),
            "ok": result["ok"] is True,
            "all four steps called": len(client.messages.calls) == 4,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            print(f"smoke FAILED: {', '.join(failed)}", file=sys.stderr)
            print(json.dumps(result, indent=2), file=sys.stderr)
            return 1
        print(f"smoke ok: 4 model-calling steps, category {steps['categorize']['label']}")
        return 0


if __name__ == "__main__":
    sys.exit(main())

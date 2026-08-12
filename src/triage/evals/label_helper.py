"""Three-pass hand-labeling helper (U6) — R11, R24; KTD8.

One annotator labels category, queue, and escalate for every thread (R24). Doing
that field-by-field within a thread anchors the later fields on the earlier ones:
queue drifts toward the team whose name echoes the category just chosen, and the
genuinely divergent cases the split exists to capture get under-labeled. That is
the same collapse the taxonomy split removed from the pipeline, reappearing in
the labels — where it is invisible, because gold labels are what everything else
is measured against.

So each field is a separate full sweep, and the isolation between sweeps is
structural rather than a display choice:

- ``LabelItem`` carries only a thread id and its text. There is no field on it
  for a label, so no prior answer can travel through a pass in memory.
- Every pass reads and writes exactly one file, derived from the pass itself.
  No function here takes another pass's path, so a pass cannot open one.
- ``merge_passes`` is the only function that opens more than one pass file, and
  it runs after all three sweeps are complete.

Each pass shuffles under its own seed, so by the second sweep threads arrive in
an order that defeats recall of the first sweep's answer. The seeds are fixed and
published through ``manifest()`` for the selection log — a reader can rerun the
ordering and check it.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from triage.prompts import fragments

ESCALATE_OPTIONS: tuple[str, ...] = ("true", "false")

GOLD_HEADER: tuple[str, ...] = ("thread_id", "category", "queue", "escalate")


class LabelHelperError(Exception):
    """Raised for an off-taxonomy answer or an incomplete/inconsistent merge."""


@dataclass(frozen=True)
class LabelItem:
    """One thread as a pass sees it.

    Deliberately has no label field: this is the structural guarantee that a
    pass cannot carry another pass's answer.
    """

    thread_id: int
    text: str


@dataclass(frozen=True)
class LabelPass:
    """One full sweep over the eval set, collecting exactly one field."""

    name: str
    column: str
    prompt: str
    options: tuple[str, ...]
    seed: int

    @property
    def filename(self) -> str:
        return f"pass_{self.name}.csv"


CATEGORY_PASS = LabelPass(
    name="category",
    column="category",
    prompt="What is this thread about?",
    options=fragments.CATEGORY_LABELS,
    seed=20260812,
)

QUEUE_PASS = LabelPass(
    name="queue",
    column="queue",
    prompt="Which team should own this thread?",
    options=fragments.QUEUE_LABELS,
    seed=778301,
)

ESCALATE_PASS = LabelPass(
    name="escalate",
    column="escalate",
    prompt="Does this thread need a human now?",
    options=ESCALATE_OPTIONS,
    seed=4419907,
)

PASSES: tuple[LabelPass, ...] = (CATEGORY_PASS, QUEUE_PASS, ESCALATE_PASS)


def manifest() -> dict[str, dict[str, object]]:
    """Pass seeds and label spaces, for the selection log's methodology entry."""
    return {
        p.name: {"seed": p.seed, "options": list(p.options), "file": p.filename}
        for p in PASSES
    }


def pass_path(pass_: LabelPass, out_dir: Path | str) -> Path:
    """The single file a pass may touch."""
    return Path(out_dir) / pass_.filename


def pass_order(pass_: LabelPass, thread_ids) -> list[int]:
    """Thread ids shuffled under this pass's own seed."""
    ordered = sorted(thread_ids)
    random.Random(pass_.seed).shuffle(ordered)
    return ordered


def read_pass(pass_: LabelPass, out_dir: Path | str) -> dict[int, str]:
    """Answers recorded so far in this pass, keyed by thread id."""
    path = pass_path(pass_, out_dir)
    if not path.is_file():
        return {}
    answers: dict[int, str] = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            answers[int(row["thread_id"])] = row[pass_.column]
    return answers


def record_answer(
    pass_: LabelPass, out_dir: Path | str, thread_id: int, value: str
) -> None:
    """Append one answer to this pass's file, refusing off-taxonomy values."""
    if value not in pass_.options:
        raise LabelHelperError(
            f"{value!r} is not a valid {pass_.name} value; allowed: "
            f"{', '.join(pass_.options)}"
        )
    path = pass_path(pass_, out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.is_file()
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(("thread_id", pass_.column))
        writer.writerow((thread_id, value))


def pending_items(
    pass_: LabelPass, out_dir: Path | str, items
) -> tuple[LabelItem, ...]:
    """Unanswered items for this pass, in this pass's shuffled order."""
    answered = read_pass(pass_, out_dir)
    by_id = {item.thread_id: item for item in items}
    return tuple(
        by_id[tid]
        for tid in pass_order(pass_, by_id)
        if tid not in answered
    )


def merge_passes(out_dir: Path | str, dest: Path | str) -> int:
    """Combine the three completed passes into the gold-labels CSV.

    The only function that opens more than one pass file. Refuses to write
    unless all three passes cover exactly the same thread ids, so a partial
    sweep can never masquerade as a frozen label set.
    """
    answers = {p.name: read_pass(p, out_dir) for p in PASSES}

    empty = [name for name, recorded in answers.items() if not recorded]
    if empty:
        raise LabelHelperError(
            f"no answers recorded for pass(es): {', '.join(sorted(empty))}"
        )

    id_sets = {name: set(recorded) for name, recorded in answers.items()}
    shared = set.intersection(*id_sets.values())
    union = set.union(*id_sets.values())
    if shared != union:
        missing = {
            name: sorted(union - ids) for name, ids in id_sets.items() if ids != union
        }
        detail = "; ".join(f"{name} missing {ids}" for name, ids in sorted(missing.items()))
        raise LabelHelperError(f"passes do not cover the same threads: {detail}")

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(shared)
    with open(dest, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(GOLD_HEADER)
        for thread_id in rows:
            writer.writerow(
                (
                    thread_id,
                    answers["category"][thread_id],
                    answers["queue"][thread_id],
                    answers["escalate"][thread_id],
                )
            )
    return len(rows)

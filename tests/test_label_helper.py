"""Tests for the three-pass labeling helper (U6) — R11, R24; KTD8.

The helper exists to stop the annotator anchoring one field on another. Its
central claim is structural, not cosmetic: a pass cannot reach another pass's
answers, so these tests assert isolation at the type and filesystem level
rather than asserting that a screen does not render a value.
"""

import builtins
import csv

import pytest

from triage.evals.label_helper import (
    CATEGORY_PASS,
    ESCALATE_PASS,
    PASSES,
    QUEUE_PASS,
    LabelHelperError,
    LabelItem,
    manifest,
    merge_passes,
    pass_order,
    pass_path,
    pending_items,
    read_pass,
    record_answer,
)
from triage.evals.runner import load_gold_labels
from triage.prompts import fragments

ITEMS = tuple(
    LabelItem(thread_id=i, text=f"customer says thing {i}") for i in range(1, 21)
)


def answer_all(pass_, out_dir, value_for):
    for item in ITEMS:
        record_answer(pass_, out_dir, item.thread_id, value_for(item.thread_id))


class TestPassIsolation:
    def test_label_item_cannot_hold_another_passs_answer(self):
        # Structural: the type carried through a pass has no label field, so
        # there is no in-memory channel for a prior answer to travel through.
        assert set(LabelItem.__dataclass_fields__) == {"thread_id", "text"}

    def test_a_pass_opens_only_its_own_file(self, tmp_path, monkeypatch):
        for pass_, value in (
            (CATEGORY_PASS, "Billing/Payments"),
            (QUEUE_PASS, "Billing Ops"),
        ):
            answer_all(pass_, tmp_path, lambda _tid, v=value: v)

        opened = []
        real_open = builtins.open

        def spy(file, *args, **kwargs):
            opened.append(str(file))
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", spy)
        pending_items(ESCALATE_PASS, tmp_path, ITEMS)
        record_answer(ESCALATE_PASS, tmp_path, 1, "true")
        read_pass(ESCALATE_PASS, tmp_path)

        own = str(pass_path(ESCALATE_PASS, tmp_path))
        assert opened, "expected the pass to touch its own file"
        assert all(path == own for path in opened), (
            f"a pass opened files outside its own: "
            f"{sorted(set(opened) - {own})}"
        )

    def test_each_pass_writes_a_distinct_file(self, tmp_path):
        paths = {str(pass_path(p, tmp_path)) for p in PASSES}
        assert len(paths) == len(PASSES)


class TestSeededShuffling:
    def test_each_pass_orders_threads_differently(self):
        ids = [item.thread_id for item in ITEMS]
        orders = [pass_order(p, ids) for p in PASSES]
        for first, second in ((0, 1), (0, 2), (1, 2)):
            assert orders[first] != orders[second]

    def test_order_is_deterministic_and_total(self):
        ids = [item.thread_id for item in ITEMS]
        assert pass_order(CATEGORY_PASS, ids) == pass_order(CATEGORY_PASS, ids)
        assert sorted(pass_order(CATEGORY_PASS, ids)) == sorted(ids)

    def test_manifest_records_the_seeds(self):
        recorded = manifest()
        assert set(recorded) == {p.name for p in PASSES}
        assert len({entry["seed"] for entry in recorded.values()}) == len(PASSES)


class TestResume:
    def test_pending_skips_answered_threads_in_pass_order(self, tmp_path):
        record_answer(CATEGORY_PASS, tmp_path, 3, "General Inquiry")
        pending = pending_items(CATEGORY_PASS, tmp_path, ITEMS)
        assert 3 not in [item.thread_id for item in pending]
        assert len(pending) == len(ITEMS) - 1
        expected = [
            tid
            for tid in pass_order(CATEGORY_PASS, [i.thread_id for i in ITEMS])
            if tid != 3
        ]
        assert [item.thread_id for item in pending] == expected

    def test_answers_survive_a_reopen(self, tmp_path):
        record_answer(QUEUE_PASS, tmp_path, 7, "Logistics")
        assert read_pass(QUEUE_PASS, tmp_path) == {7: "Logistics"}

    def test_off_taxonomy_answer_is_refused(self, tmp_path):
        with pytest.raises(LabelHelperError, match="Weather"):
            record_answer(CATEGORY_PASS, tmp_path, 1, "Weather")

    def test_a_category_value_is_refused_by_the_queue_pass(self, tmp_path):
        # The split only holds if the helper enforces the vocabularies.
        with pytest.raises(LabelHelperError):
            record_answer(QUEUE_PASS, tmp_path, 1, "Billing/Payments")


class TestMerge:
    def test_merge_produces_a_loadable_gold_file(self, tmp_path):
        categories = fragments.CATEGORY_LABELS
        queues = fragments.QUEUE_LABELS
        answer_all(CATEGORY_PASS, tmp_path, lambda tid: categories[tid % len(categories)])
        answer_all(QUEUE_PASS, tmp_path, lambda tid: queues[tid % len(queues)])
        answer_all(ESCALATE_PASS, tmp_path, lambda tid: "true" if tid % 2 else "false")

        dest = tmp_path / "gold_labels.csv"
        written = merge_passes(tmp_path, dest)
        assert written == len(ITEMS)

        with dest.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [r["thread_id"] for r in rows] == [str(i.thread_id) for i in ITEMS]

        gold = load_gold_labels(dest)
        assert set(gold) == {item.thread_id for item in ITEMS}

    def test_merge_refuses_incomplete_passes(self, tmp_path):
        answer_all(CATEGORY_PASS, tmp_path, lambda _tid: "General Inquiry")
        answer_all(QUEUE_PASS, tmp_path, lambda _tid: "Tier-1 General")
        record_answer(ESCALATE_PASS, tmp_path, 1, "true")

        with pytest.raises(LabelHelperError, match="escalate"):
            merge_passes(tmp_path, tmp_path / "gold_labels.csv")

    def test_merge_refuses_mismatched_thread_sets(self, tmp_path):
        for pass_, value in (
            (CATEGORY_PASS, "General Inquiry"),
            (QUEUE_PASS, "Tier-1 General"),
            (ESCALATE_PASS, "true"),
        ):
            record_answer(pass_, tmp_path, 1, value)
        record_answer(CATEGORY_PASS, tmp_path, 2, "General Inquiry")

        with pytest.raises(LabelHelperError):
            merge_passes(tmp_path, tmp_path / "gold_labels.csv")

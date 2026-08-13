"""Tests for candidate-pool construction (U6, KTD8; R11, R30).

The properties that matter here are not "it returns a list" but the ones the
pool's validity rests on:

- the SQL prefilter can only over-admit relative to ``degenerate_reason``, so
  it is a performance step and not a second definition of degeneracy;
- the volume reduction is a seeded uniform draw, so nothing about a thread's
  wording changes its odds of being scanned;
- rough passes cannot see each other's answers, so a rough queue label is not
  an echo of the rough category label;
- an unmet class floor is reported, never silently under-filled;
- the labeling path reads pool membership only and can never surface the
  model's rough guess to the annotator.

Every test is fully mocked: zero network.
"""

import csv
import json
from types import SimpleNamespace

import pytest
from test_tools import FIXTURE_CSV, make_thread

from triage import cli
from triage.evals import pool
from triage.ingest.store import get_thread, ingest_csv, open_store
from triage.prompts import fragments
from triage.tools.retrieval import degenerate_reason


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "pool.db"
    ingest_csv(FIXTURE_CSV, db_path)
    return open_store(db_path)


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "pool.db"
    ingest_csv(FIXTURE_CSV, path)
    return path


class RoughMessages:
    """Scripted rough-pass answers, dispatched on the requested schema."""

    def __init__(self, answers, refuse=()):
        self.answers = answers
        self.refuse = set(refuse)
        self.calls = []

    def parse(self, **kwargs):
        schema = kwargs["output_format"]
        text = kwargs["messages"][0]["content"]
        self.calls.append(
            {"schema": schema, "system": kwargs["system"], "text": text}
        )
        answer = self.answers[text]
        if schema is pool.RoughCategory:
            field = "category"
        elif schema is pool.RoughQueue:
            field = "queue"
        else:
            field = "escalate"
        if (text, field) in self.refuse:
            return SimpleNamespace(parsed_output=None, stop_reason="refusal")
        value = answer[field]
        parsed = {
            "category": lambda: pool.RoughCategory(label=value),
            "queue": lambda: pool.RoughQueue(queue=value),
            "escalate": lambda: pool.RoughEscalate(escalate=value),
        }[field]()
        return SimpleNamespace(parsed_output=parsed, stop_reason="end_turn")


class RoughClient:
    def __init__(self, answers, refuse=()):
        self.messages = RoughMessages(answers, refuse)


def rough_of(category, queue, escalate):
    return {"category": category, "queue": queue, "escalate": escalate}


def add_thread(store, *, root, author, text, inbound=1):
    """Append a single-tweet thread to the store, returning its thread id."""
    store.execute(
        "INSERT INTO threads (root_tweet_id, customer_author_id, truncated, "
        "cycle_flagged) VALUES (?, ?, 0, 0)",
        (root, author),
    )
    thread_id = store.execute("SELECT last_insert_rowid()").fetchone()[0]
    store.execute(
        "INSERT INTO tweets (tweet_id, author_id, inbound, created_at, text) "
        "VALUES (?, ?, ?, 'Tue Oct 31 22:00:00 +0000 2017', ?)",
        (root, author, inbound, text),
    )
    store.execute(
        "INSERT INTO thread_tweets (thread_id, position, tweet_id) VALUES (?, 0, ?)",
        (thread_id, root),
    )
    store.commit()
    return thread_id


class TestStructuralPrefilter:
    def test_admits_every_thread_the_real_predicate_accepts(self, store):
        # The safety property the whole design rests on: the SQL floor is a
        # performance step, so it must never reject a thread degenerate_reason
        # would have accepted. Seeded with short-but-diagnosable text, which is
        # where an over-eager floor would break the property.
        add_thread(store, root=9300, author="700", text="app crashed today")
        add_thread(store, root=9301, author="701", text="a b c")
        add_thread(store, root=9302, author="702", text="I have sent you a DM just now")

        eligible = set(pool.structural_prefilter(store)[0])
        all_ids = [row[0] for row in store.execute("SELECT thread_id FROM threads")]
        accepted = [
            tid for tid in all_ids if degenerate_reason(get_thread(store, tid)) is None
        ]
        assert accepted
        for thread_id in accepted:
            assert thread_id in eligible

    def test_counts_rejections_per_criterion(self, store):
        thread_id = add_thread(
            store, root=9001, author="brand", text="we replied", inbound=0
        )
        eligible, rejected = pool.structural_prefilter(store)
        assert thread_id not in eligible
        assert rejected[pool.CRITERION_NO_CUSTOMER_TWEET] == 1

    def test_rejects_customer_text_below_the_floor(self, store):
        thread_id = add_thread(store, root=9100, author="888", text="hi")
        eligible, rejected = pool.structural_prefilter(store)
        assert thread_id not in eligible
        assert rejected[pool.CRITERION_TEXT_BELOW_FLOOR] == 1

    def test_floor_derives_from_the_diagnosable_word_minimum(self):
        # Three content words need at least three characters plus two
        # separators; a lower floor than this would start rejecting threads
        # the real predicate accepts.
        from triage.tools.retrieval import MIN_DIAGNOSABLE_WORDS

        assert pool.MIN_CUSTOMER_CHARS == 2 * MIN_DIAGNOSABLE_WORDS - 1


class TestScanSample:
    def test_is_deterministic_under_the_published_seed(self):
        eligible = list(range(1000))
        first = pool.sample_scan_pool(eligible, 50)
        second = pool.sample_scan_pool(eligible, 50)
        assert first == second
        assert len(first) == 50
        assert set(first) <= set(eligible)

    def test_different_seeds_select_differently(self):
        eligible = list(range(1000))
        assert pool.sample_scan_pool(eligible, 50, seed=1) != pool.sample_scan_pool(
            eligible, 50, seed=2
        )

    def test_returns_everything_when_scan_size_exceeds_the_corpus(self):
        assert pool.sample_scan_pool([3, 1, 2], 99) == [1, 2, 3]


class TestDegeneracyScreen:
    def test_drops_degenerates_and_counts_them(self, store):
        # A deflect-to-DM thread clears the SQL character floor but fails the
        # real predicate — the case that proves the screen is doing work the
        # prefilter cannot do.
        deflected = add_thread(
            store, root=9200, author="777", text="I have sent you a DM just now"
        )
        eligible, _ = pool.structural_prefilter(store)
        assert deflected in eligible

        survivors, rejected = pool.screen_degenerate(store, eligible)
        assert rejected[pool.CRITERION_DEGENERATE] == 1
        assert all(tid != deflected for tid, _ in survivors)
        assert len(survivors) == len(eligible) - 1

    def test_screen_agrees_with_the_predicate_on_every_thread(self, store):
        add_thread(store, root=9201, author="776", text="I have sent you a DM just now")
        eligible, _ = pool.structural_prefilter(store)
        survivors, _ = pool.screen_degenerate(store, eligible)
        kept = {tid for tid, _ in survivors}
        for thread_id in eligible:
            accepted = degenerate_reason(get_thread(store, thread_id)) is None
            assert (thread_id in kept) is accepted

    def test_honors_the_caller_exclusion_set(self, store):
        all_ids = [row[0] for row in store.execute("SELECT thread_id FROM threads")]
        survivors, rejected = pool.screen_degenerate(
            store, all_ids, exclude={all_ids[0]}
        )
        assert rejected[pool.CRITERION_EXCLUDED] == 1
        assert all(tid != all_ids[0] for tid, _ in survivors)


class TestOpeningMessage:
    def test_takes_the_first_customer_message_cleaned(self):
        thread = make_thread(
            ["@brand my payment failed twice today", "@100001 sorry!"],
            inbound_flags=[True, False],
        )
        assert pool.opening_message(thread) == "my payment failed twice today"

    def test_caps_the_payload(self):
        thread = make_thread(["word " * 500])
        assert len(pool.opening_message(thread)) <= pool.OPENING_CHAR_LIMIT


class TestRoughClassify:
    def test_collects_every_field(self):
        survivors = [(1, "payment failed")]
        answers = {"payment failed": rough_of("Billing/Payments", "Billing Ops", True)}
        client = RoughClient(answers)
        rough, failures = pool.rough_classify(
            survivors, model="claude-haiku-4-5", client=client
        )
        assert rough[1] == pool.RoughLabels("Billing/Payments", "Billing Ops", True)
        assert not failures

    def test_a_failed_field_drops_the_thread_and_is_counted(self):
        survivors = [(1, "payment failed")]
        answers = {"payment failed": rough_of("Billing/Payments", "Billing Ops", True)}
        client = RoughClient(answers, refuse={("payment failed", "queue")})
        rough, failures = pool.rough_classify(
            survivors, model="claude-haiku-4-5", client=client
        )
        assert rough == {}
        assert failures[f"{pool.CRITERION_ROUGH_FAILED}:queue"] == 1

    def test_queue_pass_never_sees_the_category_answer(self):
        # Structural isolation, the same guarantee the hand-labeling helper
        # makes: stratifying on a queue label anchored to the category label
        # would under-represent category/queue divergence.
        survivors = [(1, "payment failed")]
        answers = {"payment failed": rough_of("Billing/Payments", "Billing Ops", True)}
        client = RoughClient(answers)
        pool.rough_classify(survivors, model="claude-haiku-4-5", client=client)

        queue_calls = [
            call for call in client.messages.calls if call["schema"] is pool.RoughQueue
        ]
        assert queue_calls
        for call in queue_calls:
            assert "Billing/Payments" not in call["system"]
            assert "Billing/Payments" not in call["text"]

    def test_payload_carries_no_label_definitions(self):
        survivors = [(1, "payment failed")]
        answers = {"payment failed": rough_of("Billing/Payments", "Billing Ops", True)}
        client = RoughClient(answers)
        pool.rough_classify(survivors, model="claude-haiku-4-5", client=client)
        definition = fragments.CATEGORY_TAXONOMY[0][1]
        assert all(definition not in call["system"] for call in client.messages.calls)


def spread_rough(per_class):
    """One rough entry per (category, queue) pair, repeated ``per_class`` times."""
    rough = {}
    thread_id = 1
    pairs = list(zip(fragments.CATEGORY_LABELS, fragments.QUEUE_LABELS))
    for _ in range(per_class):
        for index, (category, queue) in enumerate(pairs):
            rough[thread_id] = pool.RoughLabels(category, queue, index % 2 == 0)
            thread_id += 1
    return rough


class TestBucketKeys:
    def test_every_support_key_is_a_declared_quota_key(self):
        # Support keys and quota keys are looked up against each other. A
        # second encoding of the scheme would not raise — the quotas would
        # simply never be met — so pin that both sides agree.
        quotas = set(pool.quota_buckets(1))
        for category in fragments.CATEGORY_LABELS:
            for queue in fragments.QUEUE_LABELS:
                for escalate in (True, False):
                    labels = pool.RoughLabels(category, queue, escalate)
                    assert set(labels.buckets()) <= quotas

    def test_escalate_buckets_use_the_shared_encoding(self):
        labels = pool.RoughLabels("General Inquiry", "Tier-1 General", True)
        assert labels.buckets()[2] in pool.ESCALATE_BUCKETS
        assert pool.bucket_key(pool.BUCKET_ESCALATE, False) in pool.ESCALATE_BUCKETS


class TestStratifiedSelect:
    def test_meets_every_floor_when_the_data_allows(self):
        rough = spread_rough(per_class=4)
        selected, support, shortfalls = pool.stratified_select(
            rough, target_n=10, floor=3
        )
        assert not shortfalls
        for label in fragments.CATEGORY_LABELS:
            assert support[f"category:{label}"] >= 3
        for label in fragments.QUEUE_LABELS:
            assert support[f"queue:{label}"] >= 3
        for bucket in pool.ESCALATE_BUCKETS:
            assert support[bucket] >= 3
        assert len(set(selected)) == len(selected)

    def test_reports_shortfalls_rather_than_under_filling_silently(self):
        rough = spread_rough(per_class=1)
        _selected, _support, shortfalls = pool.stratified_select(
            rough, target_n=5, floor=3
        )
        assert shortfalls
        assert all(achieved < 3 for achieved in shortfalls.values())

    def test_is_deterministic(self):
        rough = spread_rough(per_class=4)
        first, _, _ = pool.stratified_select(rough, target_n=10, floor=3)
        second, _, _ = pool.stratified_select(rough, target_n=10, floor=3)
        assert first == second

    def test_tops_up_to_target_when_quotas_alone_fall_short(self):
        rough = spread_rough(per_class=4)
        selected, _, _ = pool.stratified_select(rough, target_n=20, floor=1)
        assert len(selected) == 20

    def test_never_truncates_below_a_satisfied_quota(self):
        rough = spread_rough(per_class=4)
        selected, support, shortfalls = pool.stratified_select(
            rough, target_n=2, floor=3
        )
        assert not shortfalls
        assert len(selected) > 2
        assert min(support.values()) >= 3


class TestFreezeAndRead:
    def test_membership_file_carries_no_rough_labels(self, tmp_path):
        # The isolation guarantee: the labeling path reads this file, so a
        # model guess in it would anchor the annotator's gold label.
        rough = {1: pool.RoughLabels("Billing/Payments", "Billing Ops", True)}
        pool.write_pool(tmp_path, [1], rough, {"selected": 1})
        text = (tmp_path / pool.POOL_FILE).read_text(encoding="utf-8")
        assert "Billing/Payments" not in text
        assert "Billing Ops" not in text
        with open(tmp_path / pool.POOL_FILE, encoding="utf-8", newline="") as handle:
            assert [field for field in csv.DictReader(handle).fieldnames] == ["thread_id"]

    def test_rough_labels_are_kept_in_their_own_file(self, tmp_path):
        rough = {1: pool.RoughLabels("Billing/Payments", "Billing Ops", True)}
        pool.write_pool(tmp_path, [1], rough, {"selected": 1})
        with open(tmp_path / pool.ROUGH_FILE, encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        assert row == {
            "thread_id": "1",
            "category": "Billing/Payments",
            "queue": "Billing Ops",
            "escalate": "true",
        }

    def test_round_trips_membership(self, tmp_path):
        rough = {
            tid: pool.RoughLabels("General Inquiry", "Tier-1 General", False)
            for tid in (3, 1, 2)
        }
        pool.write_pool(tmp_path, [1, 2, 3], rough, {"selected": 3})
        assert pool.read_pool(tmp_path) == [1, 2, 3]

    def test_stats_are_written_as_json(self, tmp_path):
        pool.write_pool(tmp_path, [], {}, {"selected": 0, "shortfalls": {}})
        assert json.loads((tmp_path / pool.STATS_FILE).read_text())["selected"] == 0

    def test_missing_pool_raises_with_an_instructive_message(self, tmp_path):
        with pytest.raises(pool.PoolError, match="triage pool"):
            pool.read_pool(tmp_path)

    def test_empty_pool_is_rejected(self, tmp_path):
        (tmp_path / pool.POOL_FILE).write_text("thread_id\n", encoding="utf-8")
        with pytest.raises(pool.PoolError, match="empty"):
            pool.read_pool(tmp_path)


class TestCostEstimate:
    def test_itemizes_calls_and_tokens(self):
        estimate = pool.estimate_scan_cost(
            100, model="claude-haiku-4-5", tokens_in=200, tokens_out=10, passes=3
        )
        assert estimate["calls"] == 300
        assert estimate["tokens_in_total"] == 60_000
        assert estimate["tokens_out_total"] == 3_000
        # haiku 4.5 at $1/$5 per MTok: 0.06 * 1 + 0.003 * 5 = 0.075
        assert estimate["cost_before_retries"] == pytest.approx(0.075)
        assert estimate["cost"] == pytest.approx(0.075 * 1.10)

    def test_unpriced_model_reports_none_rather_than_free(self):
        estimate = pool.estimate_scan_cost(10, model="claude-sonnet-5")
        assert estimate["cost"] is None


class TestBuildPool:
    def test_builds_freezes_and_reports(self, store, tmp_path):
        prep = pool.prepare_scan(store, scan_size=100)
        answers = {
            text: rough_of("Technical/Product", "Technical Support", False)
            for _tid, text in prep.survivors
        }
        client = RoughClient(answers)
        stats = pool.build_pool(
            store,
            out_dir=tmp_path,
            target_n=1,
            floor=1,
            client=client,
            prep=prep,
        )
        assert stats["selected"] >= 1
        assert stats["corpus_threads"] > 0
        assert pool.read_pool(tmp_path) == sorted(pool.read_pool(tmp_path))
        # Only one category and queue were scripted, so the other floors must
        # be reported short rather than quietly ignored.
        assert stats["shortfalls"]


class TestLabelCandidatesUsesTheFrozenPool:
    def test_reads_pool_membership_not_a_store_scan(self, db_path, tmp_path):
        conn = open_store(db_path)
        all_ids = [row[0] for row in conn.execute("SELECT thread_id FROM threads")]
        keep = [
            tid for tid in all_ids if degenerate_reason(get_thread(conn, tid)) is None
        ][:1]
        rough = {
            tid: pool.RoughLabels("General Inquiry", "Tier-1 General", False)
            for tid in keep
        }
        pool.write_pool(tmp_path, keep, rough, {"selected": len(keep)})

        items, _excluded = cli._label_candidates(conn, None, tmp_path)
        assert [item.thread_id for item in items] == keep

    def test_shown_text_never_carries_the_rough_label(self, db_path, tmp_path):
        conn = open_store(db_path)
        all_ids = [row[0] for row in conn.execute("SELECT thread_id FROM threads")]
        keep = [
            tid for tid in all_ids if degenerate_reason(get_thread(conn, tid)) is None
        ][:1]
        rough = {tid: pool.RoughLabels("Billing/Payments", "Billing Ops", True) for tid in keep}
        pool.write_pool(tmp_path, keep, rough, {"selected": len(keep)})

        items, _ = cli._label_candidates(conn, None, tmp_path)
        for item in items:
            assert "Billing/Payments" not in item.text
            assert "Billing Ops" not in item.text

    def test_missing_pool_is_an_input_error(self, db_path, tmp_path, capsys):
        code = cli.main(
            ["label", "--pass", "category", "--db", str(db_path), "--out", str(tmp_path)]
        )
        assert code == cli.EXIT_INPUT_ERROR
        assert "triage pool" in capsys.readouterr().err


class TestPoolCli:
    def test_dry_run_makes_no_model_call(self, db_path, tmp_path, capsys):
        class ExplodingClient:
            @property
            def messages(self):
                raise AssertionError("dry run must not touch the LLM client")

        code = cli.main(
            [
                "pool",
                "--db",
                str(db_path),
                "--out",
                str(tmp_path),
                "--dry-run",
            ],
            client=ExplodingClient(),
        )
        out = capsys.readouterr().out
        assert code == cli.EXIT_OK
        assert "dry run: no model calls made." in out
        assert "corpus threads:" in out
        assert not (tmp_path / pool.POOL_FILE).exists()

    def test_dry_run_shows_the_itemized_cost(self, db_path, tmp_path, capsys):
        cli.main(
            ["pool", "--db", str(db_path), "--out", str(tmp_path), "--dry-run"]
        )
        out = capsys.readouterr().out
        assert "claude-haiku-4-5" in out
        assert "passes" in out
        assert "retry = $" in out

    def test_missing_store_is_an_input_error(self, tmp_path, capsys):
        code = cli.main(["pool", "--db", str(tmp_path / "nope.db")])
        assert code == cli.EXIT_INPUT_ERROR

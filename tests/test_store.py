"""Tests for triage.ingest.store (and the ingest CLI path) — KTD9, R25, R29.

Covers:
- CSV loading, including the instructive missing-CSV error naming the download
  command (R25) and the empty-CSV edge.
- End-to-end ingest of the bundled fixture into SQLite, with R31 flags persisted
  and threads queryable by tweet id.
- Distribution-mode metadata defaulting to "pending" (R2 verdict not fabricated)
  and both eval-set export modes (R29).
- The gold-label guard: a re-ingest may not silently move or rewrite a frozen
  label's thread (R11).
- `triage ingest --sample` via cli.main, zero network.

No test touches the network or kagglehub.
"""

import sqlite3
from pathlib import Path

import pytest

from triage.cli import main as cli_main
from triage.ingest import IngestError
from triage.ingest import store as store_mod
from triage.ingest.store import (
    DIST_MODE_ID,
    DIST_MODE_PENDING,
    DIST_MODE_TEXT,
    add_eval_item,
    export_eval_set,
    get_distribution_mode,
    get_thread,
    ingest_csv,
    load_tweets_csv,
    open_store,
    set_distribution_mode,
    thread_ids_for_tweet,
)

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "sample_tweets.csv"

CSV_HEADER = (
    "tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id\n"
)


def _write_csv(path: Path, rows) -> Path:
    """Write a minimal dataset CSV; rows are (tweet_id, author_id, inbound, parent_id).

    ``created_at`` is derived from the tweet id so chronological order matches id
    order (ids must stay below 60).
    """
    lines = [CSV_HEADER]
    for tweet_id, author_id, inbound, parent in rows:
        lines.append(
            f"{tweet_id},{author_id},{inbound},"
            f"Wed Nov 01 12:{tweet_id:02d}:00 +0000 2017,msg {tweet_id},,"
            f"{'' if parent is None else parent}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _membership_snapshot(db_path) -> dict[int, tuple[int, ...]]:
    """Every stored thread's ordered tweet ids, keyed by thread id."""
    conn = open_store(db_path)
    try:
        thread_ids = [row["thread_id"] for row in conn.execute("SELECT thread_id FROM threads")]
        return {tid: get_thread(conn, tid).tweet_ids for tid in thread_ids}
    finally:
        conn.close()


@pytest.fixture
def store_conn(tmp_path):
    db_path = tmp_path / "triage.db"
    stats = ingest_csv(FIXTURE_CSV, db_path)
    conn = open_store(db_path)
    yield conn, stats
    conn.close()


class TestLoadCsv:
    def test_fixture_loads_all_rows(self):
        tweets = load_tweets_csv(FIXTURE_CSV)
        assert len(tweets) == 15

    def test_field_types(self):
        tweets = load_tweets_csv(FIXTURE_CSV)
        reply = tweets[2]
        assert reply.author_id == "sprintcare"
        assert reply.inbound is False
        assert reply.in_response_to_tweet_id == 1
        root = tweets[1]
        assert root.inbound is True
        assert root.in_response_to_tweet_id is None

    def test_missing_csv_names_download_command(self, tmp_path):
        with pytest.raises(IngestError, match="triage ingest --download"):
            load_tweets_csv(tmp_path / "nope" / "twcs.csv")

    def test_empty_csv_loads_no_tweets(self, tmp_path):
        empty = tmp_path / "empty.csv"
        empty.write_text(
            "tweet_id,author_id,inbound,created_at,text,response_tweet_id,"
            "in_response_to_tweet_id\n",
            encoding="utf-8",
        )
        assert load_tweets_csv(empty) == {}


class TestReIngest:
    """Ingest must be re-runnable, and thread ids must survive it.

    Gold labels key on ``thread_id``, so a re-ingest that renumbered threads
    would silently repoint frozen labels at different conversations — the
    published numbers would describe the wrong threads.
    """

    def test_second_ingest_of_the_same_csv_succeeds(self, tmp_path):
        db_path = tmp_path / "store.db"
        first = ingest_csv(FIXTURE_CSV, db_path)
        second = ingest_csv(FIXTURE_CSV, db_path)
        assert second == first

    def test_thread_ids_are_stable_across_re_ingest(self, tmp_path):
        db_path = tmp_path / "store.db"
        ingest_csv(FIXTURE_CSV, db_path)
        conn = open_store(db_path)
        try:
            before = {
                row["thread_id"]: (row["root_tweet_id"], row["customer_author_id"])
                for row in conn.execute(
                    "SELECT thread_id, root_tweet_id, customer_author_id FROM threads"
                )
            }
        finally:
            conn.close()
        ingest_csv(FIXTURE_CSV, db_path)
        conn = open_store(db_path)
        try:
            after = {
                row["thread_id"]: (row["root_tweet_id"], row["customer_author_id"])
                for row in conn.execute(
                    "SELECT thread_id, root_tweet_id, customer_author_id FROM threads"
                )
            }
        finally:
            conn.close()
        assert after == before

    def test_re_ingest_does_not_duplicate_thread_membership(self, tmp_path):
        db_path = tmp_path / "store.db"
        ingest_csv(FIXTURE_CSV, db_path)
        ingest_csv(FIXTURE_CSV, db_path)
        conn = open_store(db_path)
        try:
            (thread_id,) = thread_ids_for_tweet(conn, 1)
            thread = get_thread(conn, thread_id)
            assert [t.tweet_id for t in thread.tweets] == [1, 2, 3, 4]
        finally:
            conn.close()


class TestGoldLabelGuard:
    """A re-ingest must never silently move or rewrite a gold-labeled thread (R11).

    Two hazards, both invisible after the fact: a labeled thread's membership can
    change under a stable ``thread_id``, and a labeled thread can be superseded
    when a previously-missing parent tweet arrives and re-roots the conversation
    under a new ``thread_id``. Either one leaves published metrics describing
    content nobody labeled.
    """

    LINEAR = ((1, "C", True, None), (2, "brand", False, 1), (3, "C", True, 2))
    # Root tweet 1 is absent, so tweet 2 reconstructs as a truncated root (R31).
    TRUNCATED = ((2, "C", True, 1), (3, "brand", False, 2))
    EXTENDED = (*LINEAR, (4, "brand", False, 3))

    def _labeled_store(self, tmp_path, rows=LINEAR):
        csv_path = _write_csv(tmp_path / "a.csv", rows)
        db_path = tmp_path / "store.db"
        ingest_csv(csv_path, db_path)
        conn = open_store(db_path)
        try:
            (thread_id,) = thread_ids_for_tweet(conn, rows[-1][0])
            add_eval_item(conn, thread_id, "billing")
        finally:
            conn.close()
        return db_path, thread_id

    def test_unchanged_re_ingest_of_a_labeled_store_is_silent(self, tmp_path):
        db_path, _ = self._labeled_store(tmp_path)
        before = _membership_snapshot(db_path)
        stats = ingest_csv(tmp_path / "a.csv", db_path)
        assert stats["labeled_thread_hazards"] == []
        assert _membership_snapshot(db_path) == before

    def test_unlabeled_store_absorbs_a_membership_change_silently(self, tmp_path):
        db_path = tmp_path / "store.db"
        ingest_csv(_write_csv(tmp_path / "a.csv", self.LINEAR), db_path)
        stats = ingest_csv(_write_csv(tmp_path / "b.csv", self.EXTENDED), db_path)
        assert stats["labeled_thread_hazards"] == []
        assert sorted(_membership_snapshot(db_path).values()) == [(1, 2, 3, 4)]

    def test_membership_change_under_a_labeled_thread_is_refused(self, tmp_path):
        db_path, thread_id = self._labeled_store(tmp_path)
        with pytest.raises(IngestError) as excinfo:
            ingest_csv(_write_csv(tmp_path / "b.csv", self.EXTENDED), db_path)
        message = str(excinfo.value)
        assert f"thread {thread_id}" in message
        assert "membership" in message
        assert "1,2,3,4" in message  # the operator can see what it became

    def test_refused_ingest_leaves_the_store_untouched(self, tmp_path):
        db_path, _ = self._labeled_store(tmp_path)
        before = _membership_snapshot(db_path)
        with pytest.raises(IngestError):
            ingest_csv(_write_csv(tmp_path / "b.csv", self.EXTENDED), db_path)
        assert _membership_snapshot(db_path) == before
        conn = open_store(db_path)
        try:
            assert conn.execute("SELECT 1 FROM tweets WHERE tweet_id = 4").fetchone() is None
        finally:
            conn.close()

    def test_supersession_by_a_late_parent_is_refused_naming_the_thread(self, tmp_path):
        db_path, thread_id = self._labeled_store(tmp_path, self.TRUNCATED)
        with pytest.raises(IngestError) as excinfo:
            ingest_csv(
                _write_csv(tmp_path / "b.csv", ((1, "C", True, None), *self.TRUNCATED)),
                db_path,
            )
        message = str(excinfo.value)
        assert f"thread {thread_id}" in message
        assert "superseded" in message
        assert "root 1" in message  # where the conversation moved

    def test_labeled_thread_untouched_by_the_new_csv_is_silent(self, tmp_path):
        db_path, _ = self._labeled_store(tmp_path)
        before = _membership_snapshot(db_path)
        other = _write_csv(tmp_path / "c.csv", ((11, "D", True, None), (12, "brand", False, 11)))
        stats = ingest_csv(other, db_path)
        assert stats["labeled_thread_hazards"] == []
        assert before.items() <= _membership_snapshot(db_path).items()

    def test_explicit_acceptance_completes_and_names_the_change(self, tmp_path):
        db_path, thread_id = self._labeled_store(tmp_path)
        stats = ingest_csv(
            _write_csv(tmp_path / "b.csv", self.EXTENDED),
            db_path,
            accept_labeled_thread_changes=True,
        )
        (hazard,) = stats["labeled_thread_hazards"]
        assert hazard["thread_id"] == thread_id
        assert hazard["hazard"] == "membership_changed"
        assert _membership_snapshot(db_path)[thread_id] == (1, 2, 3, 4)

    def test_membership_is_persisted_per_thread(self, tmp_path):
        db_path, thread_id = self._labeled_store(tmp_path)
        conn = open_store(db_path)
        try:
            row = conn.execute(
                "SELECT membership FROM threads WHERE thread_id = ?", (thread_id,)
            ).fetchone()
        finally:
            conn.close()
        assert row["membership"] == "1,2,3"

    def test_store_predating_the_membership_column_is_migrated_and_backfilled(self, tmp_path):
        db_path = tmp_path / "legacy.db"
        legacy = sqlite3.connect(db_path)
        try:
            legacy.executescript(
                "CREATE TABLE threads (thread_id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "root_tweet_id INTEGER NOT NULL, customer_author_id TEXT, "
                "truncated INTEGER NOT NULL, cycle_flagged INTEGER NOT NULL);"
                "CREATE TABLE thread_tweets (thread_id INTEGER NOT NULL, "
                "position INTEGER NOT NULL, tweet_id INTEGER NOT NULL, "
                "PRIMARY KEY (thread_id, position));"
                "INSERT INTO threads VALUES (7, 1, 'C', 0, 0);"
                "INSERT INTO thread_tweets VALUES (7, 0, 1), (7, 1, 2), (7, 2, 3);"
            )
            legacy.commit()
        finally:
            legacy.close()
        conn = open_store(db_path)
        try:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(threads)")}
            row = conn.execute("SELECT membership FROM threads WHERE thread_id = 7").fetchone()
        finally:
            conn.close()
        assert "membership" in columns
        assert row["membership"] == "1,2,3"


class TestIngestEndToEnd:
    def test_stats(self, store_conn):
        _, stats = store_conn
        assert stats == {
            "tweets": 15,
            "threads": 6,
            "truncated": 1,
            "cycle_flagged": 1,
            "labeled_thread_hazards": [],
        }

    def test_mid_thread_tweet_resolves_to_same_thread_as_root(self, store_conn):
        conn, _ = store_conn
        assert thread_ids_for_tweet(conn, 3) == thread_ids_for_tweet(conn, 1)

    def test_thread_query_returns_chronological_texts(self, store_conn):
        conn, _ = store_conn
        (thread_id,) = thread_ids_for_tweet(conn, 1)
        thread = get_thread(conn, thread_id)
        assert [t.tweet_id for t in thread.tweets] == [1, 2, 3, 4]
        assert thread.tweets[0].text.startswith("@sprintcare my data plan")

    def test_truncation_flag_persisted(self, store_conn):
        conn, _ = store_conn
        (thread_id,) = thread_ids_for_tweet(conn, 11)
        thread = get_thread(conn, thread_id)
        assert thread.truncated is True
        assert thread.root_tweet_id == 10

    def test_cycle_flag_persisted(self, store_conn):
        conn, _ = store_conn
        (thread_id,) = thread_ids_for_tweet(conn, 15)
        thread = get_thread(conn, thread_id)
        assert thread.cycle_flagged is True

    def test_shared_prefix_tweet_belongs_to_both_branch_threads(self, store_conn):
        conn, _ = store_conn
        # Tweet 21 (brand reply) is on both customers' branches; 22 only on B's.
        assert len(thread_ids_for_tweet(conn, 21)) == 2
        (b_thread_id,) = thread_ids_for_tweet(conn, 22)
        b_thread = get_thread(conn, b_thread_id)
        assert [t.tweet_id for t in b_thread.tweets] == [20, 21, 22]

    def test_unknown_thread_id_raises(self, store_conn):
        conn, _ = store_conn
        with pytest.raises(IngestError, match="9999"):
            get_thread(conn, 9999)

    def test_empty_csv_ingests_zero_threads(self, tmp_path):
        empty = tmp_path / "empty.csv"
        empty.write_text(
            "tweet_id,author_id,inbound,created_at,text,response_tweet_id,"
            "in_response_to_tweet_id\n",
            encoding="utf-8",
        )
        stats = ingest_csv(empty, tmp_path / "empty.db")
        assert stats["tweets"] == 0
        assert stats["threads"] == 0


class TestDistributionMode:
    def test_defaults_to_pending(self, store_conn):
        conn, _ = store_conn
        assert get_distribution_mode(conn) == DIST_MODE_PENDING

    def test_set_and_get(self, store_conn):
        conn, _ = store_conn
        set_distribution_mode(conn, DIST_MODE_TEXT)
        assert get_distribution_mode(conn) == DIST_MODE_TEXT

    def test_invalid_mode_raises(self, store_conn):
        conn, _ = store_conn
        with pytest.raises(IngestError, match="mode"):
            set_distribution_mode(conn, "shrug")


class TestEvalSetExport:
    def _label_linear_thread(self, conn) -> int:
        (thread_id,) = thread_ids_for_tweet(conn, 1)
        add_eval_item(conn, thread_id, "service-outage")
        return thread_id

    def test_text_mode_includes_full_tweet_text(self, store_conn):
        conn, _ = store_conn
        thread_id = self._label_linear_thread(conn)
        export = export_eval_set(conn, DIST_MODE_TEXT)
        assert export["mode"] == DIST_MODE_TEXT
        (item,) = export["items"]
        assert item["thread_id"] == thread_id
        assert item["label"] == "service-outage"
        assert [t["tweet_id"] for t in item["tweets"]] == [1, 2, 3, 4]
        assert "data plan" in item["tweets"][0]["text"]

    def test_id_mode_ships_ids_and_rebuild_command_only(self, store_conn):
        conn, _ = store_conn
        self._label_linear_thread(conn)
        export = export_eval_set(conn, DIST_MODE_ID)
        assert export["mode"] == DIST_MODE_ID
        assert "triage ingest --download" in export["rebuild_command"]
        (item,) = export["items"]
        assert item["tweet_ids"] == [1, 2, 3, 4]
        assert "tweets" not in item  # no text redistribution in ID mode
        assert "text" not in str(item.keys())

    def test_pending_mode_is_not_exportable(self, store_conn):
        conn, _ = store_conn
        with pytest.raises(IngestError, match="R2"):
            export_eval_set(conn, DIST_MODE_PENDING)

    def test_label_unknown_thread_raises(self, store_conn):
        conn, _ = store_conn
        with pytest.raises(IngestError, match="4242"):
            add_eval_item(conn, 4242, "whatever")


class TestCli:
    def test_ingest_sample_builds_store_without_network(self, tmp_path, capsys):
        db_path = tmp_path / "sample.db"
        rc = cli_main(["ingest", "--sample", "--db", str(db_path)])
        assert rc == 0
        assert db_path.is_file()
        out = capsys.readouterr().out
        assert "6 threads" in out
        assert "R2" in out  # manual license-verdict reminder surfaced

    def test_ingest_missing_csv_is_instructive(self, tmp_path, capsys):
        rc = cli_main(
            ["ingest", "--csv", str(tmp_path / "absent.csv"), "--db", str(tmp_path / "x.db")]
        )
        assert rc == 1
        err = capsys.readouterr().err
        assert "triage ingest --download" in err

    def test_no_command_prints_usage(self, capsys):
        assert cli_main([]) == 0
        assert "usage" in capsys.readouterr().out.lower()

    def test_sample_store_is_queryable(self, tmp_path):
        db_path = tmp_path / "sample.db"
        assert cli_main(["ingest", "--sample", "--db", str(db_path)]) == 0
        conn = store_mod.open_store(db_path)
        try:
            (thread_id,) = thread_ids_for_tweet(conn, 30)
            thread = get_thread(conn, thread_id)
            assert thread.tweets[0].author_id == "100007"
        finally:
            conn.close()

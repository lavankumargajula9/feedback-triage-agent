"""Tests for triage.ingest.reconstruct — deterministic thread reconstruction (R1, R31).

The R31 contract under test:
- walk parent links to the root or to the first missing parent (flagging truncation);
- the thread is the single root-to-leaf branch containing the requested tweet plus
  direct brand replies, in chronological order;
- cycles are guarded by a visited set and flagged;
- sibling replies from other customers are excluded.

All fixtures are hand-crafted in-memory; no network, no files.
"""

import pytest

from triage.ingest import IngestError
from triage.ingest.reconstruct import Tweet, reconstruct_all, reconstruct_thread


def ts(minute: int, hour: int = 10) -> str:
    """A twcs-format timestamp (``%a %b %d %H:%M:%S %z %Y``)."""
    return f"Wed Nov 01 {hour:02d}:{minute:02d}:00 +0000 2017"


def tw(
    tid: int,
    author: str,
    inbound: bool,
    created_at: str,
    parent: int | None = None,
) -> Tweet:
    return Tweet(
        tweet_id=tid,
        author_id=author,
        inbound=inbound,
        created_at=created_at,
        text=f"tweet {tid}",
        in_response_to_tweet_id=parent,
    )


def by_id(*tweets: Tweet) -> dict[int, Tweet]:
    return {t.tweet_id: t for t in tweets}


# --- Fixtures -----------------------------------------------------------------

def linear_thread() -> dict[int, Tweet]:
    """customer -> brand -> customer -> brand, a plain reply chain."""
    return by_id(
        tw(1, "100001", True, ts(0)),
        tw(2, "sprintcare", False, ts(5), parent=1),
        tw(3, "100001", True, ts(10), parent=2),
        tw(4, "sprintcare", False, ts(15), parent=3),
    )


def orphaned_thread() -> dict[int, Tweet]:
    """Tweet 10's parent (999) is absent from the dataset."""
    return by_id(
        tw(10, "100002", True, ts(0), parent=999),
        tw(11, "AppleSupport", False, ts(5), parent=10),
    )


def cyclic_thread() -> dict[int, Tweet]:
    """15 and 16 each claim the other as parent."""
    return by_id(
        tw(15, "100004", True, ts(0), parent=16),
        tw(16, "Uber_Support", False, ts(5), parent=15),
    )


def multi_customer_thread() -> dict[int, Tweet]:
    """Customer B butts into A's thread; brand double-replies to A's follow-up."""
    return by_id(
        tw(20, "100005", True, ts(0)),
        tw(21, "AmazonHelp", False, ts(5), parent=20),
        tw(22, "100006", True, ts(10), parent=21),  # other customer's sibling reply
        tw(23, "100005", True, ts(15), parent=21),
        tw(24, "AmazonHelp", False, ts(20), parent=23),
        tw(25, "AmazonHelp", False, ts(20), parent=23),  # same timestamp as 24
    )


class TestHappyPath:
    def test_linear_chain_in_chronological_order(self):
        thread = reconstruct_thread(1, linear_thread())
        assert [t.tweet_id for t in thread.tweets] == [1, 2, 3, 4]

    def test_brand_replies_included(self):
        thread = reconstruct_thread(1, linear_thread())
        assert [t.author_id for t in thread.tweets if not t.inbound] == [
            "sprintcare",
            "sprintcare",
        ]

    def test_flags_clear_on_clean_thread(self):
        thread = reconstruct_thread(1, linear_thread())
        assert thread.truncated is False
        assert thread.cycle_flagged is False

    def test_root_and_customer_identified(self):
        thread = reconstruct_thread(1, linear_thread())
        assert thread.root_tweet_id == 1
        assert thread.customer_author_id == "100001"

    def test_order_is_chronological_not_by_id(self):
        # Tweet ids deliberately not monotonic with time.
        tweets = by_id(
            tw(50, "100009", True, ts(0)),
            tw(52, "brandX", False, ts(5), parent=50),
            tw(51, "100009", True, ts(10), parent=52),
        )
        thread = reconstruct_thread(50, tweets)
        assert [t.tweet_id for t in thread.tweets] == [50, 52, 51]


class TestThreadIdentity:
    def test_mid_thread_id_resolves_to_same_thread_as_root(self):
        tweets = linear_thread()
        assert reconstruct_thread(3, tweets) == reconstruct_thread(1, tweets)

    def test_leaf_brand_reply_resolves_to_same_thread(self):
        tweets = linear_thread()
        assert reconstruct_thread(4, tweets) == reconstruct_thread(1, tweets)


class TestTruncation:
    def test_missing_parent_flags_truncated(self):
        thread = reconstruct_thread(11, orphaned_thread())
        assert thread.truncated is True
        assert thread.cycle_flagged is False

    def test_truncated_root_is_first_reachable_ancestor(self):
        thread = reconstruct_thread(11, orphaned_thread())
        assert thread.root_tweet_id == 10
        assert [t.tweet_id for t in thread.tweets] == [10, 11]


class TestCycles:
    def test_cycle_terminates_and_flags(self):
        thread = reconstruct_thread(15, cyclic_thread())
        assert thread.cycle_flagged is True
        assert sorted(t.tweet_id for t in thread.tweets) == [15, 16]

    def test_cycle_same_thread_from_either_entry_point(self):
        tweets = cyclic_thread()
        assert reconstruct_thread(15, tweets) == reconstruct_thread(16, tweets)


class TestSiblingExclusion:
    def test_other_customer_sibling_excluded(self):
        thread = reconstruct_thread(20, multi_customer_thread())
        ids = [t.tweet_id for t in thread.tweets]
        assert 22 not in ids
        assert ids == [20, 21, 23, 24, 25]

    def test_other_customer_gets_own_thread(self):
        thread = reconstruct_thread(22, multi_customer_thread())
        assert [t.tweet_id for t in thread.tweets] == [20, 21, 22]
        assert thread.customer_author_id == "100006"

    def test_both_direct_brand_replies_included(self):
        thread = reconstruct_thread(23, multi_customer_thread())
        ids = [t.tweet_id for t in thread.tweets]
        assert 24 in ids and 25 in ids

    def test_timestamp_tie_broken_by_tweet_id(self):
        thread = reconstruct_thread(20, multi_customer_thread())
        ids = [t.tweet_id for t in thread.tweets]
        assert ids.index(24) < ids.index(25)


class TestEdges:
    def test_single_tweet_thread(self):
        tweets = by_id(tw(30, "100007", True, ts(0)))
        thread = reconstruct_thread(30, tweets)
        assert [t.tweet_id for t in thread.tweets] == [30]
        assert thread.root_tweet_id == 30
        assert thread.truncated is False
        assert thread.cycle_flagged is False

    def test_unknown_tweet_id_raises(self):
        with pytest.raises(IngestError, match="12345"):
            reconstruct_thread(12345, linear_thread())

    def test_empty_dataset_reconstruct_all(self):
        assert reconstruct_all({}) == []


class TestReconstructAll:
    def all_fixture_tweets(self) -> dict[int, Tweet]:
        merged: dict[int, Tweet] = {}
        for fixture in (
            linear_thread(),
            orphaned_thread(),
            cyclic_thread(),
            multi_customer_thread(),
        ):
            merged.update(fixture)
        merged.update(by_id(tw(30, "100007", True, ts(0))))
        return merged

    def test_deduplicates_threads(self):
        threads = reconstruct_all(self.all_fixture_tweets())
        # linear, orphaned, cyclic, multi-customer A, multi-customer B, single.
        assert len(threads) == 6

    def test_flag_counts(self):
        threads = reconstruct_all(self.all_fixture_tweets())
        assert sum(1 for t in threads if t.truncated) == 1
        assert sum(1 for t in threads if t.cycle_flagged) == 1

    def test_every_tweet_appears_in_some_thread(self):
        tweets = self.all_fixture_tweets()
        covered: set[int] = set()
        for thread in reconstruct_all(tweets):
            covered.update(t.tweet_id for t in thread.tweets)
        assert covered == set(tweets)

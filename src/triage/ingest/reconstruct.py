"""Deterministic conversation-thread reconstruction from reply-chain fields (R1, R31).

The reconstruction contract (R31), implemented exactly:

- Walk parent links (``in_response_to_tweet_id``) from the requested tweet to the
  root, or to the first missing parent — flagging truncation in the latter case.
- The thread is the single root-to-leaf branch containing the requested tweet,
  plus direct brand replies, in chronological order.
- Cycles are guarded by a visited set and flagged (``cycle_flagged``).
- Sibling replies from other customers are excluded.

``response_tweet_id`` is treated as secondary/derivable; only parent links drive
reconstruction. Chronological order is by parsed ``created_at``, with ties broken
by ``tweet_id`` so ordering is fully deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from triage.ingest import IngestError

_EPOCH = datetime.min.replace(tzinfo=UTC)


@dataclass(frozen=True)
class Tweet:
    """One row of the dataset, keyed by ``tweet_id``."""

    tweet_id: int
    author_id: str
    inbound: bool  # True = customer-authored
    created_at: str
    text: str
    in_response_to_tweet_id: int | None


@dataclass(frozen=True)
class Thread:
    """A reconstructed conversation thread — the unit of triage everywhere (R1)."""

    root_tweet_id: int
    customer_author_id: str | None
    tweets: tuple[Tweet, ...]  # chronological order
    truncated: bool
    cycle_flagged: bool

    @property
    def tweet_ids(self) -> tuple[int, ...]:
        return tuple(t.tweet_id for t in self.tweets)

    @property
    def signature(self) -> tuple[int, str | None]:
        """Identity of a thread: its root plus the customer whose branch it is."""
        return (self.root_tweet_id, self.customer_author_id)


def parse_created_at(value: str) -> datetime | None:
    """Parse a twcs (or ISO-8601) timestamp; None when unparseable."""
    for attempt in (
        # twcs format, e.g. "Tue Oct 31 22:10:47 +0000 2017" (aware via %z).
        lambda v: datetime.strptime(v, "%a %b %d %H:%M:%S %z %Y"),
        datetime.fromisoformat,
    ):
        try:
            parsed = attempt(value.strip())
        except (ValueError, AttributeError):
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed
    return None


def _sort_key(tweet: Tweet) -> tuple[datetime, int]:
    return (parse_created_at(tweet.created_at) or _EPOCH, tweet.tweet_id)


def build_child_index(tweets: dict[int, Tweet]) -> dict[int, list[int]]:
    """Map each tweet id to the ids of tweets replying to it (derived, not trusted
    from ``response_tweet_id``)."""
    children: dict[int, list[int]] = {}
    for tweet in tweets.values():
        parent = tweet.in_response_to_tweet_id
        if parent is not None:
            children.setdefault(parent, []).append(tweet.tweet_id)
    for sibling_ids in children.values():
        sibling_ids.sort()
    return children


def _downstream_customer(
    chain: list[int], tweets: dict[int, Tweet], children: dict[int, list[int]]
) -> str | None:
    """The customer of the branch below a chain with no customer of its own (R31).

    Breadth-first with tweet-id tiebreak, so several customers replying to the
    same parentless brand tweet resolve deterministically.
    """
    seen = set(chain)
    frontier = list(chain)
    while frontier:
        inbound_ids = sorted(
            child_id
            for node in frontier
            for child_id in children.get(node, ())
            if child_id in tweets and tweets[child_id].inbound
        )
        if inbound_ids:
            return tweets[inbound_ids[0]].author_id
        next_frontier = []
        for node in frontier:
            for child_id in children.get(node, ()):
                if child_id not in seen and child_id in tweets:
                    seen.add(child_id)
                    next_frontier.append(child_id)
        frontier = sorted(next_frontier)
    return None


def reconstruct_thread(
    tweet_id: int,
    tweets: dict[int, Tweet],
    children: dict[int, list[int]] | None = None,
) -> Thread:
    """Reconstruct the thread containing ``tweet_id`` per the R31 contract.

    ``children`` may be passed to amortize index construction across calls
    (see :func:`reconstruct_all`); it must be ``build_child_index(tweets)``.
    """
    if tweet_id not in tweets:
        raise IngestError(
            f"Tweet id {tweet_id} is not present in the dataset; cannot reconstruct "
            "a thread for it."
        )
    if children is None:
        children = build_child_index(tweets)

    # Upward walk: requested -> root (or first missing parent / cycle closure).
    chain: list[int] = []  # requested first, root last
    visited: set[int] = set()
    truncated = False
    cycle_flagged = False
    current = tweet_id
    while True:
        if current in visited:
            cycle_flagged = True
            break
        visited.add(current)
        chain.append(current)
        parent = tweets[current].in_response_to_tweet_id
        if parent is None:
            break
        if parent not in tweets:
            truncated = True
            break
        current = parent

    # Root: deepest reachable ancestor. In a cycle there is no true root, so the
    # minimum id in the cycle chain is used — canonical from any entry point.
    root_id = min(chain) if cycle_flagged else chain[-1]

    # The thread's customer: the requested tweet's author if customer-authored,
    # else the nearest customer-authored ancestor (chain is nearest-first).
    customer: str | None = None
    for chain_id in chain:
        if tweets[chain_id].inbound:
            customer = tweets[chain_id].author_id
            break
    if customer is None:
        customer = _downstream_customer(chain, tweets, children)

    # Downward walk from the chain: follow the customer's branch and direct brand
    # replies; sibling replies from other customers are excluded (R31).
    included: set[int] = set(chain)
    frontier: list[int] = list(chain)
    while frontier:
        node = frontier.pop()
        for child_id in children.get(node, ()):
            if child_id in included:
                continue  # visited-set guard (covers reply cycles)
            child = tweets.get(child_id)
            if child is None:
                continue
            if not child.inbound or child.author_id == customer:
                included.add(child_id)
                frontier.append(child_id)

    ordered = tuple(sorted((tweets[i] for i in included), key=_sort_key))
    return Thread(
        root_tweet_id=root_id,
        customer_author_id=customer,
        tweets=ordered,
        truncated=truncated,
        cycle_flagged=cycle_flagged,
    )


def reconstruct_all(tweets: dict[int, Tweet]) -> list[Thread]:
    """Reconstruct every distinct thread in the dataset.

    Threads are deduplicated by :attr:`Thread.signature` (root id, customer), so a
    brand reply shared by two customers' branches yields two threads with a common
    prefix rather than one merged thread.
    """
    children = build_child_index(tweets)
    threads: dict[tuple[int, str | None], Thread] = {}
    done: set[int] = set()
    for tweet_id in sorted(tweets):
        if tweet_id in done:
            continue
        thread = reconstruct_thread(tweet_id, tweets, children)
        threads.setdefault(thread.signature, thread)
        if not thread.cycle_flagged:
            # Any member tweet whose own reconstruction provably yields this same
            # signature need not be revisited: the requested tweet itself, and
            # this customer's tweets on the branch.
            done.add(tweet_id)
            for member in thread.tweets:
                if member.inbound and member.author_id == thread.customer_author_id:
                    done.add(member.tweet_id)
    return list(threads.values())

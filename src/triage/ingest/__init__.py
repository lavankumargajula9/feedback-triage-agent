"""Dataset ingestion: download, thread reconstruction, and SQLite store (R1, R25, KTD9).

Submodules:

- ``download``: reproducible kagglehub acquisition of the pinned dataset (R25).
- ``reconstruct``: deterministic reply-chain thread reconstruction (R31).
- ``store``: SQLite thread store with dual-mode eval-set export (KTD9, R29).
"""

from __future__ import annotations


class IngestError(Exception):
    """Raised for ingestion failures: missing dataset files, bad rows, store misuse."""


__all__ = ["IngestError"]

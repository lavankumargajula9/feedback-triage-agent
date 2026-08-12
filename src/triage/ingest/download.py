"""Reproducible dataset acquisition via kagglehub (R25, R2).

The dataset handle is pinned here and nowhere else; every "CSV is missing" error
elsewhere must name :data:`DOWNLOAD_COMMAND` so the reproduction path is always
one instruction away (R25).

Note (R2): downloading says nothing about authenticity or redistribution rights.
That verdict is a manual human step, recorded in the README data section and in
the store's distribution mode — see ``triage.ingest.store``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from triage.ingest import IngestError

# Pinned Kaggle dataset (R25). Single CSV of customer-support tweets.
DATASET_HANDLE = "thoughtvector/customer-support-on-twitter"
CSV_FILENAME = "twcs.csv"

DEFAULT_RAW_DIR = Path("data") / "raw"
DEFAULT_CSV_PATH = DEFAULT_RAW_DIR / CSV_FILENAME

# The exact command users are told to run whenever the CSV is missing.
DOWNLOAD_COMMAND = "triage ingest --download"


def missing_csv_error(csv_path: Path | str) -> IngestError:
    """The instructive error for a missing dataset CSV (R25)."""
    return IngestError(
        f"Dataset CSV not found at {csv_path}. Run `{DOWNLOAD_COMMAND}` to fetch the "
        f"pinned Kaggle dataset {DATASET_HANDLE!r} via kagglehub into {DEFAULT_RAW_DIR}/ "
        f"(or pass --csv with the path to an existing {CSV_FILENAME})."
    )


def download_dataset(dest_dir: Path | str = DEFAULT_RAW_DIR) -> Path:
    """Fetch the pinned dataset into ``dest_dir`` and return the local CSV path.

    kagglehub is imported lazily so that nothing else in the package — including
    the test suite — ever touches it or the network.
    """
    import kagglehub  # lazy by design; see docstring.

    cache_dir = Path(kagglehub.dataset_download(DATASET_HANDLE))
    source = cache_dir / CSV_FILENAME
    if not source.is_file():
        matches = sorted(cache_dir.rglob(CSV_FILENAME))
        if not matches:
            raise IngestError(
                f"kagglehub downloaded {DATASET_HANDLE!r} to {cache_dir} but no "
                f"{CSV_FILENAME} was found there; the dataset layout may have changed."
            )
        source = matches[0]

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / CSV_FILENAME
    shutil.copyfile(source, dest)
    return dest

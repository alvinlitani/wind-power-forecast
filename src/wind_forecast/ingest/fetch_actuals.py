"""
Fetch the latest IESO Generator Output and Capability monthly CSV.

Downloads the current month's CSV (always re-downloaded since IESO updates
it throughout the month). If today is the 1st or 2nd of the month, also
downloads the previous month's CSV to ensure the last 48 hours of data
are available.

Saves to <DATA_ROOT>/raw/ieso/PUB_GenOutputCapabilityMonth_YYYYMM.csv

Usage:
    python -m wind_forecast.ingest.fetch_actuals
    python -m wind_forecast.ingest.fetch_actuals --output-dir gs://bucket/raw/ieso
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone

import requests

from wind_forecast import storage


BASE_URL = "https://reports-public.ieso.ca/public/GenOutputCapabilityMonth"
FILENAME_TEMPLATE = "PUB_GenOutputCapabilityMonth_{yyyymm}.csv"

# If today is within this many days of the start of the month,
# also download the previous month's CSV
MONTH_BOUNDARY_DAYS = 2


def download_month(yyyymm: str, output_dir: str) -> bool:
    """Download a monthly CSV from IESO.

    Always overwrites existing files since the current month's CSV
    is updated throughout the month.

    Args:
        yyyymm: Year-month string (e.g., '202605').
        output_dir: Root output directory (e.g., data/raw/ieso). May be a
            local path or a gs:// URI.

    Returns:
        True if successful, False otherwise.
    """
    filename = FILENAME_TEMPLATE.format(yyyymm=yyyymm)
    url = f"{BASE_URL}/{filename}"
    dest = f"{output_dir.rstrip('/')}/{filename}"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  FAIL {filename}: {e}")
        return False

    storage.write_bytes(resp.content, dest)
    size_kb = len(resp.content) / 1024
    print(f"  OK   {filename} ({size_kb:.0f} KB)")
    return True


def get_months_to_download(now: datetime) -> list[str]:
    """Determine which monthly CSVs to download.

    Always includes the current month. If today is within
    MONTH_BOUNDARY_DAYS of the 1st, also includes the previous month.

    Returns:
        List of YYYYMM strings to download.
    """
    months = [now.strftime("%Y%m")]

    if now.day <= MONTH_BOUNDARY_DAYS:
        prev = now.replace(day=1) - timedelta(days=1)
        months.insert(0, prev.strftime("%Y%m"))

    return months

def fetch_actuals(output_dir: str, now: datetime | None = None) -> dict:
    """Download the monthly CSV(s) the daily pipeline needs, with a tolerant
    policy for the current month.

    Near the start of a month the current month isn't published yet, so a
    failed download of *that* month is acceptable and recorded as skipped.
    Any previous month requested by get_months_to_download is required — its
    failure raises, so a genuine transient error on data that should exist is
    surfaced (and, under Prefect, retried).

    Returns {"downloaded": [...], "skipped": [...]}.
    Raises RuntimeError if a required (non-current) month fails.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    months = get_months_to_download(now)
    current = now.strftime("%Y%m")

    downloaded, skipped = [], []
    for yyyymm in months:
        if download_month(yyyymm, output_dir):
            downloaded.append(yyyymm)
        elif yyyymm == current:
            skipped.append(yyyymm)  # not published yet — acceptable
        else:
            raise RuntimeError(f"Required IESO month failed to download: {yyyymm}")
    return {"downloaded": downloaded, "skipped": skipped}

def main():
    parser = argparse.ArgumentParser(
        description="Fetch latest IESO monthly CSV(s) for daily pipeline."
    )
    parser.add_argument(
        "--output-dir",
        default=storage.data_path("raw", "ieso"),
        help="Root output directory (local or gs://). Defaults to <DATA_ROOT>/raw/ieso",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    now = datetime.now(timezone.utc)

    result = fetch_actuals(output_dir, now=now)
    print(f"Downloaded: {result['downloaded']}")
    if result["skipped"]:
        print(f"Skipped (not published yet): {result['skipped']}")
    print(f"Files saved to {output_dir}/")


if __name__ == "__main__":
    main()

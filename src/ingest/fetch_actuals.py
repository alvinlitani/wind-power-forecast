"""
Fetch the latest IESO Generator Output and Capability monthly CSV.

Downloads the current month's CSV (always re-downloaded since IESO updates
it throughout the month). If today is the 1st or 2nd of the month, also
downloads the previous month's CSV to ensure the last 48 hours of data
are available.

Saves to data/raw/ieso/{year}/PUB_GenOutputCapabilityMonth_YYYYMM.csv

Usage:
    python fetch_actuals.py
    python fetch_actuals.py --output-dir ../../data/raw/ieso
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests


BASE_URL = "https://reports-public.ieso.ca/public/GenOutputCapabilityMonth"
FILENAME_TEMPLATE = "PUB_GenOutputCapabilityMonth_{yyyymm}.csv"

# If today is within this many days of the start of the month,
# also download the previous month's CSV
MONTH_BOUNDARY_DAYS = 2


def download_month(yyyymm: str, output_dir: Path) -> bool:
    """Download a monthly CSV from IESO.

    Always overwrites existing files since the current month's CSV
    is updated throughout the month.

    Args:
        yyyymm: Year-month string (e.g., '202605').
        output_dir: Root output directory (e.g., data/raw/ieso).

    Returns:
        True if successful, False otherwise.
    """
    filename = FILENAME_TEMPLATE.format(yyyymm=yyyymm)
    url = f"{BASE_URL}/{filename}"
    year = yyyymm[:4]
    year_dir = output_dir / year
    year_dir.mkdir(parents=True, exist_ok=True)
    dest = year_dir / filename

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  FAIL {filename}: {e}")
        return False

    dest.write_bytes(resp.content)
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


def main():
    parser = argparse.ArgumentParser(
        description="Fetch latest IESO monthly CSV(s) for daily pipeline."
    )
    parser.add_argument(
        "--output-dir",
        default="../../data/raw/ieso",
        help="Root output directory (default: data/raw/ieso)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    now = datetime.now(timezone.utc)

    months = get_months_to_download(now)
    print(f"Date: {now.strftime('%Y-%m-%d')}")
    print(f"Downloading {len(months)} monthly CSV(s)...\n")

    failed = []
    for yyyymm in months:
        if not download_month(yyyymm, output_dir):
            failed.append(yyyymm)

    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"\nDone. Files saved to {output_dir}/")


if __name__ == "__main__":
    main()

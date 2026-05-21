"""
Download IESO Generator Output and Capability monthly CSV files.

Backfill mode: downloads all monthly files for a given date range
and organizes them into year-based subdirectories.

Usage:
    python download_ieso.py --start 2023-01 --end 2026-04 --output data/raw/ieso

Produces:
    data/raw/ieso/2023/PUB_GenOutputCapabilityMonth_202301.csv
    data/raw/ieso/2023/PUB_GenOutputCapabilityMonth_202302.csv
    ...
    data/raw/ieso/2026/PUB_GenOutputCapabilityMonth_202604.csv
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import requests

BASE_URL = "https://reports-public.ieso.ca/public/GenOutputCapabilityMonth"
FILENAME_TEMPLATE = "PUB_GenOutputCapabilityMonth_{yyyymm}.csv"


def generate_months(start: str, end: str) -> list[str]:
    """Generate list of YYYYMM strings from start to end (inclusive).

    Args:
        start: 'YYYY-MM' format
        end: 'YYYY-MM' format
    """
    start_dt = datetime.strptime(start, "%Y-%m")
    end_dt = datetime.strptime(end, "%Y-%m")

    months = []
    current = start_dt
    while current <= end_dt:
        months.append(current.strftime("%Y%m"))
        # Advance to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return months


def download_month(yyyymm: str, output_dir: Path) -> bool:
    """Download a single monthly CSV from IESO.

    Returns True if successful, False otherwise.
    """
    filename = FILENAME_TEMPLATE.format(yyyymm=yyyymm)
    url = f"{BASE_URL}/{filename}"
    year = yyyymm[:4]
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / filename

    if dest.exists():
        print(f"  SKIP {filename} (already exists)")
        return True

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  FAIL {filename}: {e}")
        return False

    dest.write_bytes(resp.content)
    print(f"  OK   {filename}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download IESO Generator Output and Capability monthly CSVs."
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start month in YYYY-MM format (e.g. 2023-01)",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End month in YYYY-MM format (e.g. 2026-04)",
    )
    parser.add_argument(
        "--output",
        default="../../data/raw/ieso",
        help="Base output directory (default: data/raw/ieso)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    months = generate_months(args.start, args.end)
    print(f"Downloading {len(months)} monthly files to {output_dir}/")

    success = 0
    failed = 0
    for yyyymm in months:
        if download_month(yyyymm, output_dir):
            success += 1
        else:
            failed += 1

    print(f"\nDone: {success} downloaded, {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
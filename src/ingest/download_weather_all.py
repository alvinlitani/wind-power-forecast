"""
Download weather data for all generators listed in mapping.csv.

Calls download_weather.py for each generator sequentially with a delay
between requests to avoid rate limiting.

Usage:
    python download_weather_all.py --start 2023-01-01 --end 2023-03-31
    python download_weather_all.py --start 2023-01-01 --end 2023-03-31 --delay 5
"""

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


def load_generator_names(mapping_path: Path) -> list[str]:
    """Load IESO generator names from mapping CSV."""
    names = []
    with open(mapping_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row["IESO name"])
    return names


def main():
    parser = argparse.ArgumentParser(
        description="Download weather data for all generators in mapping.csv."
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--mapping",
        default="../../data/mapping.csv",
        help="Path to generator mapping CSV (default: data/mapping.csv)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=60.0,
        help="Seconds to wait between API calls (default: 60.0)",
    )
    args = parser.parse_args()

    mapping_path = Path(args.mapping)
    if not mapping_path.exists():
        print(f"Mapping file not found: {mapping_path}")
        sys.exit(1)

    names = load_generator_names(mapping_path)
    print(f"Downloading weather for {len(names)} generators")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Delay between requests: {args.delay}s\n")

    success = 0
    failed = 0
    for i, name in enumerate(names):
        print(f"[{i + 1}/{len(names)}] {name}")
        result = subprocess.run(
            [
                sys.executable,
                "download_weather.py",
                "--name", name,
                "--start", args.start,
                "--end", args.end,
                "--mapping", str(args.mapping),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(result.stdout.rstrip())
            success += 1
        else:
            print(f"  FAIL: {result.stderr.rstrip()}")
            failed += 1

        if i < len(names) - 1:
            time.sleep(args.delay)

    print(f"\nDone: {success} succeeded, {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
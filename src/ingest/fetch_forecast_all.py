"""
Fetch weather forecasts for all wind farm sites.

Reads the site list from mapping.csv and calls fetch_forecast.py
as a subprocess for each site. Continues through all sites on failure
and reports a summary at the end.

Usage:
    python fetch_forecast_all.py
    python fetch_forecast_all.py --mapping-csv ../../data/mapping.csv
"""

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


def load_site_names(mapping_path: Path) -> list[str]:
    """Extract unique IESO names from mapping.csv.

    Returns:
        Sorted list of IESO generator names.
    """
    names = set()
    with open(mapping_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.add(row["IESO name"])
    return sorted(names)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch weather forecasts for all sites."
    )
    parser.add_argument(
        "--mapping-csv",
        default="../../data/mapping.csv",
        help="Path to mapping.csv",
    )
    parser.add_argument(
        "--forecast-script",
        default="fetch_forecast.py",
        help="Path to fetch_forecast.py (default: same directory)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (passed to fetch_forecast.py)",
    )
    args = parser.parse_args()

    mapping_path = Path(args.mapping_csv)
    if not mapping_path.exists():
        print(f"Mapping file not found: {mapping_path}")
        sys.exit(1)

    forecast_script = Path(args.forecast_script)
    if not forecast_script.exists():
        print(f"Forecast script not found: {forecast_script}")
        sys.exit(1)

    # --- Load site list ---
    print(f"Loading sites from {mapping_path}...")
    sites = load_site_names(mapping_path)
    print(f"  Found {len(sites)} sites\n")

    # --- Fetch forecasts for each site ---
    results = []
    total_start = time.time()

    for i, name in enumerate(sites, 1):
        print(f"[{i}/{len(sites)}] {name}")

        cmd = [
            sys.executable, str(forecast_script),
            "--name", name,
            "--mapping-csv", str(mapping_path),
        ]
        if args.output_dir:
            cmd += ["--output-dir", args.output_dir]

        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0

        status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        results.append((name, result.returncode, elapsed))

        print(f"  {status} ({elapsed:.1f}s)\n")

    # --- Summary ---
    total_elapsed = time.time() - total_start
    succeeded = sum(1 for _, rc, _ in results if rc == 0)
    failed = [r for r in results if r[1] != 0]

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Total sites: {len(results)}")
    print(f"  Succeeded:   {succeeded}")
    print(f"  Failed:      {len(failed)}")
    print(f"  Total time:  {total_elapsed:.1f}s")

    if failed:
        print(f"\nFailed sites:")
        for name, rc, elapsed in failed:
            print(f"  {name} - exit code {rc}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

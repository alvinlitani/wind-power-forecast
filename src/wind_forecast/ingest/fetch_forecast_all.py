"""
Fetch weather forecasts for all wind farm sites.

Reads the site list from mapping.csv and fetches each site's forecast
in-process by calling fetch_forecast.fetch_site(). Continues through all
sites on failure and reports a summary at the end.

Running in-process (rather than spawning a subprocess per site) avoids
paying Python/import startup cost 45 times and keeps everything inside a
single container process for the daily flow. A per-site try/except
preserves the "one failure doesn't stop the run" behavior.

Usage:
    python -m wind_forecast.ingest.fetch_forecast_all
    python -m wind_forecast.ingest.fetch_forecast_all --mapping-csv gs://bucket/mapping.csv
"""

import argparse
import time
from datetime import datetime

from wind_forecast import storage
from wind_forecast.ingest.fetch_forecast import fetch_site, load_mapping


# Sites legitimately expected to be absent from a complete run. Empty today:
# every site in mapping.csv must produce a weather snapshot. A future
# legitimate exclusion (e.g. a decommissioned farm) is a deliberate one-line
# edit here, never a silently loosened comparison.
EXPECTED_EXCLUSIONS: set[str] = set()

# Seconds to wait between site fetches, pacing requests under Open-Meteo's
# rate ceiling. Retry (in fetch_forecast._SESSION) handles drops that still
# occur; this reduces how often they happen.
INTER_SITE_DELAY_S = 0.5


def main():
    parser = argparse.ArgumentParser(
        description="Fetch weather forecasts for all sites."
    )
    parser.add_argument(
        "--mapping-csv",
        default=storage.data_path("mapping.csv"),
        help="Path to mapping.csv (local or gs://). Defaults to <DATA_ROOT>/mapping.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=storage.data_path("predictions", "weather"),
        help="Output directory (local or gs://). Defaults to <DATA_ROOT>/predictions/weather",
    )
    parser.add_argument(
        "--run-timestamp",
        default=None,
        help="YYYYMMDD_HHMM timestamp shared by all sites in this run. "
        "Defaults to the time this script starts. Pass from the orchestrator "
        "to pair predict outputs back to this fetch.",
    )
    args = parser.parse_args()

    mapping_path = args.mapping_csv
    if not storage.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    # Generate one timestamp at the start so all sites in this run share it.
    # This is what lets predict scripts later say "use the weather batch
    # tagged 20260528_1100" and get a consistent set across sites.
    run_timestamp = args.run_timestamp or datetime.now().strftime("%Y%m%d_%H%M")
    print(f"Run timestamp: {run_timestamp}\n")

    # --- Load site list ---
    print(f"Loading sites from {mapping_path}...")
    generators = load_mapping(mapping_path)
    sites = sorted(generators.keys())
    print(f"  Found {len(sites)} sites\n")

    # --- Fetch forecasts for each site (in-process) ---
    results = []
    total_start = time.time()

    for i, name in enumerate(sites, 1):
        print(f"[{i}/{len(sites)}] {name}")
        t0 = time.time()
        try:
            fetch_site(name, generators, args.output_dir, run_timestamp=run_timestamp)
            rc = 0
            status = "OK"
        except Exception as e:
            rc = 1
            status = f"FAILED ({e})"
        elapsed = time.time() - t0

        results.append((name, rc, elapsed))
        print(f"  {status} ({elapsed:.1f}s)\n")

        if i < len(sites):
            time.sleep(INTER_SITE_DELAY_S)

    # --- Summary ---
    total_elapsed = time.time() - total_start
    succeeded_sites = {name for name, rc, _ in results if rc == 0}
    failed = [r for r in results if r[1] != 0]

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Run timestamp: {run_timestamp}")
    print(f"  Total sites:   {len(results)}")
    print(f"  Succeeded:     {len(succeeded_sites)}")
    print(f"  Failed:        {len(failed)}")
    print(f"  Total time:    {total_elapsed:.1f}s")

    if failed:
        print(f"\nFailed sites:")
        for name, rc, elapsed in failed:
            print(f"  {name} - {rc}")

    # --- Roster-completeness gate (fail closed) ---
    # Expected = every site in mapping minus any explicit, named exclusion.
    # If any expected site did not produce a snapshot, the Ontario aggregate
    # downstream would be silently incomplete, so we raise. Recovery is the
    # caller's job: the Prefect task carries retries=3, so a transient
    # Open-Meteo failure re-runs the whole fetch and typically self-heals.
    expected = set(sites) - EXPECTED_EXCLUSIONS
    missing = expected - succeeded_sites
    if missing:
        raise RuntimeError(
            f"Incomplete weather batch: {len(missing)} of {len(expected)} "
            f"expected sites missing snapshots: {sorted(missing)}"
        )

    print(f"\nRoster complete: {len(succeeded_sites)}/{len(expected)} expected sites.")


if __name__ == "__main__":
    main()
"""
Run fine-tuning for all wind farm sites.

Reads the site mapping from features.csv, then calls finetune_cf.py
as a subprocess for each site. Continues through all sites on failure
and reports a summary at the end.

Usage:
    python finetune_all.py
    python finetune_all.py --features-csv ../../data/processed/features.csv
    python finetune_all.py --lr 0.0001 --max-epochs 50
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


def load_site_mapping(features_csv: Path) -> list[tuple[int, str]]:
    """Extract unique (site_id, generator_name) pairs from features.csv.

    Returns:
        Sorted list of (site_id, site_name) tuples.
    """
    df = pd.read_csv(features_csv, usecols=["generator_name", "site_id"])
    mapping = df.drop_duplicates().sort_values("site_id")
    return list(zip(mapping["site_id"], mapping["generator_name"]))


def main():
    parser = argparse.ArgumentParser(
        description="Run fine-tuning for all wind farm sites."
    )
    parser.add_argument(
        "--features-csv",
        default="../../data/processed/features.csv",
        help="Path to features.csv for site mapping",
    )
    parser.add_argument(
        "--finetune-script",
        default="finetune_cf.py",
        help="Path to finetune_cf.py (default: finetune_cf.py in same directory)",
    )
    # Pass-through arguments for finetune_cf.py
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--pretrained-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    args = parser.parse_args()

    features_csv = Path(args.features_csv)
    if not features_csv.exists():
        print(f"Features file not found: {features_csv}")
        sys.exit(1)

    finetune_script = Path(args.finetune_script)
    if not finetune_script.exists():
        print(f"Fine-tune script not found: {finetune_script}")
        sys.exit(1)

    # --- Load site mapping ---
    print(f"Loading site mapping from {features_csv}...")
    sites = load_site_mapping(features_csv)
    print(f"  Found {len(sites)} sites\n")

    # --- Build common arguments ---
    common_args = []
    if args.data_dir:
        common_args += ["--data-dir", args.data_dir]
    if args.pretrained_dir:
        common_args += ["--pretrained-dir", args.pretrained_dir]
    if args.batch_size is not None:
        common_args += ["--batch-size", str(args.batch_size)]
    if args.lr is not None:
        common_args += ["--lr", str(args.lr)]
    if args.max_epochs is not None:
        common_args += ["--max-epochs", str(args.max_epochs)]
    if args.patience is not None:
        common_args += ["--patience", str(args.patience)]
    if args.stride is not None:
        common_args += ["--stride", str(args.stride)]

    # --- Run fine-tuning for each site ---
    results = []
    total_start = time.time()

    for i, (site_id, site_name) in enumerate(sites, 1):
        print("=" * 60)
        print(f"[{i}/{len(sites)}] Fine-tuning site_id={site_id} ({site_name})")
        print("=" * 60)

        cmd = [
            sys.executable, str(finetune_script),
            "--site-id", str(site_id),
            "--site-name", site_name,
        ] + common_args

        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0

        status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        results.append((site_id, site_name, result.returncode, elapsed))

        print(f"\n  {status} in {elapsed:.1f}s\n")

    # --- Summary ---
    total_elapsed = time.time() - total_start
    succeeded = sum(1 for _, _, rc, _ in results if rc == 0)
    failed = [r for r in results if r[2] != 0]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total sites: {len(results)}")
    print(f"  Succeeded:   {succeeded}")
    print(f"  Failed:      {len(failed)}")
    print(f"  Total time:  {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

    if failed:
        print(f"\nFailed sites:")
        for site_id, site_name, rc, elapsed in failed:
            print(f"  site_id={site_id} ({site_name}) - exit code {rc}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

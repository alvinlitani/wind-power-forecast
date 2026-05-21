"""
Preprocess raw IESO Generator Output and Capability CSVs.

Reads from data/raw/ieso/{year}/, writes cleaned files to
data/processed/ieso/{year}/. Preserves the same directory structure.

Preprocessing steps:
1. Strip leading comment lines (lines starting with '\\')
2. Filter to WIND fuel type rows only
3. Handle trailing commas (None-keyed fields from DictReader)

Usage:
    python preprocess_ieso.py --input data/raw/ieso --output data/processed/ieso
"""

import argparse
import csv
import sys
from pathlib import Path


def strip_leading_comments(lines: list[str]) -> list[str]:
    """Remove comment lines from the start of the file.

    IESO CSVs sometimes have up to 3 leading lines starting with '\\'.
    """
    cleaned = []
    for i, line in enumerate(lines):
        if i < 3 and line.startswith("\\"):
            continue
        cleaned.append(line)
    return cleaned


def filter_wind_rows(lines: list[str]) -> tuple[list[str], list[dict]]:
    """Parse CSV lines, filter to WIND fuel type, clean trailing-comma artifacts.

    Returns (fieldnames, rows) where rows are dicts with None keys removed.
    """
    import io

    reader = csv.DictReader(io.StringIO("".join(lines)))
    fieldnames = reader.fieldnames

    rows = [row for row in reader if row.get("Fuel Type") == "WIND"]

    # Drop None key created by trailing commas in some files
    rows = [{k: v for k, v in row.items() if k is not None} for row in rows]

    return fieldnames, rows


def preprocess_file(src: Path, dest: Path) -> int:
    """Preprocess a single IESO CSV file.

    Returns the number of WIND rows written.
    """
    with open(src, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines = strip_leading_comments(lines)
    fieldnames, rows = filter_wind_rows(lines)

    if not rows:
        return 0

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw IESO CSVs: strip comments, filter to WIND."
    )
    parser.add_argument(
        "--input",
        default="../../data/raw/ieso",
        help="Raw data directory (default: data/raw/ieso)",
    )
    parser.add_argument(
        "--output",
        default="../../data/processed/ieso",
        help="Processed output directory (default: data/processed/ieso)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    csv_files = sorted(input_dir.rglob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        sys.exit(1)

    print(f"Processing {len(csv_files)} files from {input_dir}/ -> {output_dir}/")

    total_rows = 0
    for src in csv_files:
        # Preserve year subdirectory structure
        rel = src.relative_to(input_dir)
        dest = output_dir / rel

        n_rows = preprocess_file(src, dest)
        total_rows += n_rows
        print(f"  {rel}: {n_rows} WIND rows")

    print(f"\nDone: {total_rows} total WIND rows across {len(csv_files)} files")


if __name__ == "__main__":
    main()
"""
Preprocess raw IESO Generator Output and Capability CSVs.

Reads from <DATA_ROOT>/raw/ieso/, writes cleaned files to
<DATA_ROOT>/processed/ieso/ with the same filenames.

Preprocessing steps:
1. Strip leading comment lines (lines starting with '\\')
2. Filter to WIND fuel type rows only
3. Handle trailing commas (None-keyed fields from DictReader)

Usage:
    python -m wind_forecast.ingest.preprocess_ieso
    python -m wind_forecast.ingest.preprocess_ieso --input gs://bucket/raw/ieso --output gs://bucket/processed/ieso
"""

import argparse
import csv
import io
import sys

from wind_forecast import storage


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
    reader = csv.DictReader(io.StringIO("".join(lines)))
    fieldnames = reader.fieldnames

    rows = [row for row in reader if row.get("Fuel Type") == "WIND"]

    # Drop None key created by trailing commas in some files
    rows = [{k: v for k, v in row.items() if k is not None} for row in rows]

    return fieldnames, rows


def preprocess_file(src: str, dest: str) -> int:
    """Preprocess a single IESO CSV file.

    src and dest may be local paths or gs:// URIs.

    Returns the number of WIND rows written.
    """
    lines = storage.read_text(src).splitlines(keepends=True)

    lines = strip_leading_comments(lines)
    fieldnames, rows = filter_wind_rows(lines)

    if not rows:
        return 0

    # Build the CSV in memory, then write through the storage helper so the
    # same code path works for local disk and GCS.
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)

    storage.write_bytes(buf.getvalue().encode("utf-8"), dest)

    return len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw IESO CSVs: strip comments, filter to WIND."
    )
    parser.add_argument(
        "--input",
        default=storage.data_path("raw", "ieso"),
        help="Raw data directory (local or gs://). Defaults to <DATA_ROOT>/raw/ieso",
    )
    parser.add_argument(
        "--output",
        default=storage.data_path("processed", "ieso"),
        help="Processed output directory (local or gs://). Defaults to <DATA_ROOT>/processed/ieso",
    )
    args = parser.parse_args()

    input_dir = args.input.rstrip("/")
    output_dir = args.output.rstrip("/")

    csv_files = sorted(storage.glob(f"{input_dir}/*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        sys.exit(1)

    print(f"Processing {len(csv_files)} files from {input_dir}/ -> {output_dir}/")

    total_rows = 0
    for src in csv_files:
        filename = src.rstrip("/").split("/")[-1]
        dest = f"{output_dir}/{filename}"

        n_rows = preprocess_file(src, dest)
        total_rows += n_rows
        print(f"  {filename}: {n_rows} WIND rows")

    print(f"\nDone: {total_rows} total WIND rows across {len(csv_files)} files")


if __name__ == "__main__":
    main()
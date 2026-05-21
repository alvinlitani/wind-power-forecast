"""
Evaluate IESO's own wind forecast against actual output as a baseline.

Loads preprocessed IESO CSVs, extracts the Forecast and Output measures
for WIND generators, and computes per-site and aggregate MAE. This
serves as a baseline to compare against the LSTM model's predictions.

Reads from:
    - data/processed/ieso/{year}/*.csv (preprocessed IESO generation data)
    - data/mapping.csv (for nameplate capacity)

Writes to:
    - data/evaluations/ieso_baseline_{period}.csv (per-site detail)
    - stdout (summary)

Usage:
    python evaluate_baseline.py
    python evaluate_baseline.py --start-date 2026-05-01 --end-date 2026-05-14
    python evaluate_baseline.py --ieso-dir ../../data/processed/ieso
"""

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd


def load_ieso_forecast_and_output(
    ieso_dir: Path, start_date: str | None, end_date: str | None
) -> pd.DataFrame:
    """Load IESO Forecast and Output for WIND generators.

    Returns DataFrame with columns:
        [datetime, generator, generator_id, forecast_mwh, output_mwh]
    """
    all_frames = []
    csv_files = sorted(ieso_dir.rglob("*.csv"))

    if not csv_files:
        print(f"No IESO CSV files found in {ieso_dir}")
        sys.exit(1)

    for f in csv_files:
        df = pd.read_csv(f)
        df = df[df["Fuel Type"] == "WIND"].copy()
        df = df[df["Measurement"].isin(["Forecast", "Output"])].copy()

        hour_cols = [c for c in df.columns if c.startswith("Hour ")]
        id_cols = [c for c in df.columns if not c.startswith("Hour ")]

        melted = df.melt(
            id_vars=id_cols,
            value_vars=hour_cols,
            var_name="hour_col",
            value_name="value",
        )

        melted["hour"] = melted["hour_col"].str.extract(r"(\d+)").astype(int) - 1
        melted["datetime"] = pd.to_datetime(
            melted["Delivery Date"]
        ) + pd.to_timedelta(melted["hour"], unit="h")
        melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
        melted = melted[["datetime", "Generator", "Measurement", "value"]]
        melted.columns = ["datetime", "generator", "measure_type", "value"]

        all_frames.append(melted)

    combined = pd.concat(all_frames, ignore_index=True)

    # Pivot so Forecast and Output are separate columns
    pivoted = combined.pivot_table(
        index=["datetime", "generator"],
        columns="measure_type",
        values="value",
        aggfunc="first",
    ).reset_index()
    pivoted.columns.name = None
    pivoted = pivoted.rename(columns={
        "Forecast": "forecast_mwh",
        "Output": "output_mwh",
    })

    # Add generator_id
    pivoted["generator_id"] = pivoted["generator"].str.replace(" ", "", regex=False)

    # Filter to date range if specified
    if start_date:
        pivoted = pivoted[pivoted["datetime"] >= start_date]
    if end_date:
        pivoted = pivoted[pivoted["datetime"] <= end_date + " 23:00:00"]

    # Drop rows where either forecast or output is NaN
    pivoted = pivoted.dropna(subset=["forecast_mwh", "output_mwh"])

    return pivoted


def load_capacity_map(mapping_path: Path) -> dict[str, float]:
    """Load nameplate capacity per generator_id from mapping.csv."""
    capacity = {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gen_id = row["IESO name"].replace(" ", "")
            capacity[gen_id] = float(row["Nameplate Capacity"])
    return capacity


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate IESO forecast vs actual output as baseline."
    )
    parser.add_argument(
        "--ieso-dir", default="../../data/processed/ieso",
        help="Preprocessed IESO data directory",
    )
    parser.add_argument(
        "--mapping-csv", default="../../data/mapping.csv",
        help="Generator mapping CSV",
    )
    parser.add_argument(
        "--output-dir", default="../../data/evaluations",
        help="Output directory for evaluation CSVs",
    )
    parser.add_argument(
        "--start-date", default=None,
        help="Start date YYYY-MM-DD (default: all available)",
    )
    parser.add_argument(
        "--end-date", default=None,
        help="End date YYYY-MM-DD (default: all available)",
    )
    args = parser.parse_args()

    ieso_dir = Path(args.ieso_dir)
    mapping_path = Path(args.mapping_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print("Loading IESO forecast and output data...")
    df = load_ieso_forecast_and_output(ieso_dir, args.start_date, args.end_date)

    n_sites = df["generator_id"].nunique()
    date_min = df["datetime"].min()
    date_max = df["datetime"].max()
    print(f"  {n_sites} generators, {len(df)} rows")
    print(f"  Date range: {date_min} to {date_max}")

    # --- Load capacity ---
    capacity_map = load_capacity_map(mapping_path)
    df["capacity_mw"] = df["generator_id"].map(capacity_map)

    # --- Compute errors ---
    print("Computing errors...")
    df["forecast_cf"] = df["forecast_mwh"] / df["capacity_mw"]
    df["actual_cf"] = df["output_mwh"] / df["capacity_mw"]
    df["error_mwh"] = df["forecast_mwh"] - df["output_mwh"]
    df["abs_error_mwh"] = df["error_mwh"].abs()
    df["error_cf"] = df["forecast_cf"] - df["actual_cf"]
    df["abs_error_cf"] = df["error_cf"].abs()

    # --- Per-site metrics ---
    site_metrics = df.groupby("generator_id").agg(
        mae_cf=("abs_error_cf", "mean"),
        mae_mwh=("abs_error_mwh", "mean"),
        mean_error_cf=("error_cf", "mean"),
        mean_error_mwh=("error_mwh", "mean"),
        capacity_mw=("capacity_mw", "first"),
        n_hours=("abs_error_cf", "count"),
    ).reset_index()

    site_metrics["mae_pct"] = site_metrics["mae_cf"] * 100
    site_metrics = site_metrics.sort_values("generator_id")

    # --- Aggregate metrics ---
    agg_mae_cf = df["abs_error_cf"].mean()
    agg_mae_mwh = df["abs_error_mwh"].mean()
    agg_mean_error_cf = df["error_cf"].mean()

    # --- Save detailed CSV ---
    period = f"{date_min.strftime('%Y%m%d')}_{date_max.strftime('%Y%m%d')}"
    detail_path = output_dir / f"ieso_baseline_{period}.csv"
    detail_df = df[[
        "datetime", "generator_id", "forecast_mwh", "output_mwh",
        "error_mwh", "abs_error_mwh", "forecast_cf", "actual_cf",
        "error_cf", "abs_error_cf",
    ]].sort_values(["generator_id", "datetime"])
    detail_df.to_csv(detail_path, index=False)
    print(f"\nDetailed results saved to {detail_path}")

    # Save per-site summary
    summary_path = output_dir / f"ieso_baseline_{period}_summary.csv"
    site_metrics.to_csv(summary_path, index=False)
    print(f"Per-site summary saved to {summary_path}")

    # --- Print summary ---
    print(f"\n{'='*65}")
    print(f"IESO FORECAST BASELINE")
    print(f"{'='*65}")
    print(f"  Period:            {date_min} to {date_max}")
    print(f"  Sites:             {n_sites}")
    print(f"  Total hours:       {len(df)}")
    print(f"")
    print(f"  Aggregate MAE (CF):  {agg_mae_cf:.4f} ({agg_mae_cf*100:.2f}%)")
    print(f"  Aggregate MAE (MWh): {agg_mae_mwh:.2f}")
    print(f"  Mean bias (CF):      {agg_mean_error_cf:+.4f} ({agg_mean_error_cf*100:+.2f}%)")
    print(f"")
    print(f"{'Generator':<28} {'MAE (MWh)':>10} {'Capacity':>10} {'MAE %':>8}")
    print(f"{'-'*65}")
    for _, row in site_metrics.iterrows():
        print(
            f"{row['generator_id']:<28} {row['mae_mwh']:>10.2f} "
            f"{row['capacity_mw']:>10.1f} {row['mae_pct']:>7.1f}%"
        )
    print(f"{'-'*65}")
    print(
        f"{'AGGREGATE':<28} {agg_mae_mwh:>10.2f} "
        f"{'':>10} {agg_mae_cf*100:>7.1f}%"
    )


if __name__ == "__main__":
    main()

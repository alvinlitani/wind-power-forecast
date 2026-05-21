"""
Evaluate predictions against IESO actuals.

Takes a prediction CSV (output of predict.py), loads the matching
IESO actuals for the predicted datetime range, and computes per-site
and aggregate error metrics.

Reads from:
    - A prediction CSV (e.g., data/predictions/20260514_1318.csv)
    - data/processed/ieso/{year}/*.csv (preprocessed IESO generation data)
    - data/mapping.csv (for nameplate capacity, used to convert CF errors to MWh)

Writes to:
    - data/evaluations/{prediction_filename}_eval.csv (per-site detail)
    - stdout (summary)

Usage:
    python evaluate_daily.py --prediction data/predictions/20260514_1318.csv
    python evaluate_daily.py --prediction data/predictions/20260514_1318.csv --ieso-dir data/processed/ieso
"""

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd


def load_prediction(path: Path) -> pd.DataFrame:
    """Load a prediction CSV.

    Expected columns: datetime, generator_id, predicted_cf, predicted_mwh,
                      encoder_end, staleness_hours
    """
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def load_ieso_actuals(ieso_dir: Path, date_range: tuple[str, str]) -> pd.DataFrame:
    """Load IESO preprocessed CSVs covering the prediction date range.

    Returns DataFrame with columns:
        [datetime, generator, output_mwh, available_capacity_mw]
    """
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])

    # Determine which monthly files could contain data in range
    months_needed = set()
    current = start_date.replace(day=1)
    while current <= end_date:
        months_needed.add(current.strftime("%Y%m"))
        current += pd.DateOffset(months=1)

    all_frames = []
    for yyyymm in sorted(months_needed):
        year = yyyymm[:4]
        csv_files = sorted((ieso_dir / year).glob(f"*{yyyymm}*.csv"))
        for f in csv_files:
            df = pd.read_csv(f)
            df = df[df["Fuel Type"] == "WIND"].copy()

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

    if not all_frames:
        return pd.DataFrame(columns=["datetime", "generator", "output_mwh"])

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined[combined["measure_type"] == "Output"]
    pivoted = combined.pivot_table(
        index=["datetime", "generator"],
        columns="measure_type",
        values="value",
        aggfunc="first",
    ).reset_index()
    pivoted.columns.name = None
    pivoted = pivoted.rename(columns={"Output": "output_mwh"})

    # Add generator_id (no spaces) for joining with predictions
    pivoted["generator_id"] = pivoted["generator"].str.replace(" ", "", regex=False)

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
        description="Evaluate predictions against IESO actuals."
    )
    parser.add_argument(
        "--prediction", required=True,
        help="Path to prediction CSV (output of predict.py)",
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
    args = parser.parse_args()

    pred_path = Path(args.prediction)
    if not pred_path.exists():
        print(f"Prediction file not found: {pred_path}")
        sys.exit(1)

    ieso_dir = Path(args.ieso_dir)
    mapping_path = Path(args.mapping_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load predictions ---
    print(f"Loading predictions from {pred_path.name}...")
    pred_df = load_prediction(pred_path)

    n_sites = pred_df["generator_id"].nunique()
    pred_start = pred_df["datetime"].min()
    pred_end = pred_df["datetime"].max()
    staleness = pred_df["staleness_hours"].iloc[0]
    encoder_end = pred_df["encoder_end"].iloc[0]

    print(f"  {n_sites} sites, {len(pred_df)} rows")
    print(f"  Prediction window: {pred_start} to {pred_end}")
    print(f"  Encoder end: {encoder_end}, staleness: {staleness}h")

    # --- Load IESO actuals ---
    print("Loading IESO actuals...")
    date_range = (pred_start.strftime("%Y-%m-%d"), pred_end.strftime("%Y-%m-%d"))
    actual_df = load_ieso_actuals(ieso_dir, date_range)

    if actual_df.empty:
        print("No IESO actuals found for the prediction window.")
        print("Actuals may not be published yet (IESO publishes at ~6am next day).")
        sys.exit(1)

    # Filter to prediction window
    actual_df = actual_df[
        (actual_df["datetime"] >= pred_start)
        & (actual_df["datetime"] <= pred_end)
    ]
    print(f"  {actual_df['generator_id'].nunique()} generators, {len(actual_df)} rows in window")

    # --- Load capacity map ---
    capacity_map = load_capacity_map(mapping_path)

    # --- Join predictions with actuals ---
    print("Computing errors...")
    merged = pd.merge(
        pred_df[["datetime", "generator_id", "predicted_cf", "predicted_mwh"]],
        actual_df[["datetime", "generator_id", "output_mwh"]],
        on=["datetime", "generator_id"],
        how="inner",
    )

    matched_sites = merged["generator_id"].nunique()
    matched_hours = merged.groupby("generator_id").size().min() if len(merged) > 0 else 0
    print(f"  Matched: {matched_sites} sites, {len(merged)} rows")

    if merged.empty:
        print("No matching rows between predictions and actuals.")
        sys.exit(1)

    # Compute actual capacity factor
    merged["capacity_mw"] = merged["generator_id"].map(capacity_map)
    merged["actual_cf"] = merged["output_mwh"] / merged["capacity_mw"]

    # Compute errors
    merged["error_cf"] = merged["predicted_cf"] - merged["actual_cf"]
    merged["abs_error_cf"] = merged["error_cf"].abs()
    merged["error_mwh"] = merged["predicted_mwh"] - merged["output_mwh"]
    merged["abs_error_mwh"] = merged["error_mwh"].abs()

    # --- Per-site metrics ---
    site_metrics = merged.groupby("generator_id").agg(
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
    agg_mae_cf = merged["abs_error_cf"].mean()
    agg_mae_mwh = merged["abs_error_mwh"].mean()
    agg_mean_error_cf = merged["error_cf"].mean()

    # --- Save detailed CSV ---
    output_filename = f"{pred_path.stem}_eval.csv"
    output_path = output_dir / output_filename

    # Detailed per-hour results
    detail_df = merged[[
        "datetime", "generator_id", "predicted_cf", "actual_cf",
        "error_cf", "abs_error_cf", "predicted_mwh", "output_mwh",
        "error_mwh", "abs_error_mwh",
    ]].sort_values(["generator_id", "datetime"])
    detail_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to {output_path}")

    # --- Save per-site summary ---
    summary_path = output_dir / f"{pred_path.stem}_summary.csv"
    site_metrics.to_csv(summary_path, index=False)
    print(f"Per-site summary saved to {summary_path}")

    # --- Print summary ---
    print(f"\n{'='*65}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*65}")
    print(f"  Prediction file:   {pred_path.name}")
    print(f"  Prediction window: {pred_start} to {pred_end}")
    print(f"  Encoder staleness: {staleness}h")
    print(f"  Sites evaluated:   {matched_sites}")
    print(f"  Hours matched:     {matched_hours} per site")
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

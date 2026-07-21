"""
Evaluate IESO's own wind forecast against actual output as a baseline.

Loads preprocessed IESO CSVs, extracts the Forecast and Output measures
for WIND generators, and computes per-site and aggregate MAE. Serves as
a reference baseline that the LSTM and XGBoost models are expected to
beat.

Two entry points:
    - main()                 — CLI tool for ad-hoc multi-day analysis
                              (e.g., "compute baseline MAE over April")
    - baseline_for_window()  — function the daily flow calls to get the
                              IESO baseline MAE over the same window a
                              model just predicted, for direct comparison

Reads from:
    - <DATA_ROOT>/processed/ieso/*.csv     (preprocessed IESO data)
    - <DATA_ROOT>/mapping.csv              (nameplate capacity per site)

Writes (CLI only):
    - <DATA_ROOT>/evaluations/baseline/ieso_baseline_{start}_{end}.csv
    - <DATA_ROOT>/evaluations/baseline/ieso_baseline_{start}_{end}_summary.csv

Usage:
    python -m wind_forecast.evaluate.evaluate_baseline
    python -m wind_forecast.evaluate.evaluate_baseline --start-date 2026-05-01 --end-date 2026-05-14
"""

import argparse
import csv

import pandas as pd

from wind_forecast import storage


def load_ieso_forecast_and_output(
    ieso_dir: str, start_date: str | None, end_date: str | None
) -> pd.DataFrame:
    """Load IESO Forecast and Output for WIND generators.

    Files live in a flat layout under ieso_dir (per the preprocess_ieso
    output). Date filtering happens after melting, so any extra months
    pulled in are harmless.

    Returns DataFrame with columns:
        [datetime, generator, generator_id, forecast_mwh, output_mwh]
    """
    all_frames = []
    csv_files = sorted(storage.glob(f"{ieso_dir.rstrip('/')}/*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No IESO CSV files found in {ieso_dir}")

    for f in csv_files:
        df = storage.read_csv(f)
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

    pivoted["generator_id"] = pivoted["generator"].str.replace(" ", "", regex=False)

    if start_date:
        pivoted = pivoted[pivoted["datetime"] >= start_date]
    if end_date:
        pivoted = pivoted[pivoted["datetime"] <= end_date + " 23:00:00"]

    pivoted = pivoted.dropna(subset=["forecast_mwh", "output_mwh"])

    return pivoted


def load_capacity_map(mapping_path: str) -> dict[str, float]:
    """Load nameplate capacity per generator_id from mapping.csv."""
    capacity = {}
    with storage.open_file(mapping_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gen_id = row["IESO name"].replace(" ", "")
            capacity[gen_id] = float(row["Nameplate Capacity"])
    return capacity


def _compute_metrics(df: pd.DataFrame, capacity_map: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Compute error columns, per-site metrics, and aggregate metrics.

    Pure data transform — no I/O. Both main() and baseline_for_window() use it.

    Args:
        df: DataFrame with [datetime, generator_id, forecast_mwh, output_mwh].
        capacity_map: generator_id -> nameplate capacity.

    Returns:
        (detail_df, site_metrics, aggregate_dict)
    """
    df = df.copy()
    df["capacity_mw"] = df["generator_id"].map(capacity_map)
    df["forecast_cf"] = df["forecast_mwh"] / df["capacity_mw"]
    df["actual_cf"] = df["output_mwh"] / df["capacity_mw"]
    df["error_mwh"] = df["forecast_mwh"] - df["output_mwh"]
    df["abs_error_mwh"] = df["error_mwh"].abs()
    df["error_cf"] = df["forecast_cf"] - df["actual_cf"]
    df["abs_error_cf"] = df["error_cf"].abs()

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

    aggregate = {
        "mae_cf": float(df["abs_error_cf"].mean()),
        "mae_mwh": float(df["abs_error_mwh"].mean()),
        "mae_pct": float(df["abs_error_cf"].mean() * 100),
        "bias_cf": float(df["error_cf"].mean()),
        "n_sites": int(df["generator_id"].nunique()),
        "n_rows": int(len(df)),
    }
    return df, site_metrics, aggregate


def baseline_for_window(
    ieso_dir: str,
    mapping_path: str,
    pred_start: pd.Timestamp,
    pred_end: pd.Timestamp,
) -> dict:
    """Compute IESO baseline metrics for a specific prediction window.

    Lets the daily flow compute a baseline directly comparable to a model
    prediction: the IESO forecast for exactly the hours the model predicted.
    Returned metrics share the same shape as evaluate_daily's so dashboards
    can plot LSTM/XGBoost/IESO baseline on one chart.

    Returns a metrics dict (or None if no overlapping data exists), shaped:
        {model, run_timestamp, n_sites, n_rows, mae_cf, mae_mwh, mae_pct,
         bias_cf, prediction_window_start, prediction_window_end,
         staleness_hours: None}
    """
    start_str = pred_start.strftime("%Y-%m-%d")
    end_str = pred_end.strftime("%Y-%m-%d")
    df = load_ieso_forecast_and_output(ieso_dir, start_str, end_str)
    if df.empty:
        return None

    # Tight bounds: load_ieso_forecast_and_output filters by date strings
    # (whole-day granularity), so narrow further to the exact window.
    df = df[(df["datetime"] >= pred_start) & (df["datetime"] <= pred_end)]
    if df.empty:
        return None

    capacity_map = load_capacity_map(mapping_path)
    _, _, aggregate = _compute_metrics(df, capacity_map)

    return {
        "model": "ieso_baseline",
        "run_timestamp": pred_start.strftime("%Y%m%d_%H%M"),
        "n_sites": aggregate["n_sites"],
        "n_rows": aggregate["n_rows"],
        "mae_cf": aggregate["mae_cf"],
        "mae_mwh": aggregate["mae_mwh"],
        "mae_pct": aggregate["mae_pct"],
        "bias_cf": aggregate["bias_cf"],
        "prediction_window_start": str(pred_start),
        "prediction_window_end": str(pred_end),
        "staleness_hours": None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate IESO forecast vs actual output as baseline."
    )
    parser.add_argument(
        "--ieso-dir", default=storage.data_path("processed", "ieso"),
        help="Preprocessed IESO data directory (local or gs://). "
        "Defaults to <DATA_ROOT>/processed/ieso",
    )
    parser.add_argument(
        "--mapping-csv", default=storage.data_path("mapping.csv"),
        help="Generator mapping CSV (local or gs://). Defaults to <DATA_ROOT>/mapping.csv",
    )
    parser.add_argument(
        "--output-dir", default=storage.data_path("evaluations", "baseline"),
        help="Output directory for evaluation CSVs (local or gs://). "
        "Defaults to <DATA_ROOT>/evaluations/baseline",
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

    # --- Load data ---
    print("Loading IESO forecast and output data...")
    df = load_ieso_forecast_and_output(args.ieso_dir, args.start_date, args.end_date)

    n_sites = df["generator_id"].nunique()
    date_min = df["datetime"].min()
    date_max = df["datetime"].max()
    print(f"  {n_sites} generators, {len(df)} rows")
    print(f"  Date range: {date_min} to {date_max}")

    # --- Compute errors ---
    print("Computing errors...")
    capacity_map = load_capacity_map(args.mapping_csv)
    detail_df, site_metrics, aggregate = _compute_metrics(df, capacity_map)

    # --- Save outputs ---
    period = f"{date_min.strftime('%Y%m%d')}_{date_max.strftime('%Y%m%d')}"
    out_root = args.output_dir.rstrip("/")
    detail_path = f"{out_root}/ieso_baseline_{period}.csv"
    summary_path = f"{out_root}/ieso_baseline_{period}_summary.csv"

    detail_out = detail_df[[
        "datetime", "generator_id", "forecast_mwh", "output_mwh",
        "error_mwh", "abs_error_mwh", "forecast_cf", "actual_cf",
        "error_cf", "abs_error_cf",
    ]].sort_values(["generator_id", "datetime"])
    storage.write_csv(detail_out, detail_path)
    print(f"\nDetailed results saved to {detail_path}")

    storage.write_csv(site_metrics, summary_path)
    print(f"Per-site summary saved to {summary_path}")

    # --- Print summary ---
    print(f"\n{'='*65}")
    print(f"IESO FORECAST BASELINE")
    print(f"{'='*65}")
    print(f"  Period:            {date_min} to {date_max}")
    print(f"  Sites:             {n_sites}")
    print(f"  Total hours:       {len(df)}")
    print(f"")
    print(f"  Aggregate MAE (CF):  {aggregate['mae_cf']:.4f} ({aggregate['mae_pct']:.2f}%)")
    print(f"  Aggregate MAE (MWh): {aggregate['mae_mwh']:.2f}")
    print(f"  Mean bias (CF):      {aggregate['bias_cf']:+.4f} ({aggregate['bias_cf']*100:+.2f}%)")
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
        f"{'AGGREGATE':<28} {aggregate['mae_mwh']:>10.2f} "
        f"{'':>10} {aggregate['mae_pct']:>7.1f}%"
    )


if __name__ == "__main__":
    main()

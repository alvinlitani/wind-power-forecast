"""
Evaluate a prediction CSV against IESO actuals.

Works for both LSTM and XGBoost prediction outputs — they share the
columns this script needs (datetime, generator_id, predicted_cf,
predicted_mwh) and any model-specific columns (encoder_end /
staleness_hours for LSTM, run_timestamp for XGBoost) are picked up
opportunistically.

Reads from:
    - A prediction CSV (e.g. <DATA_ROOT>/predictions/lstm/20260528_0815.csv
                          or <DATA_ROOT>/predictions/pc/20260528_1400.csv)
    - <DATA_ROOT>/processed/ieso/*.csv     (preprocessed IESO generation data)
    - <DATA_ROOT>/mapping.csv              (nameplate capacity per site)

Writes to:
    - <DATA_ROOT>/evaluations/{model}/{run_timestamp}_eval.csv     (per-hour, per-site)
    - <DATA_ROOT>/evaluations/{model}/{run_timestamp}_summary.csv  (per-site metrics)
    - stdout summary

Usage:
    python -m wind_forecast.evaluate.evaluate_daily \\
        --prediction <DATA_ROOT>/predictions/lstm/20260528_0815.csv
    python -m wind_forecast.evaluate.evaluate_daily \\
        --prediction <DATA_ROOT>/predictions/pc/20260528_1400.csv --model pc
"""

import argparse
import csv

import pandas as pd
import numpy as np

from wind_forecast import storage


def load_prediction(path: str) -> pd.DataFrame:
    """Load a prediction CSV from either model.

    Required columns: datetime, generator_id, predicted_cf, predicted_mwh
    Optional model-specific columns (preserved if present):
        encoder_end, staleness_hours  (LSTM)
        run_timestamp                 (XGBoost)
    """
    df = storage.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def infer_model_from_path(path: str) -> str:
    """Best-effort guess at which model produced the prediction file.

    The convention is <DATA_ROOT>/predictions/{model}/{timestamp}.csv,
    so the parent-directory basename is the model name. Falls back to
    'unknown' if the convention isn't followed.
    """
    parts = path.rstrip("/").split("/")
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


def load_ieso_actuals(ieso_dir: str, date_range: tuple[str, str]) -> pd.DataFrame:
    """Load IESO preprocessed CSVs covering the prediction date range.

    Files live in a flat layout (<DATA_ROOT>/processed/ieso/*.csv) and are
    named with the year-month suffix (PUB_GenOutputCapabilityMonth_YYYYMM.csv),
    so we determine which months overlap the range and load only those.

    Returns DataFrame with columns:
        [datetime, generator, output_mwh, generator_id]
    """
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])

    # Determine which monthly files could contain data in range.
    months_needed = set()
    current = start_date.replace(day=1)
    while current <= end_date:
        months_needed.add(current.strftime("%Y%m"))
        current += pd.DateOffset(months=1)

    all_frames = []
    for yyyymm in sorted(months_needed):
        # Flat layout: filename embeds the year-month.
        pattern = f"{ieso_dir.rstrip('/')}/*{yyyymm}*.csv"
        csv_files = sorted(storage.glob(pattern))
        for f in csv_files:
            df = storage.read_csv(f)
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

    # IESO's Generator column is already the canonical spaced name
    # (e.g. "PORT BURWELL"); use it directly so it matches prediction files,
    # which now also write the spaced form.
    pivoted["generator_id"] = pivoted["generator"]

    return pivoted


def load_capacity_map(mapping_path: str) -> dict[str, float]:
    """Load nameplate capacity per generator_id from mapping.csv."""
    capacity = {}
    with storage.open_file(mapping_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gen_id = row["IESO name"]  # spaced canonical; matches predictions + actuals
            capacity[gen_id] = float(row["Nameplate Capacity"])
    return capacity


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a prediction CSV against IESO actuals."
    )
    parser.add_argument(
        "--prediction", required=True,
        help="Path to prediction CSV (local or gs://).",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name ('lstm', 'pc', ...). Determines the evaluation output "
        "subdirectory. Inferred from the prediction path if not given.",
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
        "--output-dir", default=storage.data_path("evaluations"),
        help="Output directory for evaluation CSVs (local or gs://). "
        "The {model} subdirectory is appended. Defaults to <DATA_ROOT>/evaluations",
    )
    args = parser.parse_args()

    return run_evaluation(
        prediction_path=args.prediction,
        model=args.model,
        ieso_dir=args.ieso_dir,
        mapping_path=args.mapping_csv,
        output_dir=args.output_dir,
    )


def run_evaluation(
    prediction_path: str,
    model: str | None,
    ieso_dir: str,
    mapping_path: str,
    output_dir: str,
) -> dict:
    """Evaluate a single prediction file and return aggregate metrics.

    Split out from main() so the Prefect orchestrator can call this directly
    and capture the returned metrics dict for downstream tasks (W&B logging,
    Grafana push, alerting on aggregate MAE).

    Returns:
        dict with keys: model, run_timestamp, n_sites, n_rows,
        mae_cf, mae_mwh, bias_cf, prediction_window_start/end.
    """
    if not storage.exists(prediction_path):
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")

    # Resolve model name (used to route output, label rows for comparison).
    model_name = model or infer_model_from_path(prediction_path)

    # Derive the run_timestamp from the filename stem.
    filename = prediction_path.rstrip("/").split("/")[-1]
    file_stem = filename.removesuffix(".csv")

    # --- Load predictions ---
    print(f"Loading predictions from {filename} (model={model_name})...")
    pred_df = load_prediction(prediction_path)

    n_sites = pred_df["generator_id"].nunique()
    pred_start = pred_df["datetime"].min()
    pred_end = pred_df["datetime"].max()

    print(f"  {n_sites} sites, {len(pred_df)} rows")
    print(f"  Prediction window: {pred_start} to {pred_end}")

    # LSTM-specific context fields — surface if present, silent if absent.
    staleness = None
    encoder_end = None
    if "staleness_hours" in pred_df.columns:
        staleness = pred_df["staleness_hours"].iloc[0]
        encoder_end = pred_df["encoder_end"].iloc[0]
        print(f"  Encoder end: {encoder_end}, staleness: {staleness}h")

    # --- Load IESO actuals ---
    # Window is widened one day earlier than the prediction start so this same
    # load also supplies the persistence reference (actual output at the same
    # hour on D-1). D-1 is the most recent data available when the 02:00 batch
    # is issued, so the reference is information-fair against the model.
    print("Loading IESO actuals...")
    lag_start = pred_start - pd.Timedelta(days=1)
    date_range = (lag_start.strftime("%Y-%m-%d"), pred_end.strftime("%Y-%m-%d"))
    actual_all = load_ieso_actuals(ieso_dir, date_range)

    if actual_all.empty:
        raise RuntimeError(
            "No IESO actuals found for the prediction window (may not be published yet)."
        )

    actual_df = actual_all[
        (actual_all["datetime"] >= pred_start)
        & (actual_all["datetime"] <= pred_end)
    ]

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
        raise RuntimeError("No matching rows between predictions and actuals.")

    merged["capacity_mw"] = merged["generator_id"].map(capacity_map)
    merged["actual_cf"] = merged["output_mwh"] / merged["capacity_mw"]
    merged["error_cf"] = merged["predicted_cf"] - merged["actual_cf"]
    merged["abs_error_cf"] = merged["error_cf"].abs()
    merged["error_mwh"] = merged["predicted_mwh"] - merged["output_mwh"]
    merged["abs_error_mwh"] = merged["error_mwh"].abs()
    # Tag every row with the model so combined eval files (e.g. Grafana
    # ingest) can split metrics per model without ambiguity.
    merged["model"] = model_name

    # --- Persistence reference (naive baseline) ---
    # Shifting D-1 actuals forward 24h aligns them with the hours predicted.
    # Note: a 24h shift is calendar-naive, so DST transition days will mismatch
    # one hour; harmless in practice, worth knowing if a spring/fall run looks odd.
    lag = actual_all[["datetime", "generator_id", "output_mwh"]].copy()
    lag["datetime"] = lag["datetime"] + pd.Timedelta(hours=24)
    lag = lag.rename(columns={"output_mwh": "persist_mwh"})
    merged = merged.merge(lag, on=["datetime", "generator_id"], how="left")
    merged["persist_cf"] = merged["persist_mwh"] / merged["capacity_mw"]
    merged["persist_error_cf"] = merged["persist_cf"] - merged["actual_cf"]
    merged["persist_abs_error_cf"] = merged["persist_error_cf"].abs()

    # --- Per-site metrics ---
    site_metrics = merged.groupby("generator_id").agg(
        mae_cf=("abs_error_cf", "mean"),
        mae_mwh=("abs_error_mwh", "mean"),
        mean_error_cf=("error_cf", "mean"),
        mean_error_mwh=("error_mwh", "mean"),
        capacity_mw=("capacity_mw", "first"),
        n_hours=("abs_error_cf", "count"),
        rmse_cf=("error_cf", lambda s: float(np.sqrt((s ** 2).mean()))),
    ).reset_index()
    site_metrics["mae_pct"] = site_metrics["mae_cf"] * 100
    site_metrics["model"] = model_name
    site_metrics = site_metrics.sort_values("generator_id")

    # --- Aggregate metrics ---
    # CF is output/nameplate, so CF-space errors ARE the capacity-normalized
    # metrics (nMAE / nRMSE) standard in wind power forecasting; x100 for percent.
    agg_mae_cf = merged["abs_error_cf"].mean()
    agg_mae_mwh = merged["abs_error_mwh"].mean()
    agg_mean_error_cf = merged["error_cf"].mean()
    agg_rmse_cf = float(np.sqrt((merged["error_cf"] ** 2).mean()))
    agg_rmse_mwh = float(np.sqrt((merged["error_mwh"] ** 2).mean()))

    # --- Skill vs persistence ---
    # skill = 1 - (model error / reference error). Both sides must be scored on
    # the SAME rows, so model error is recomputed on the subset where a
    # persistence value exists rather than reusing the full-window mean.
    pers = merged.dropna(subset=["persist_cf"])
    n_rows_persist = int(len(pers))
    if n_rows_persist > 0:
        persist_mae_cf = float(pers["persist_abs_error_cf"].mean())
        persist_rmse_cf = float(np.sqrt((pers["persist_error_cf"] ** 2).mean()))
        model_mae_matched = float(pers["abs_error_cf"].mean())
        model_rmse_matched = float(np.sqrt((pers["error_cf"] ** 2).mean()))
        skill_mae = 1.0 - (model_mae_matched / persist_mae_cf) if persist_mae_cf > 0 else None
        skill_rmse = 1.0 - (model_rmse_matched / persist_rmse_cf) if persist_rmse_cf > 0 else None
    else:
        persist_mae_cf = persist_rmse_cf = None
        model_mae_matched = model_rmse_matched = None
        skill_mae = skill_rmse = None

    # --- Save outputs ---
    # Subdirectory per model so LSTM/XGB eval files don't collide
    # (they share filenames when run_timestamp matches across models).
    model_output_dir = f"{output_dir.rstrip('/')}/{model_name}"

    detail_path = f"{model_output_dir}/{file_stem}_eval.csv"
    summary_path = f"{model_output_dir}/{file_stem}_summary.csv"

    detail_df = merged[[
        "datetime", "generator_id", "model", "predicted_cf", "actual_cf",
        "error_cf", "abs_error_cf", "predicted_mwh", "output_mwh",
        "error_mwh", "abs_error_mwh",
    ]].sort_values(["generator_id", "datetime"])
    storage.write_csv(detail_df, detail_path)
    print(f"\nDetailed results saved to {detail_path}")

    storage.write_csv(site_metrics, summary_path)
    print(f"Per-site summary saved to {summary_path}")

    # --- Print summary ---
    print(f"\n{'='*65}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*65}")
    print(f"  Model:             {model_name}")
    print(f"  Prediction file:   {filename}")
    print(f"  Prediction window: {pred_start} to {pred_end}")
    if staleness is not None:
        print(f"  Encoder staleness: {staleness}h")
    print(f"  Sites evaluated:   {matched_sites}")
    print(f"  Hours matched:     {matched_hours} per site")
    print(f"")
    print(f"  Aggregate MAE (CF):  {agg_mae_cf:.4f} ({agg_mae_cf*100:.2f}%)")
    print(f"  Aggregate MAE (MWh): {agg_mae_mwh:.2f}")
    print(f"  Mean bias (CF):      {agg_mean_error_cf:+.4f} ({agg_mean_error_cf*100:+.2f}%)")
    print(f"  Aggregate RMSE (CF): {agg_rmse_cf:.4f} ({agg_rmse_cf*100:.2f}%)")
    print(f"  Aggregate RMSE (MWh):{agg_rmse_mwh:.2f}")
    print(f"")
    if n_rows_persist > 0:
        print(f"  --- vs persistence (same-hour D-1), {n_rows_persist} matched rows ---")
        print(f"  Persistence nMAE:    {persist_mae_cf*100:.2f}%")
        print(f"  Persistence nRMSE:   {persist_rmse_cf*100:.2f}%")
        print(f"  Model nMAE (matched):{model_mae_matched*100:.2f}%")
        if skill_mae is not None:
            verdict = "better than naive" if skill_mae > 0 else "no better than naive"
            print(f"  Skill score (MAE):   {skill_mae:+.3f}  ({verdict})")
        if skill_rmse is not None:
            print(f"  Skill score (RMSE):  {skill_rmse:+.3f}")
    else:
        print(f"  No persistence reference available (D-1 actuals missing).")
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

    # --- Return aggregate metrics for Prefect / W&B / Grafana ingest ---
    return {
        "model": model_name,
        "run_timestamp": file_stem,
        "n_sites": int(matched_sites),
        "n_rows": int(len(merged)),
        "mae_cf": float(agg_mae_cf),
        "mae_mwh": float(agg_mae_mwh),
        "mae_pct": float(agg_mae_cf * 100),
        "bias_cf": float(agg_mean_error_cf),
        "prediction_window_start": str(pred_start),
        "prediction_window_end": str(pred_end),
        "staleness_hours": float(staleness) if staleness is not None else None,
        "nmae_pct": float(agg_mae_cf * 100),
        "rmse_cf": agg_rmse_cf,
        "rmse_mwh": agg_rmse_mwh,
        "nrmse_pct": float(agg_rmse_cf * 100),
        "persistence_nmae_pct": float(persist_mae_cf * 100) if persist_mae_cf is not None else None,
        "persistence_nrmse_pct": float(persist_rmse_cf * 100) if persist_rmse_cf is not None else None,
        "skill_score_mae": skill_mae,
        "skill_score_rmse": skill_rmse,
        "n_rows_persistence": n_rows_persist,
        "detail_path": detail_path,
        "summary_path": summary_path,
    }


if __name__ == "__main__":
    main()

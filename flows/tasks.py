"""Prefect @task wrappers around the wind_forecast package functions.

Each task is a thin shim: get a logger, call the underlying function, return
its result. All the real work lives in `wind_forecast.*` so the scripts
remain runnable as plain Python (CLI / unit tests) without Prefect involved.

Retry policies are set here per task based on what each one depends on:
    - fetch_actuals_task           : IESO might be late -> 6 retries, 10 min delay
    - fetch_forecast_all_task      : Open-Meteo transient errors -> 3 retries, 5 min delay
    - wait_for_ieso_task           : poll for late publishes -> 12 retries, 10 min delay (2h patience)
    - predict_lstm_task / predict_pc_task : flaky GCS reads -> 1 retry
    - eval_*_task                  : same -> 1 retry
    - preprocess_ieso_task         : pure local compute -> no retry useful
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional
import time

import pandas as pd
from prefect import get_run_logger, task

from wind_forecast import storage
from wind_forecast.evaluate import evaluate_baseline, evaluate_daily
from wind_forecast.ingest import fetch_actuals, fetch_forecast, fetch_forecast_all, preprocess_ieso
from wind_forecast.predict import predict, predict_pc


# ---------- Ingest ----------

@task(name="fetch-ieso-actuals", retries=6, retry_delay_seconds=60)
def fetch_actuals_task(output_dir: Optional[str] = None) -> str:
    logger = get_run_logger()
    output_dir = output_dir or storage.data_path("raw", "ieso")
    logger.info(f"Fetching IESO actuals to {output_dir}")

    result = fetch_actuals.fetch_actuals(output_dir)
    logger.info(
        f"IESO actuals downloaded={result['downloaded']} skipped={result['skipped']}"
    )
    return output_dir


@task(name="preprocess-ieso", retries=0)
def preprocess_ieso_task(
    input_dir: Optional[str] = None, output_dir: Optional[str] = None
) -> str:
    """Strip comments and filter to WIND rows for every CSV in input_dir."""
    logger = get_run_logger()
    input_dir = input_dir or storage.data_path("raw", "ieso")
    output_dir = output_dir or storage.data_path("processed", "ieso")

    csv_files = sorted(storage.glob(f"{input_dir.rstrip('/')}/*.csv"))
    if not csv_files:
        raise RuntimeError(f"No IESO CSV files found in {input_dir}")

    total = 0
    for src in csv_files:
        filename = src.rstrip("/").split("/")[-1]
        dest = f"{output_dir.rstrip('/')}/{filename}"
        n = preprocess_ieso.preprocess_file(src, dest)
        total += n
        logger.info(f"  {filename}: {n} WIND rows")

    logger.info(f"Preprocessed {len(csv_files)} files, {total} WIND rows total")
    return output_dir


# ---------- Weather fetch ----------

@task(name="fetch-weather-all-sites", retries=3, retry_delay_seconds=300)
def fetch_forecast_all_task(
    run_timestamp: str,
    mapping_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> str:
    """Fetch the weather snapshot for every site under one run_timestamp.

    Per-site failures (one bad Open-Meteo response) are caught individually
    inside the loop so one site doesn't abort the others. After the loop a
    roster-completeness gate fails the task (fail closed) if any expected site
    is missing — the task's retries=3 then re-runs the whole fetch, so a
    transient failure self-heals while a persistent one surfaces loudly.

    Returns the run_timestamp so downstream tasks pair against the same batch.
    """
    logger = get_run_logger()
    mapping_path = mapping_path or storage.data_path("mapping.csv")
    output_dir = output_dir or storage.data_path("predictions", "weather")

    if not storage.exists(mapping_path):
        raise RuntimeError(f"Mapping file not found: {mapping_path}")

    generators = fetch_forecast.load_mapping(mapping_path)
    sites = sorted(generators.keys())
    logger.info(f"Fetching weather for {len(sites)} sites (run_timestamp={run_timestamp})")

    succeeded_sites, failed = set(), []
    for i, name in enumerate(sites):
        try:
            fetch_forecast.fetch_site(name, generators, output_dir, run_timestamp=run_timestamp)
            succeeded_sites.add(name)
        except Exception as e:
            failed.append((name, str(e)))
            logger.warning(f"  {name}: FAILED ({e})")
        if i < len(sites) - 1:
            time.sleep(fetch_forecast_all.INTER_SITE_DELAY_S)

    logger.info(f"Weather fetch complete: {len(succeeded_sites)} ok, {len(failed)} failed")

    # Roster-completeness gate (fail closed). Mirrors the CLI behavior in
    # fetch_forecast_all.main(): every expected site must produce a snapshot,
    # else the downstream Ontario aggregate is silently incomplete. The task's
    # retries=3 provides recovery — a transient Open-Meteo failure re-runs the
    # whole fetch and typically self-heals.
    expected = set(sites) - fetch_forecast_all.EXPECTED_EXCLUSIONS
    missing = expected - succeeded_sites
    if missing:
        first_err = failed[0][1] if failed else "no per-site error recorded"
        raise RuntimeError(
            f"Incomplete weather batch: {len(missing)} of {len(expected)} "
            f"expected sites missing: {sorted(missing)}. First error: {first_err}"
        )
    return run_timestamp


# ---------- Predictions ----------

@task(name="predict-lstm", retries=1)
def predict_lstm_task(run_timestamp: str) -> str:
    """Run the LSTM (capacity-factor) prediction for the given weather batch.

    Returns the path of the written prediction CSV.
    """
    logger = get_run_logger()
    logger.info(f"Running LSTM prediction for run_timestamp={run_timestamp}")

    # Invoke the script's main() via argparse-equivalent — pass everything
    # explicitly so the task is fully deterministic from its arguments.
    import sys
    saved_argv = sys.argv
    sys.argv = [
        "predict",
        "--run-timestamp", run_timestamp,
    ]
    try:
        predict.main()
    finally:
        sys.argv = saved_argv

    output_path = f"{storage.data_path('predictions', 'lstm').rstrip('/')}/{run_timestamp}.csv"
    if not storage.exists(output_path):
        raise RuntimeError(f"LSTM prediction not written: {output_path}")
    logger.info(f"LSTM prediction written to {output_path}")
    return output_path


@task(name="predict-xgboost", retries=1)
def predict_pc_task(run_timestamp: str) -> str:
    """Run the per-site XGBoost prediction for the given weather batch.

    Returns the path of the written prediction CSV.
    """
    logger = get_run_logger()
    logger.info(f"Running XGBoost prediction for run_timestamp={run_timestamp}")

    import sys
    saved_argv = sys.argv
    sys.argv = [
        "predict_pc",
        "--run-timestamp", run_timestamp,
    ]
    try:
        predict_pc.main()
    finally:
        sys.argv = saved_argv

    output_path = f"{storage.data_path('predictions', 'pc').rstrip('/')}/{run_timestamp}.csv"
    if not storage.exists(output_path):
        raise RuntimeError(f"XGBoost prediction not written: {output_path}")
    logger.info(f"XGBoost prediction written to {output_path}")
    return output_path


# ---------- Evaluation ----------

@task(name="wait-for-ieso-actuals", retries=12, retry_delay_seconds=600)
def wait_for_ieso_actuals_task(target_date: str) -> str:
    """Poll for yesterday's IESO data to be in processed/.

    target_date: YYYY-MM-DD. The task raises (and retries) until a file
    containing that month's data is found, up to 2 hours total. Allows the
    evaluate flow to start at 08:15 in parallel with ingest_flow — if ingest
    finishes first, this returns immediately; if ingest is slow, this waits.
    """
    logger = get_run_logger()
    yyyymm = pd.Timestamp(target_date).strftime("%Y%m")
    pattern = f"{storage.data_path('processed', 'ieso').rstrip('/')}/*{yyyymm}*.csv"
    matches = storage.glob(pattern)
    if not matches:
        raise RuntimeError(
            f"No processed IESO file for {yyyymm} yet (waiting for ingest_flow to finish)."
        )
    logger.info(f"Found IESO data for {yyyymm}: {matches[0]}")
    return matches[0]


@task(name="evaluate-lstm", retries=1)
def evaluate_lstm_task(run_timestamp: str) -> Optional[dict]:
    """Evaluate the LSTM prediction CSV named {run_timestamp}.csv.

    Returns the metrics dict, or None if the prediction file doesn't exist
    (e.g. the LSTM run didn't happen that day — not an error).
    """
    logger = get_run_logger()
    pred_path = f"{storage.data_path('predictions', 'lstm').rstrip('/')}/{run_timestamp}.csv"
    if not storage.exists(pred_path):
        logger.warning(f"No LSTM prediction at {pred_path}; skipping eval.")
        return None
    metrics = evaluate_daily.run_evaluation(
        prediction_path=pred_path,
        model="lstm",
        ieso_dir=storage.data_path("processed", "ieso"),
        mapping_path=storage.data_path("mapping.csv"),
        output_dir=storage.data_path("evaluations"),
    )
    logger.info(f"LSTM MAE={metrics['mae_pct']:.2f}% over {metrics['n_rows']} rows")
    return metrics


@task(name="evaluate-pc", retries=1)
def evaluate_pc_task(run_timestamp: str) -> Optional[dict]:
    """Evaluate the XGBoost prediction CSV named {run_timestamp}.csv."""
    logger = get_run_logger()
    pred_path = f"{storage.data_path('predictions', 'pc').rstrip('/')}/{run_timestamp}.csv"
    if not storage.exists(pred_path):
        logger.warning(f"No XGBoost prediction at {pred_path}; skipping eval.")
        return None
    metrics = evaluate_daily.run_evaluation(
        prediction_path=pred_path,
        model="pc",
        ieso_dir=storage.data_path("processed", "ieso"),
        mapping_path=storage.data_path("mapping.csv"),
        output_dir=storage.data_path("evaluations"),
    )
    logger.info(f"XGBoost MAE={metrics['mae_pct']:.2f}% over {metrics['n_rows']} rows")
    return metrics


@task(name="evaluate-baseline", retries=1)
def evaluate_baseline_task(
    pred_start: pd.Timestamp, pred_end: pd.Timestamp
) -> Optional[dict]:
    """Compute the IESO baseline MAE over the given window for direct comparison."""
    logger = get_run_logger()
    metrics = evaluate_baseline.baseline_for_window(
        ieso_dir=storage.data_path("processed", "ieso"),
        mapping_path=storage.data_path("mapping.csv"),
        pred_start=pred_start,
        pred_end=pred_end,
    )
    if metrics is None:
        logger.warning("Baseline computation returned no data.")
        return None
    logger.info(f"IESO baseline MAE={metrics['mae_pct']:.2f}% over {metrics['n_rows']} rows")
    return metrics


# ---------- Helpers ----------

def find_yesterdays_run_timestamps(
    model: str, target_date: str
) -> list[str]:
    """List all run_timestamps for which a prediction CSV exists on target_date.

    Used by evaluate_flow to discover which predictions to evaluate without
    needing the upstream flow to publish a "manifest." The on-disk filenames
    are the manifest.

    Args:
        model: 'lstm' or 'pc'.
        target_date: YYYY-MM-DD.

    Returns:
        Sorted list of run_timestamp strings (YYYYMMDD_HHMM).
    """
    yyyymmdd = pd.Timestamp(target_date).strftime("%Y%m%d")
    pattern = f"{storage.data_path('predictions', model).rstrip('/')}/{yyyymmdd}_*.csv"
    paths = storage.glob(pattern)
    timestamps = []
    for p in paths:
        stem = p.rstrip("/").split("/")[-1].removesuffix(".csv")
        timestamps.append(stem)
    return sorted(timestamps)
"""
Evaluate flow: compute MAE for yesterday's predictions vs IESO actuals.

Schedule: 08:15 ET daily, in parallel with ingest_flow and predict_flow.

This flow has a soft dependency on ingest_flow having finished — it needs
yesterday's IESO actuals in processed/. Rather than using flow-to-flow
dependencies (more machinery, tighter coupling), the first task simply polls
for the processed file with a generous retry budget. If ingest is fast, this
returns instantly; if ingest is slow (late IESO publish), this waits up to
~2 hours before failing.

What gets evaluated:
    - All XGBoost predictions written yesterday (likely 4 — one per cycle).
    - The single LSTM prediction written yesterday at 08:15.
    - The IESO own-forecast baseline over the same window for each model run.

Each evaluation produces a metrics dict that gets logged. Persistent storage
of those metrics (W&B, Grafana, CSV log) is deferred until the dashboard
piece — for now the dict gets printed and the eval CSVs are written to GCS.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from prefect import flow, get_run_logger

from flows.tasks import (
    evaluate_baseline_task,
    evaluate_lstm_task,
    evaluate_pc_task,
    find_yesterdays_run_timestamps,
    wait_for_ieso_actuals_task,
)


@flow(name="wind-forecast-evaluate")
def evaluate_flow(target_date: str | None = None):
    """Evaluate yesterday's predictions.

    Args:
        target_date: YYYY-MM-DD. Defaults to yesterday (relative to run time).
                     Override for backfill: evaluate_flow("2026-05-25").
    """
    logger = get_run_logger()

    if target_date is None:
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info(f"evaluate_flow starting for target_date={target_date}")

    # Step 1: ensure yesterday's IESO data is on disk. Polls/retries until
    # ingest_flow has finished (or until the retry budget is exhausted).
    wait_for_ieso_actuals_task(target_date=target_date)

    # Step 2: discover every prediction run from target_date. The filenames
    # ARE the manifest — no upstream task hands us a list.
    lstm_runs = find_yesterdays_run_timestamps("lstm", target_date)
    pc_runs = find_yesterdays_run_timestamps("pc", target_date)
    logger.info(f"Found {len(lstm_runs)} LSTM run(s), {len(pc_runs)} XGBoost run(s)")

    # No predictions at all for target_date. Not an evaluation failure —
    # evaluate's contract is "score predictions that exist," not "detect that
    # predictions ran." A genuinely missing scheduled prediction is owned and
    # alerted by predict_flow's own run status (Prefect). This WARNING only
    # makes the absence visible in eval logs; the flow still exits cleanly.
    if not lstm_runs and not pc_runs:
        logger.warning(
            f"No prediction files found for target_date={target_date} "
            f"(neither lstm nor pc). Nothing to evaluate. If a prediction was "
            f"scheduled for this date, check predict_flow's run status."
        )
        
    all_metrics = []

    # Step 3: evaluate each prediction + a matching IESO baseline. For each
    # prediction we also compute the baseline over the same window so dashboard
    # plots show LSTM/XGB/IESO on directly comparable footing.
    for ts in lstm_runs:
        m = evaluate_lstm_task(run_timestamp=ts)
        if m is not None:
            all_metrics.append(m)
            base = evaluate_baseline_task(
                pred_start=pd.Timestamp(m["prediction_window_start"]),
                pred_end=pd.Timestamp(m["prediction_window_end"]),
            )
            if base is not None:
                all_metrics.append(base)

    for ts in pc_runs:
        m = evaluate_pc_task(run_timestamp=ts)
        if m is not None:
            all_metrics.append(m)
            base = evaluate_baseline_task(
                pred_start=pd.Timestamp(m["prediction_window_start"]),
                pred_end=pd.Timestamp(m["prediction_window_end"]),
            )
            if base is not None:
                all_metrics.append(base)

    logger.info(
        f"evaluate_flow complete: {len(all_metrics)} metric row(s) "
        f"({len(lstm_runs)} LSTM evals, {len(pc_runs)} XGBoost evals)"
    )
    for m in all_metrics:
        logger.info(
            f"  {m['model']:14s} ts={m['run_timestamp']:14s} "
            f"MAE={m['mae_pct']:5.2f}%  bias_cf={m['bias_cf']:+.4f}"
        )

    return all_metrics


if __name__ == "__main__":
    evaluate_flow()

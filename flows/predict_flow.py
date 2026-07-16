"""
Predict flow: fetch weather snapshot + run requested model(s).

Local / manual-run orchestration only — this is NOT the production path.
Production scheduling runs the standalone Cloud Run Job (`wf-daily`) once
daily at 02:00 ET, triggered by the thin fire-and-forget flow in
`orchestration/`. This flow exists for local development and ad-hoc/backfill
runs on a laptop.

Defaults to XGBoost only (`pc=True, lstm=False`) — matching the production
job, which serves the per-site XGBoost power curve. The LSTM path is dormant
by default: torch is imported lazily inside `predict_lstm_task`, so the
XGBoost-only run has no torch dependency. Pass `lstm=True` to run it (requires
the `[dl]` extra installed) — it stays available as a benchmark model without
being wired into the default path.

The flow generates one run_timestamp at the start and threads it through every
downstream task. That's what pairs the weather batch with the prediction
outputs and (later) with the evaluation.
"""

from __future__ import annotations

from datetime import datetime

from prefect import flow, get_run_logger

from flows.tasks import fetch_forecast_all_task, predict_lstm_task, predict_pc_task


@flow(name="wind-forecast-predict")
def predict_flow(lstm: bool = False, pc: bool = True, run_timestamp: str | None = None):
    """Run the predict pipeline for one batch.

    Args:
        lstm: Run the LSTM prediction.
        pc:   Run the per-site XGBoost prediction.
        run_timestamp: YYYYMMDD_HHMM; defaults to now. Override for backfill.
    """
    logger = get_run_logger()

    if not lstm and not pc:
        raise ValueError("predict_flow called with both lstm=False and pc=False")

    ts = run_timestamp or datetime.now().strftime("%Y%m%d_%H%M")
    logger.info(f"predict_flow starting (lstm={lstm}, pc={pc}, run_timestamp={ts})")

    # Step 1: fetch weather for all sites under this timestamp.
    fetch_forecast_all_task(run_timestamp=ts)

    # Step 2: run requested model(s). They share the weather batch but are
    # independent. XGBoost is the production model, so it runs first and is
    # allowed to fail the flow (you want to know if the prediction didn't
    # produce). The LSTM is a supplementary benchmark, so its failure is
    # captured as a state and logged — it neither fails the flow nor blocks
    # the XGBoost run.
    pc_path = predict_pc_task(run_timestamp=ts) if pc else None

    lstm_path = None
    if lstm:
        lstm_state = predict_lstm_task(run_timestamp=ts, return_state=True)
        if lstm_state.is_completed():
            lstm_path = lstm_state.result()
        else:
            logger.warning(f"LSTM prediction did not complete: {lstm_state}")

    logger.info(
        f"predict_flow complete (lstm={lstm_path}, pc={pc_path}, run_timestamp={ts})"
    )
    return {"run_timestamp": ts, "lstm_path": lstm_path, "pc_path": pc_path}


if __name__ == "__main__":
    predict_flow()

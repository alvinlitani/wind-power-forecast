"""
Ingest flow: download yesterday's IESO actuals + preprocess into the
processed/ieso/ layout the rest of the pipeline expects.

Schedule: 08:15 ET daily (after IESO publishes around 06:00 ET).

This flow is self-contained — it does not depend on, and is not depended on
by, any other flow's success at runtime. The evaluate_flow polls for this
flow's output rather than waiting on flow-to-flow dependency, so a stuck
ingest doesn't block predictions or vice versa.
"""

from __future__ import annotations

from prefect import flow, get_run_logger

from flows.tasks import fetch_actuals_task, preprocess_ieso_task


@flow(name="wind-forecast-ingest")
def ingest_flow():
    """Download IESO actuals + preprocess."""
    logger = get_run_logger()
    logger.info("Starting ingest flow")

    raw_dir = fetch_actuals_task()
    processed_dir = preprocess_ieso_task(input_dir=raw_dir)

    logger.info(f"Ingest complete -> {processed_dir}")
    return processed_dir


if __name__ == "__main__":
    ingest_flow()

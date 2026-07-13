"""
Daily pipeline entrypoint for the wind-forecast-job Cloud Run Job:
fetch weather for all sites, then run XGBoost power-curve prediction,
under one shared run_timestamp. Plain Python — no Prefect — so the Job
container stays orchestration-agnostic. Either stage raising aborts the
Job with a non-zero exit, which the external orchestrator sees as failed.
"""

import sys
from datetime import datetime

from wind_forecast.ingest import fetch_forecast_all
from wind_forecast.predict import predict_pc


def _run(module, run_timestamp):
    """Call a script main() with an explicit --run-timestamp, argv-isolated."""
    saved = sys.argv
    sys.argv = [module.__name__, "--run-timestamp", run_timestamp]
    try:
        module.main()
    finally:
        sys.argv = saved


def main():
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"=== Daily run {run_timestamp}: fetch ===", flush=True)
    _run(fetch_forecast_all, run_timestamp)
    print(f"=== Daily run {run_timestamp}: predict ===", flush=True)
    _run(predict_pc, run_timestamp)
    print(f"=== Daily run {run_timestamp}: complete ===", flush=True)


if __name__ == "__main__":
    main()
"""
Evaluate one prediction batch against IESO actuals and log to W&B.

Reads/writes wherever DATA_ROOT points (local or gs://), so this evaluates
cloud batches directly once IESO actuals have been ingested to the bucket.

Note on timing: a batch can only be evaluated once its window's actuals have
published (~1 day GOCR lag), so target a prediction from 2+ days ago.

Usage:
    python -m wind_forecast.evaluate.evaluate_and_log --run-timestamp 20260714_0200
    python -m wind_forecast.evaluate.evaluate_and_log --run-timestamp 20260714_0200 --no-wandb
"""

import argparse
import os

import pandas as pd

from wind_forecast import storage
from wind_forecast.evaluate import evaluate_daily


WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "wind-forecast-ontario")


def log_to_wandb(m: dict, project: str = WANDB_PROJECT):
    """Log one batch evaluation as a single W&B run."""
    import wandb

    model_name = m["model"]
    ts = m["run_timestamp"]
    window_start = pd.Timestamp(m["prediction_window_start"])

    config = {
        "model": model_name,
        "run_timestamp": ts,
        "prediction_date": window_start.strftime("%Y-%m-%d"),
        "forecast_horizon": "24h ahead, hourly",
        "window_start": m["prediction_window_start"],
        "window_end": m["prediction_window_end"],
        "normalization": "nameplate capacity",
    }

    metrics = {
        f"{model_name}/nmae_pct": m["nmae_pct"],
        f"{model_name}/nrmse_pct": m["nrmse_pct"],
        f"{model_name}/mae_mwh": m["mae_mwh"],
        f"{model_name}/rmse_mwh": m["rmse_mwh"],
        f"{model_name}/bias_cf": m["bias_cf"],
        f"{model_name}/nmae_capwtd_pct": m["nmae_capwtd_pct"],
        f"{model_name}/nrmse_capwtd_pct": m["nrmse_capwtd_pct"],
        f"{model_name}/nmae_fleet_pct": m["nmae_fleet_pct"],
        f"{model_name}/nrmse_fleet_pct": m["nrmse_fleet_pct"],
        f"{model_name}/bias_fleet_cf": m["bias_fleet_cf"],
        "n_hours_fleet": m["n_hours_fleet"],
        "n_sites": m["n_sites"],
        "n_rows": m["n_rows"],
    }
    if m.get("staleness_hours") is not None:
        metrics[f"{model_name}/staleness_hours"] = m["staleness_hours"]

    if m.get("persistence_nmae_pct") is not None:
        metrics["persistence/nmae_pct"] = m["persistence_nmae_pct"]
        metrics["persistence/nrmse_pct"] = m["persistence_nrmse_pct"]
        metrics["n_rows_persistence"] = m["n_rows_persistence"]
        if m.get("skill_score_mae") is not None:
            metrics["skill_score_mae"] = m["skill_score_mae"]
        if m.get("skill_score_rmse") is not None:
            metrics["skill_score_rmse"] = m["skill_score_rmse"]

    run = wandb.init(
        project=project,
        name=f"{model_name}_{ts}",
        job_type="evaluation",
        tags=[model_name, "eval", "24h-ahead"],
        config=config,
        reinit="finish_previous",
    )
    wandb.log(metrics)
    for k, v in metrics.items():
        run.summary[k] = v
    run.finish()
    print(f"\nLogged to W&B: project={project} run={model_name}_{ts}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a prediction batch vs IESO actuals and log to W&B."
    )
    parser.add_argument("--run-timestamp", required=True, help="Batch to evaluate, YYYYMMDD_HHMM.")
    parser.add_argument("--model", default="pc", help="Model subdir: 'pc' or 'lstm'.")
    parser.add_argument("--ieso-dir", default=None)
    parser.add_argument("--mapping-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-wandb", action="store_true",
                        help="Compute and print metrics without logging.")
    args = parser.parse_args()

    ieso_dir = args.ieso_dir or storage.data_path("processed", "ieso")
    mapping_path = args.mapping_csv or storage.data_path("mapping.csv")
    output_dir = args.output_dir or storage.data_path("evaluations")

    pred_path = (
        f"{storage.data_path('predictions', args.model).rstrip('/')}"
        f"/{args.run_timestamp}.csv"
    )

    metrics = evaluate_daily.run_evaluation(
        prediction_path=pred_path,
        model=args.model,
        ieso_dir=ieso_dir,
        mapping_path=mapping_path,
        output_dir=output_dir,
    )

    if not args.no_wandb:
        log_to_wandb(metrics)

    return metrics


if __name__ == "__main__":
    main()
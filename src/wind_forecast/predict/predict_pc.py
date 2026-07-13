"""
Run XGBoost power curve inference for all wind farm sites.

Reads weather forecasts from <DATA_ROOT>/predictions/weather/ (written by
fetch_forecast_all), runs prediction through the per-site XGBoost models,
and outputs hourly predicted MWh.

The weather batch is identified by run_timestamp (YYYYMMDD_HHMM), which
should match the timestamp passed to fetch_forecast_all in the same flow
run. If no run_timestamp is given, the latest snapshot for today is used.

Reads from:
    - <MODELS_ROOT>/pc/power_curves.pkl                            (XGBoost models)
    - <DATA_ROOT>/mapping.csv                                      (site coordinates + capacity)
    - <DATA_ROOT>/predictions/weather/{GENERATORID}_{YYYYMMDD_HHMM}.csv  (weather snapshot)

Writes to:
    - <DATA_ROOT>/predictions/pc/{YYYYMMDD_HHMM}.csv

Output columns:
    datetime, generator_id, predicted_mwh, predicted_cf, run_timestamp, code_sha

Usage:
    python -m wind_forecast.predict.predict_pc
    python -m wind_forecast.predict.predict_pc --run-timestamp 20260528_1400
"""

import argparse
import csv
import os
import pickle
import sys
from datetime import datetime

import pandas as pd

from wind_forecast import storage


FEATURE_COLS = [
    "wind_speed_80m",
    "wind_speed_120m",
    "temperature_2m",
    "surface_pressure",
]

# Number of future hours to predict
FORECAST_HOURS_OUTPUT = 24

# Sites legitimately expected to be absent from a complete prediction run.
# Empty today: every site that is in BOTH the model set and mapping.csv must
# produce a prediction. A future legitimate exclusion is a deliberate edit
# here, never a silently loosened comparison. (Sites present in the model
# pickle but missing from mapping.csv are a separate structural mismatch,
# surfaced as a hard error below — not an exclusion.)
EXPECTED_EXCLUSIONS: set[str] = set()


def load_mapping(mapping_path: str) -> dict[str, dict]:
    """Load generator mapping CSV into a dict keyed by generator_id (no spaces)."""
    generators = {}
    with storage.open_file(mapping_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gen_id = row["IESO name"].replace(" ", "")
            generators[gen_id] = {
                "ieso_name": row["IESO name"],
                "latitude": float(row["Latitude"]),
                "longitude": float(row["Longitude"]),
                "nameplate_capacity": float(row["Nameplate Capacity"]),
            }
    return generators


def find_latest_run_timestamp(weather_dir: str, date_str: str | None = None) -> str | None:
    """Find the most recent run_timestamp under weather_dir, optionally for a given date.

    Scans filenames like {GENERATORID}_{YYYYMMDD_HHMM}.csv and returns the
    maximum YYYYMMDD_HHMM. Used when --run-timestamp isn't passed explicitly.

    Args:
        weather_dir: Directory to scan.
        date_str: If given (YYYYMMDD), restrict to that date.

    Returns:
        The latest timestamp string, or None if no matching files exist.
    """
    pattern = f"{weather_dir.rstrip('/')}/*_*.csv"
    paths = storage.glob(pattern)
    timestamps = set()
    for p in paths:
        # filename: {GEN_ID}_{YYYYMMDD}_{HHMM}.csv → split on '_' from the right
        stem = p.rstrip("/").split("/")[-1].removesuffix(".csv")
        parts = stem.rsplit("_", 2)
        if len(parts) != 3:
            continue
        _, ymd, hm = parts
        if date_str is not None and ymd != date_str:
            continue
        if len(ymd) == 8 and ymd.isdigit() and len(hm) == 4 and hm.isdigit():
            timestamps.add(f"{ymd}_{hm}")
    return max(timestamps) if timestamps else None


def load_weather_snapshot(
    weather_dir: str, generator_id: str, run_timestamp: str
) -> pd.DataFrame | None:
    """Load a per-site weather snapshot CSV written by fetch_forecast_all.

    Returns the DataFrame with datetime parsed, or None if the file is
    missing (a site whose fetch failed in the upstream ingest task).
    """
    path = f"{weather_dir.rstrip('/')}/{generator_id}_{run_timestamp}.csv"
    if not storage.exists(path):
        return None
    df = storage.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def filter_to_future_24h(df: pd.DataFrame, now_local: datetime) -> pd.DataFrame:
    """Trim a weather snapshot to the next 24 hours strictly after the current hour.

    The on-disk snapshot covers a wider window (3 days back to 1 day forward)
    because the LSTM needs encoder context. The XGBoost power curve only
    predicts forward 24h, so it filters down to that slice here.
    """
    current_hour = now_local.replace(minute=0, second=0, microsecond=0)
    return df[df["datetime"] > current_hour].head(FORECAST_HOURS_OUTPUT)


def main():
    parser = argparse.ArgumentParser(
        description="Run XGBoost power curve predictions for all sites."
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "PC_MODEL_PATH", storage.models_path("pc", "power_curves.pkl")
        ),
        help="Path to trained models pickle (local or gs://). "
        "Defaults to <MODELS_ROOT>/pc/power_curves.pkl",
    )
    parser.add_argument(
        "--mapping-csv",
        default=storage.data_path("mapping.csv"),
        help="Path to mapping.csv (local or gs://). Defaults to <DATA_ROOT>/mapping.csv",
    )
    parser.add_argument(
        "--weather-dir",
        default=storage.data_path("predictions", "weather"),
        help="Directory holding weather snapshots from fetch_forecast_all "
        "(local or gs://). Defaults to <DATA_ROOT>/predictions/weather",
    )
    parser.add_argument(
        "--output-dir",
        default=storage.data_path("predictions", "pc"),
        help="Output directory (local or gs://). Defaults to <DATA_ROOT>/predictions/pc",
    )
    parser.add_argument(
        "--run-timestamp",
        default=None,
        help="YYYYMMDD_HHMM batch identifier shared with fetch_forecast_all. "
        "Defaults to the latest snapshot for today found under --weather-dir.",
    )
    args = parser.parse_args()

    # --- Load models ---
    model_path = args.model_path
    if not storage.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading models from {model_path}...")
    models = pickle.loads(storage.read_bytes(model_path))
    print(f"  {len(models)} site models loaded")

    # --- Load mapping ---
    mapping_path = args.mapping_csv
    if not storage.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    mapping = load_mapping(mapping_path)

    # --- Resolve run_timestamp ---
    # When called from the orchestrator, --run-timestamp is passed explicitly
    # to pair this prediction with the matching fetch_forecast_all batch.
    # For standalone runs, fall back to the latest snapshot found on disk.
    weather_dir = args.weather_dir
    run_timestamp = args.run_timestamp
    if run_timestamp is None:
        today_str = datetime.now().strftime("%Y%m%d")
        run_timestamp = find_latest_run_timestamp(weather_dir, date_str=today_str)
        if run_timestamp is None:
            raise RuntimeError(
                f"No weather snapshots for today ({today_str}) in {weather_dir}. "
                "Run fetch_forecast_all first, or pass --run-timestamp."
            )
        print(f"Using latest weather snapshot: {run_timestamp}")
    else:
        print(f"Using weather snapshot: {run_timestamp}")

    # now_local is used to trim each weather snapshot to the next 24 future
    # hours. Open-Meteo returns times in America/Toronto, and datetime.now()
    # on the worker is also America/Toronto (set by the container TZ).
    now_local = datetime.now()

    # code_sha identifies the exact code version that produced this batch.
    # CI injects the git SHA as an env var on the Cloud Run Job; local/dev
    # runs fall back to the literal "local". Read once here, stamped per row
    # below so provenance travels inside the file content.
    code_sha = os.environ.get("CODE_SHA", "local")

    print(f"\nLoading weather and predicting...\n")

    all_rows = []
    failed_sites = []
    produced_sites = set()
    not_in_mapping = []

    for gen_id in sorted(models.keys()):
        if gen_id not in mapping:
            print(f"  WARNING: {gen_id} in models but not in mapping, skipping.")
            not_in_mapping.append(gen_id)
            continue

        site = mapping[gen_id]
        cap = site["nameplate_capacity"]

        # Read this site's weather snapshot, trim to next 24h
        full_weather = load_weather_snapshot(weather_dir, gen_id, run_timestamp)
        if full_weather is None:
            print(f"  {gen_id}: weather snapshot not found, skipping")
            failed_sites.append(gen_id)
            continue
        weather_df = filter_to_future_24h(full_weather, now_local)
        if len(weather_df) == 0:
            print(f"  {gen_id}: no future hours in snapshot, skipping")
            failed_sites.append(gen_id)
            continue

        # Check for missing weather data
        missing = weather_df[FEATURE_COLS].isnull().any(axis=1).sum()
        if missing > 0:
            print(f"  {gen_id}: {missing}/{len(weather_df)} hours have missing weather data")

        # Predict
        X = weather_df[FEATURE_COLS]
        pred_mwh = models[gen_id].predict(X).clip(0, cap)
        pred_cf = pred_mwh / cap

        for idx, (_, row) in enumerate(weather_df.iterrows()):
            all_rows.append({
                "datetime": row["datetime"],
                # Written value is the IESO canonical name (spaced, e.g. "PORT BURWELL")
                # so prediction files match IESO's Generator identifier. Internal keys
                # (model lookup, weather filenames) remain stripped via gen_id.
                "generator_id": site["ieso_name"],
                "predicted_mwh": round(float(pred_mwh[idx]), 2),
                "predicted_cf": round(float(pred_cf[idx]), 4),
                "run_timestamp": run_timestamp,
                "code_sha": code_sha,
            })

        print(f"  {gen_id}: {len(weather_df)} hours, "
              f"avg predicted {pred_mwh.mean():.1f} MWh "
              f"(CF {pred_cf.mean():.1%})")
        produced_sites.add(gen_id)

    if not all_rows:
        raise RuntimeError("No predictions generated.")

    # --- Roster-completeness gate (fail closed) ---
    # This stage validates its own contract independently of the fetch stage:
    # every site that is in BOTH the model set and mapping (minus explicit
    # exclusions) must have produced a prediction. A shortfall means the
    # Ontario aggregate is silently incomplete, so we raise rather than write
    # a green-but-wrong prediction file.
    #
    # Two distinct failure classes, surfaced separately:
    #   - not_in_mapping: a model exists for a site absent from mapping.csv.
    #     A structural model/mapping drift -> hard error.
    #   - missing: an expected site produced no rows (weather snapshot absent
    #     or no future hours) -> hard error; recovery is upstream (re-fetch).
    if not_in_mapping:
        raise RuntimeError(
            f"Model/mapping drift: {len(not_in_mapping)} site(s) have models "
            f"but are absent from mapping.csv: {sorted(not_in_mapping)}"
        )

    expected = (set(models) & set(mapping)) - EXPECTED_EXCLUSIONS
    missing = expected - produced_sites
    if missing:
        raise RuntimeError(
            f"Incomplete prediction batch: {len(missing)} of {len(expected)} "
            f"expected sites missing predictions: {sorted(missing)}. "
            f"(failed_sites={sorted(failed_sites)})"
        )

    # --- Save output ---
    # Output filename matches the run_timestamp so the prediction CSV is
    # paired 1:1 with the weather batch that produced it.
    filename = f"{run_timestamp}.csv"
    output_path = f"{args.output_dir.rstrip('/')}/{filename}"

    output_df = pd.DataFrame(all_rows)
    storage.write_csv(output_df, output_path)

    pred_start = output_df["datetime"].min()
    pred_end = output_df["datetime"].max()

    print(f"\nPredictions saved to {output_path}")
    print(f"  {len(produced_sites)} sites × ~{FORECAST_HOURS_OUTPUT} hours "
          f"= {len(all_rows)} rows")
    print(f"  Prediction window: {pred_start} to {pred_end}")
    print(f"  Roster complete: {len(produced_sites)}/{len(expected)} expected sites.")


if __name__ == "__main__":
    main()
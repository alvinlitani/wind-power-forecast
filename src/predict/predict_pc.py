"""
Run XGBoost power curve inference for all wind farm sites.

Fetches the latest 24-hour weather forecast directly from Open-Meteo
for each site, runs prediction through the per-site XGBoost models,
and outputs hourly predicted MWh.

Unlike the LSTM predict.py which depends on IESO actuals for its
encoder window, this script is fully self-contained — it only needs
weather forecasts and can run at any time to get predictions based
on the latest available NWP model run.

Reads from:
    - models_pc/power_curves.pkl       (trained XGBoost models)
    - data/mapping.csv                 (site coordinates + nameplate capacity)

Writes to:
    - data/predictions/pc/{YYYYMMDD_HHMM}.csv

Output columns:
    datetime, generator_id, predicted_mwh, predicted_cf,
    forecast_fetched_at

Usage:
    python predict_pc.py
    python predict_pc.py --model-path models_pc/power_curves.pkl
"""

import argparse
import csv
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARIABLES = [
    "wind_speed_80m",
    "wind_speed_120m",
    "temperature_2m",
    "surface_pressure",
]

FEATURE_COLS = [
    "wind_speed_80m",
    "wind_speed_120m",
    "temperature_2m",
    "surface_pressure",
]

# Fetch more than 24h to allow filtering out past hours
FORECAST_HOURS_FETCH = 30
FORECAST_HOURS_OUTPUT = 24

# Seconds to wait between API requests to avoid rate limiting
API_DELAY = 0.5


def load_mapping(mapping_path: Path) -> dict[str, dict]:
    """Load generator mapping CSV into a dict keyed by generator_id (no spaces)."""
    generators = {}
    with open(mapping_path, "r", encoding="utf-8") as f:
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


def fetch_forecast(lat: float, lon: float, now_local: datetime) -> pd.DataFrame:
    """Fetch next 24h weather forecast from Open-Meteo.

    Fetches extra hours, then filters to only future hours (after the
    current clock hour) and takes the first 24.

    Args:
        lat: Site latitude.
        lon: Site longitude.
        now_local: Current local time (America/Toronto), used to filter
            out past hours.

    Returns DataFrame with columns:
        [datetime, wind_speed_80m, wind_speed_120m, temperature_2m, surface_pressure]
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARIABLES),
        "forecast_hours": FORECAST_HOURS_FETCH,
        "wind_speed_unit": "ms",
        "timezone": "America/Toronto",
    }

    response = requests.get(FORECAST_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "hourly" not in data:
        raise ValueError(f"Unexpected API response: {data}")

    df = pd.DataFrame({
        "datetime": pd.to_datetime(data["hourly"]["time"]),
        **{var: data["hourly"][var] for var in HOURLY_VARIABLES},
    })

    # Filter to future hours only (strictly after current hour start)
    current_hour = now_local.replace(minute=0, second=0, microsecond=0)
    df = df[df["datetime"] > current_hour].head(FORECAST_HOURS_OUTPUT)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run XGBoost power curve predictions for all sites."
    )
    parser.add_argument(
        "--model-path",
        default="../../models_pc/power_curves.pkl",
        help="Path to trained models pickle (default: models_pc/power_curves.pkl)",
    )
    parser.add_argument(
        "--mapping-csv",
        default="../../data/mapping.csv",
        help="Path to mapping.csv (default: data/mapping.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="../../data/predictions/pc",
        help="Output directory (default: data/predictions/pc)",
    )
    args = parser.parse_args()

    # --- Load models ---
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        sys.exit(1)

    print(f"Loading models from {model_path}...")
    with open(model_path, "rb") as f:
        models = pickle.load(f)
    print(f"  {len(models)} site models loaded")

    # --- Load mapping ---
    mapping_path = Path(args.mapping_csv)
    if not mapping_path.exists():
        print(f"Mapping file not found: {mapping_path}")
        sys.exit(1)

    mapping = load_mapping(mapping_path)

    # --- Fetch forecasts and predict ---
    fetch_time = datetime.now(timezone.utc)
    fetch_time_str = fetch_time.strftime("%Y-%m-%d %H:%M UTC")

    # Local time for filtering out past hours
    # Open-Meteo returns times in America/Toronto
    now_local = datetime.now()

    print(f"\nFetching forecasts and predicting (fetched at {fetch_time_str})...\n")

    all_rows = []
    failed_sites = []

    for gen_id in sorted(models.keys()):
        if gen_id not in mapping:
            print(f"  WARNING: {gen_id} in models but not in mapping, skipping.")
            continue

        site = mapping[gen_id]
        cap = site["nameplate_capacity"]

        try:
            weather_df = fetch_forecast(site["latitude"], site["longitude"], now_local)
        except Exception as e:
            print(f"  {gen_id}: fetch failed ({e})")
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
                "generator_id": gen_id,
                "predicted_mwh": round(float(pred_mwh[idx]), 2),
                "predicted_cf": round(float(pred_cf[idx]), 4),
                "forecast_fetched_at": fetch_time_str,
            })

        print(f"  {gen_id}: {len(weather_df)} hours, "
              f"avg predicted {pred_mwh.mean():.1f} MWh "
              f"(CF {pred_cf.mean():.1%})")

        # Rate limiting
        time.sleep(API_DELAY)

    if not all_rows:
        print("No predictions generated.")
        sys.exit(1)

    # --- Save output ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_now = datetime.now()
    filename = f"{local_now.strftime('%Y%m%d_%H%M')}.csv"
    output_path = output_dir / filename

    output_df = pd.DataFrame(all_rows)
    output_df.to_csv(output_path, index=False)

    pred_start = output_df["datetime"].min()
    pred_end = output_df["datetime"].max()

    print(f"\nPredictions saved to {output_path}")
    print(f"  {len(models) - len(failed_sites)} sites × {FORECAST_HOURS_OUTPUT} hours "
          f"= {len(all_rows)} rows")
    print(f"  Prediction window: {pred_start} to {pred_end}")
    if failed_sites:
        print(f"  Failed sites ({len(failed_sites)}): {', '.join(failed_sites)}")


if __name__ == "__main__":
    main()

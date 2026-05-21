"""
Fetch weather data from the Open-Meteo Forecast API for daily inference.

Fetches a broad window of weather data (3 days past + 1 day forward)
so that predict.py can align the encoder/decoder windows based on
IESO data availability.

Saves output to data/predictions/weather/{GENERATORID}_{YYYYMMDD}.csv
where the date is the run date.

Usage:
    python fetch_forecast.py --name K2WIND
    python fetch_forecast.py --name "BOW LAKE" --mapping-csv data/mapping.csv
    python fetch_forecast.py --name K2WIND --output-dir data/predictions/weather
"""

import argparse
import csv
import sys
from datetime import datetime, timedelta
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

# Fetch 3 days back + 1 day forward to cover any run time
PAST_DAYS = 3
FORWARD_DAYS = 1


def load_mapping(mapping_path: Path) -> dict[str, dict]:
    """Load generator mapping CSV into a dict keyed by IESO name."""
    generators = {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            generators[row["IESO name"]] = {
                "latitude": float(row["Latitude"]),
                "longitude": float(row["Longitude"]),
                "hub_height": float(row["Hub Height"]),
                "nameplate_capacity": float(row["Nameplate Capacity"]),
            }
    return generators


def fetch_forecast(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """Fetch weather from Open-Meteo Forecast API using date range.

    Args:
        lat: Site latitude.
        lon: Site longitude.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).

    Returns:
        API JSON response.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARIABLES),
        "start_date": start_date,
        "end_date": end_date,
        "wind_speed_unit": "ms",
        "timezone": "America/Toronto",
    }

    response = requests.get(FORECAST_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "hourly" not in data:
        raise ValueError(f"Unexpected API response: {data}")

    return data


def to_dataframe(data: dict) -> pd.DataFrame:
    """Convert API response to DataFrame.

    Returns:
        DataFrame with datetime and weather columns.
        No role column — predict.py assigns encoder/decoder based on
        IESO data availability.
    """
    df = pd.DataFrame({
        "datetime": pd.to_datetime(data["hourly"]["time"]),
        **{var: data["hourly"][var] for var in HOURLY_VARIABLES},
    })

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Fetch weather forecast for a single site."
    )
    parser.add_argument(
        "--name", required=True,
        help="Generator IESO name (e.g., K2WIND, 'BOW LAKE')",
    )
    parser.add_argument(
        "--mapping-csv",
        default="../../data/mapping.csv",
        help="Path to mapping.csv (default: data/mapping.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="../../data/predictions/weather",
        help="Output directory (default: data/predictions/weather)",
    )
    args = parser.parse_args()

    mapping_path = Path(args.mapping_csv)
    if not mapping_path.exists():
        print(f"Mapping file not found: {mapping_path}")
        sys.exit(1)

    generators = load_mapping(mapping_path)
    if args.name not in generators:
        print(f"Generator '{args.name}' not found in mapping.")
        print(f"Available: {', '.join(sorted(generators.keys()))}")
        sys.exit(1)

    site = generators[args.name]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Date range: 3 days back to 1 day forward
    today = datetime.now()
    start_date = (today - timedelta(days=PAST_DAYS)).strftime("%Y-%m-%d")
    end_date = (today + timedelta(days=FORWARD_DAYS)).strftime("%Y-%m-%d")

    # Filename uses generator_id (no spaces) and run date
    generator_id = args.name.replace(" ", "")
    run_date = today.strftime("%Y%m%d")
    filename = f"{generator_id}_{run_date}.csv"
    output_path = output_dir / filename

    if output_path.exists():
        print(f"Already exists: {output_path}")
        return

    print(f"Fetching forecast for {args.name} ({site['latitude']}, {site['longitude']})...")
    print(f"  Date range: {start_date} to {end_date}")
    data = fetch_forecast(site["latitude"], site["longitude"], start_date, end_date)
    df = to_dataframe(data)

    print(f"  {len(df)} hourly rows")
    print(f"  Range: {df['datetime'].min()} to {df['datetime'].max()}")

    df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")


if __name__ == "__main__":
    main()

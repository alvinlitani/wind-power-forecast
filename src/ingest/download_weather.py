"""
Download weather data from Open-Meteo Historical Forecast API for a single generator.

Reads generator coordinates and hub height from data/mapping.csv.
Saves output to data/raw/weather/{GENERATOR}_{YYYYMMDD}_{YYYYMMDD}.csv

Usage:
    python download_weather.py --name K2WIND --start 2023-01-01 --end 2023-03-31
    python download_weather.py --name "BOW LAKE" --start 2023-01-01 --end 2023-03-31

The script fetches hourly data from the Historical Forecast API (not the Archive API),
so training data matches production forecast-quality inputs.

Variables fetched:
    - wind_speed_80m, wind_speed_120m (for hub-height interpolation)
    - temperature_2m
    - surface_pressure
    - boundary_layer_height
"""

import argparse
import csv
import sys
from pathlib import Path

import requests


HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

HOURLY_VARIABLES = [
    "wind_speed_80m",
    "wind_speed_120m",
    "temperature_2m",
    "surface_pressure",
]


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


def fetch_weather(lat: float, lon: float, start: str, end: str) -> dict:
    """Fetch hourly weather data from Open-Meteo Historical Forecast API.

    Args:
        lat: Latitude
        lon: Longitude
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format

    Returns:
        JSON response dict from Open-Meteo
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "auto",     # if not set to auto or America/Toronto, it will download according to UTC which will mismatch IESO data
    }

    resp = requests.get(HISTORICAL_FORECAST_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def write_csv(data: dict, output_path: Path) -> int:
    """Write Open-Meteo hourly response to CSV.

    Returns number of rows written.
    """
    hourly = data["hourly"]
    times = hourly["time"]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["time"] + HOURLY_VARIABLES
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, t in enumerate(times):
            row = {"time": t}
            for var in HOURLY_VARIABLES:
                row[var] = hourly[var][i]
            writer.writerow(row)

    return len(times)


def main():
    parser = argparse.ArgumentParser(
        description="Download Open-Meteo historical forecast data for a wind generator."
    )
    parser.add_argument(
        "--name",
        required=True,
        help="IESO generator name (e.g. K2WIND, 'BOW LAKE')",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--mapping",
        default="../../data/mapping.csv",
        help="Path to generator mapping CSV (default: data/mapping.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="../../data/raw/weather",
        help="Output directory (default: data/raw/weather)",
    )
    args = parser.parse_args()

    mapping_path = Path(args.mapping)
    if not mapping_path.exists():
        print(f"Mapping file not found: {mapping_path}")
        sys.exit(1)

    generators = load_mapping(mapping_path)
    if args.name not in generators:
        print(f"Generator '{args.name}' not found in mapping. Available:")
        for name in sorted(generators.keys()):
            print(f"  {name}")
        sys.exit(1)

    gen = generators[args.name]
    start_clean = args.start.replace("-", "")
    end_clean = args.end.replace("-", "")
    name_clean = args.name.replace(" ", "")
    filename = f"{name_clean}_{start_clean}_{end_clean}.csv"
    output_path = Path(args.output_dir) / filename

    if output_path.exists():
        print(f"SKIP {filename} (already exists)")
        return

    print(f"Fetching weather for {args.name}")
    print(f"Coordinates: {gen['latitude']}, {gen['longitude']}")
    print(f"Hub height: {gen['hub_height']}m")
    print(f"Date range: {args.start} to {args.end}")

    try:
        data = fetch_weather(gen["latitude"], gen["longitude"], args.start, args.end)
    except requests.RequestException as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    n_rows = write_csv(data, output_path)
    print(f"  OK: {n_rows} hourly rows -> {output_path}")


if __name__ == "__main__":
    main()
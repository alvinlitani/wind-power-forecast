"""
Fetch weather data from the Open-Meteo Forecast API for daily inference.

Fetches a broad window of weather data (3 days past + 1 day forward)
so that predict.py can align the encoder/decoder windows based on
IESO data availability.

Saves output to <DATA_ROOT>/predictions/weather/{GENERATORID}_{YYYYMMDD_HHMM}.csv
where the timestamp is the run time. Per-run timestamps let the LSTM (once
daily) and XGBoost (every 6 hours) flows coexist in the same directory with
each run's weather snapshot preserved for audit.

Usage:
    python -m wind_forecast.ingest.fetch_forecast --name K2WIND
    python -m wind_forecast.ingest.fetch_forecast --name "BOW LAKE" --run-timestamp 20260528_1100
"""

import argparse
import csv
import sys
from datetime import datetime, timedelta

from wind_forecast import storage

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _make_session() -> requests.Session:
    """Session with retry on connection errors and rate-limit/5xx responses.

    Open-Meteo's free tier sheds load under bursts (connection resets) and
    returns 429 when rate-limited. Retry with exponential backoff handles
    both. backoff_factor=1 -> waits 0s, 2s, 4s, 8s between attempts.
    respect_retry_after_header honors Open-Meteo's 429 Retry-After.
    """
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_SESSION = _make_session()




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


def load_mapping(mapping_path: str) -> dict[str, dict]:
    """Load generator mapping CSV into a dict keyed by IESO name."""
    generators = {}
    with storage.open_file(mapping_path, "r") as f:
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
        # "wind_speed_unit": "ms",   both predict and training should use same units
        "timezone": "America/Toronto",
    }

    response = _SESSION.get(FORECAST_URL, params=params, timeout=30)
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


def fetch_site(
    name: str,
    generators: dict[str, dict],
    output_dir: str,
    run_timestamp: str | None = None,
    today: datetime | None = None,
) -> str | None:
    """Fetch and save the forecast for a single site.

    Pure orchestration around the API call so it can be reused both by this
    script's CLI and by fetch_forecast_all (in-process, no subprocess).

    The output filename includes a YYYYMMDD_HHMM timestamp so multiple runs
    per day each produce their own snapshot (the LSTM flow runs once daily,
    the XGBoost flow every 6 hours, and we want a per-run audit trail of
    exactly which weather drove each prediction).

    Args:
        name: IESO generator name (must be a key in `generators`).
        generators: Mapping dict from load_mapping().
        output_dir: Output directory (local path or gs:// URI).
        run_timestamp: YYYYMMDD_HHMM string for the filename. If None, uses
            the current time. Pass explicitly from the orchestrator so all
            sites in a single flow run share one timestamp.
        today: Run date for the API date window; defaults to now. Injectable
            for testing.

    Returns:
        The output path written, or None if the site was not found.
    """
    if name not in generators:
        return None

    site = generators[name]
    if today is None:
        today = datetime.now()
    if run_timestamp is None:
        run_timestamp = today.strftime("%Y%m%d_%H%M")

    start_date = (today - timedelta(days=PAST_DAYS)).strftime("%Y-%m-%d")
    end_date = (today + timedelta(days=FORWARD_DAYS)).strftime("%Y-%m-%d")

    generator_id = name.replace(" ", "")
    filename = f"{generator_id}_{run_timestamp}.csv"
    output_path = f"{output_dir.rstrip('/')}/{filename}"

    print(f"Fetching forecast for {name} ({site['latitude']}, {site['longitude']})...")
    print(f"  Date range: {start_date} to {end_date}")
    data = fetch_forecast(site["latitude"], site["longitude"], start_date, end_date)
    df = to_dataframe(data)

    print(f"  {len(df)} hourly rows")
    print(f"  Range: {df['datetime'].min()} to {df['datetime'].max()}")

    storage.write_csv(df, output_path)
    print(f"  Saved to {output_path}")
    return output_path


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
        default=storage.data_path("mapping.csv"),
        help="Path to mapping.csv (local or gs://). Defaults to <DATA_ROOT>/mapping.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=storage.data_path("predictions", "weather"),
        help="Output directory (local or gs://). Defaults to <DATA_ROOT>/predictions/weather",
    )
    parser.add_argument(
        "--run-timestamp",
        default=None,
        help="Timestamp for the output filename (YYYYMMDD_HHMM). Defaults to now. "
        "Pass this from the orchestrator so all sites in one run share the same timestamp.",
    )
    args = parser.parse_args()

    mapping_path = args.mapping_csv
    if not storage.exists(mapping_path):
        print(f"Mapping file not found: {mapping_path}")
        sys.exit(1)

    generators = load_mapping(mapping_path)
    if args.name not in generators:
        print(f"Generator '{args.name}' not found in mapping.")
        print(f"Available: {', '.join(sorted(generators.keys()))}")
        sys.exit(1)

    fetch_site(args.name, generators, args.output_dir, run_timestamp=args.run_timestamp)


if __name__ == "__main__":
    main()

"""
Feature engineering for power curve (XGBoost) model.

Joins IESO generation data with Open-Meteo weather data using raw
wind speed values at 80m and 120m — no hub-height interpolation,
no derived features.

Reads from:
    - data/processed/ieso/{year}/*.csv  (preprocessed IESO generation data)
    - data/raw/weather/*.csv            (Open-Meteo weather data)

Writes to:
    - data/processed/features_pc.csv

Output columns:
    datetime, generator_id, output_mwh, available_capacity_mw,
    wind_speed_80m, wind_speed_120m, temperature_2m, surface_pressure

Usage:
    python build_features_pc.py
    python build_features_pc.py --ieso-dir data/processed/ieso --weather-dir data/raw/weather
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Maximum gap length (hours) for linear interpolation of NaN values.
# Gaps longer than this are left as NaN.
MAX_INTERP_GAP = 6


def load_ieso(ieso_dir: Path) -> pd.DataFrame:
    """Load all preprocessed IESO CSVs and melt to long format.

    Returns DataFrame with columns: [datetime, generator, measure_type, value]
    where measure_type is one of: Output, Available Capacity, Forecast
    """
    all_frames = []
    csv_files = sorted(ieso_dir.rglob("*.csv"))

    if not csv_files:
        print(f"No IESO CSV files found in {ieso_dir}")
        sys.exit(1)

    for f in csv_files:
        df = pd.read_csv(f)

        # Filter to WIND fuel type only
        df = df[df["Fuel Type"] == "WIND"].copy()

        # Melt hour columns from wide to long
        hour_cols = [c for c in df.columns if c.startswith("Hour ")]
        id_cols = [c for c in df.columns if not c.startswith("Hour ")]

        melted = df.melt(
            id_vars=id_cols,
            value_vars=hour_cols,
            var_name="hour_col",
            value_name="value",
        )

        # Extract hour number (Hour 1 = hour-ending 1 = hour-starting 0)
        melted["hour"] = melted["hour_col"].str.extract(r"(\d+)").astype(int) - 1
        melted["datetime"] = pd.to_datetime(melted["Delivery Date"]) + pd.to_timedelta(
            melted["hour"], unit="h"
        )

        melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
        melted = melted[["datetime", "Generator", "Measurement", "value"]]
        melted.columns = ["datetime", "generator", "measure_type", "value"]

        all_frames.append(melted)

    combined = pd.concat(all_frames, ignore_index=True)
    return combined


def pivot_ieso(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot IESO data so Output and Available Capacity are separate columns.

    Returns DataFrame with columns:
        [datetime, generator, output_mwh, available_capacity_mw]
    """
    df = df[df["measure_type"].isin(["Output", "Available Capacity"])].copy()

    pivoted = df.pivot_table(
        index=["datetime", "generator"],
        columns="measure_type",
        values="value",
        aggfunc="first",
    ).reset_index()

    pivoted.columns.name = None
    pivoted = pivoted.rename(
        columns={
            "Output": "output_mwh",
            "Available Capacity": "available_capacity_mw",
        }
    )

    return pivoted


def interpolate_gaps(
    df: pd.DataFrame, columns: list[str], max_gap: int
) -> pd.DataFrame:
    """Interpolate NaN values in specified columns per generator.

    Only fills gaps of max_gap or fewer consecutive NaN hours.
    Longer gaps are left as NaN.
    """
    result = df.copy()

    for gen, group in result.groupby("generator"):
        group = group.sort_values("datetime")

        for col in columns:
            series = group[col].copy()

            # Identify NaN runs and their lengths
            is_nan = series.isna()
            nan_groups = is_nan.ne(is_nan.shift()).cumsum()
            nan_lengths = is_nan.groupby(nan_groups).transform("sum")

            # Mask: only interpolate where gap length <= max_gap
            short_gaps = is_nan & (nan_lengths <= max_gap)

            # Interpolate all NaN, then restore long gaps
            interpolated = series.interpolate(method="linear", limit_direction="both")
            series = series.where(~short_gaps, interpolated)

            result.loc[group.index, col] = series

    return result


def load_weather(weather_dir: Path) -> pd.DataFrame:
    """Load all weather CSVs from data/raw/weather/.

    Extracts generator name from filename pattern:
        GENERATORNAME_YYYYMMDD_YYYYMMDD.csv

    Returns DataFrame with columns:
        [datetime, generator, wind_speed_80m, wind_speed_120m,
         temperature_2m, surface_pressure]
    """
    all_frames = []
    csv_files = sorted(weather_dir.glob("*.csv"))

    if not csv_files:
        print(f"No weather CSV files found in {weather_dir}")
        sys.exit(1)

    for f in csv_files:
        # Filename: GENERATORNAME_YYYYMMDD_YYYYMMDD.csv
        # Generator name is everything before the first _YYYYMMDD pattern
        parts = f.stem.split("_")
        # Last two parts are date ranges, everything before is the generator name
        gen_name = "_".join(parts[:-2])

        df = pd.read_csv(f)
        df["datetime"] = pd.to_datetime(df["time"])
        df["generator"] = gen_name
        df = df.drop(columns=["time"])

        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Build power curve feature dataset from IESO and weather data."
    )
    parser.add_argument(
        "--ieso-dir",
        default="../../data/ieso",
        help="Processed IESO data directory (default: data/processed/ieso)",
    )
    parser.add_argument(
        "--weather-dir",
        default="../../data/raw/weather",
        help="Weather data directory (default: data/raw/weather)",
    )
    parser.add_argument(
        "--output",
        default="../../data/processed/features_pc.csv",
        help="Output CSV path (default: data/processed/features_pc.csv)",
    )
    args = parser.parse_args()

    # --- Load IESO data ---
    print("Loading IESO data...")
    ieso_raw = load_ieso(Path(args.ieso_dir))
    ieso = pivot_ieso(ieso_raw)
    print(f"  {len(ieso)} rows, {ieso['generator'].nunique()} generators")

    # --- Interpolate short NaN gaps ---
    print(f"Interpolating NaN gaps (max {MAX_INTERP_GAP} hours)...")
    ieso = interpolate_gaps(ieso, ["output_mwh", "available_capacity_mw"], MAX_INTERP_GAP)

    # --- Load weather data ---
    print("Loading weather data...")
    weather = load_weather(Path(args.weather_dir))
    print(f"  {len(weather)} rows, {weather['generator'].nunique()} generators")

    # --- Map IESO generator names to sanitized names for join ---
    # IESO uses names with spaces, weather filenames have spaces removed
    ieso["generator_id"] = ieso["generator"].str.replace(" ", "", regex=False)

    # --- Join IESO + weather ---
    print("Joining IESO and weather data...")
    merged = pd.merge(
        ieso,
        weather,
        left_on=["datetime", "generator_id"],
        right_on=["datetime", "generator"],
        how="inner",
        suffixes=("", "_weather"),
    )
    merged = merged.drop(columns=["generator_weather", "generator"])
    print(f"  {len(merged)} rows after join")

    # --- Select output columns ---
    output_cols = [
        "datetime",
        "generator_id",
        "output_mwh",
        "available_capacity_mw",
        "wind_speed_80m",
        "wind_speed_120m",
        "temperature_2m",
        "surface_pressure",
    ]

    # Check all expected weather columns exist
    missing = [c for c in output_cols if c not in merged.columns]
    if missing:
        print(f"ERROR: Missing columns in merged data: {missing}")
        print(f"  Available columns: {sorted(merged.columns.tolist())}")
        sys.exit(1)

    result = merged[output_cols].copy()

    # --- Drop rows with NaN in any feature or target column ---
    before = len(result)
    result = result.dropna(subset=[
        "output_mwh", "wind_speed_80m", "wind_speed_120m",
        "temperature_2m", "surface_pressure",
    ])
    dropped = before - len(result)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with NaN values ({dropped/before*100:.2f}%)")

    # --- Sort and save ---
    result = result.sort_values(["generator_id", "datetime"]).reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"\nSaved {len(result)} rows to {output_path}")
    print(f"  Generators: {result['generator_id'].nunique()}")
    print(f"  Date range: {result['datetime'].min()} to {result['datetime'].max()}")


if __name__ == "__main__":
    main()

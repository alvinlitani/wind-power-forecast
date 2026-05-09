"""
Feature engineering: join IESO generation data with Open-Meteo weather data,
derive hub-height wind speed, attach static site features.

Reads from:
    - data/processed/ieso/{year}/*.csv  (preprocessed IESO generation data)
    - data/raw/weather/*.csv            (Open-Meteo weather data)
    - data/mapping.csv                  (generator coordinates, hub height, capacity)

Writes to:
    - data/processed/features.csv

Usage:
    python build_features.py
    python build_features.py --ieso-dir data/processed/ieso --weather-dir data/raw/weather
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

# Surface roughness for open farmland (m)
Z0 = 0.03

# Open-Meteo wind speed measurement heights (m)
WIND_HEIGHT_LOW = 80
WIND_HEIGHT_HIGH = 120

# Hub height snap threshold (m). If hub height is within this distance
# of a measurement level, use that level directly instead of interpolating.
SNAP_THRESHOLD = 10

# Maximum gap length (hours) for linear interpolation of NaN values.
# Gaps longer than this are left as NaN and skipped during sequence construction.
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

        melted = melted[["datetime", "Generator", "Measurement", "value"]]
        melted.columns = ["datetime", "generator", "measure_type", "value"]

        all_frames.append(melted)

    combined = pd.concat(all_frames, ignore_index=True)

    # Convert value to numeric, coercing N/A and other non-numeric to NaN
    combined["value"] = pd.to_numeric(combined["value"], errors="coerce")

    return combined


def pivot_ieso(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot IESO data so Output and Available Capacity are separate columns.

    Returns DataFrame with columns: [datetime, generator, output_mwh, available_capacity_mw]
    """
    # Keep only Output and Available Capacity
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


def interpolate_gaps(df: pd.DataFrame, columns: list[str], max_gap: int) -> pd.DataFrame:
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

    Extracts generator name from filename pattern: GENERATORNAME_YYYYMMDD_YYYYMMDD.csv
    Returns DataFrame with columns: [datetime, generator, wind_speed_80m, wind_speed_120m,
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


def build_generator_lookup(mapping_path: Path) -> dict:
    """Build a lookup from sanitized generator name to mapping info.

    Returns dict keyed by generator name (spaces removed) with values:
        {hub_height, nameplate_capacity, latitude, longitude}
    """
    df = pd.read_csv(mapping_path)
    lookup = {}
    for _, row in df.iterrows():
        name_clean = row["IESO name"].replace(" ", "")
        lookup[name_clean] = {
            "ieso_name": row["IESO name"],
            "hub_height": row["Hub Height"],
            "nameplate_capacity": row["Nameplate Capacity"],
        }
    return lookup


def compute_wind_speed_hub(
    ws_low: pd.Series,
    ws_high: pd.Series,
    hub_height: float,
) -> pd.Series:
    """Compute hub-height wind speed using log wind profile interpolation.

    If hub height is within SNAP_THRESHOLD of a measurement level, snaps
    to that level. Otherwise interpolates using the log wind profile:

        u(z) = u(z_ref) * ln(z/z0) / ln(z_ref/z0)

    Uses the two measurement levels to derive the effective roughness,
    then interpolates to hub height.
    """
    # Check snap conditions
    if abs(hub_height - WIND_HEIGHT_LOW) <= SNAP_THRESHOLD:
        return ws_low
    if abs(hub_height - WIND_HEIGHT_HIGH) <= SNAP_THRESHOLD:
        return ws_high

    # Log wind profile interpolation between the two levels
    log_hub = math.log(hub_height / Z0)
    log_low = math.log(WIND_HEIGHT_LOW / Z0)
    log_high = math.log(WIND_HEIGHT_HIGH / Z0)

    # Interpolation weight: where hub_height falls between low and high
    # in log space
    weight = (log_hub - log_low) / (log_high - log_low)
    return ws_low + weight * (ws_high - ws_low)


def main():
    parser = argparse.ArgumentParser(
        description="Build feature dataset from IESO and weather data."
    )
    parser.add_argument(
        "--ieso-dir",
        default="data/ieso",
        help="Processed IESO data directory (default: data/processed/ieso)",
    )
    parser.add_argument(
        "--weather-dir",
        default="data/raw/weather",
        help="Weather data directory (default: data/raw/weather)",
    )
    parser.add_argument(
        "--mapping",
        default="data/mapping.csv",
        help="Generator mapping CSV (default: data/mapping.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/processed/features.csv",
        help="Output CSV path (default: data/processed/features.csv)",
    )
    args = parser.parse_args()

    mapping_path = Path(args.mapping)
    gen_lookup = build_generator_lookup(mapping_path)

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
    ieso["generator_clean"] = ieso["generator"].str.replace(" ", "", regex=False)

    # --- Join IESO + weather ---
    print("Joining IESO and weather data...")
    merged = pd.merge(
        ieso,
        weather,
        left_on=["datetime", "generator_clean"],
        right_on=["datetime", "generator"],
        how="inner",
        suffixes=("", "_weather"),
    )
    merged = merged.drop(columns=["generator_weather"])
    merged = merged.rename(columns={"generator_clean": "generator_id"})

    print(f"  {len(merged)} rows after join")

    # --- Derive features ---
    print("Deriving features...")

    # Wind speed at hub height (per generator based on hub height)
    merged["wind_speed_hub"] = 0.0
    for gen_clean, gen_info in gen_lookup.items():
        mask = merged["generator_id"] == gen_clean
        if mask.sum() == 0:
            continue
        merged.loc[mask, "wind_speed_hub"] = compute_wind_speed_hub(
            merged.loc[mask, "wind_speed_80m"],
            merged.loc[mask, "wind_speed_120m"],
            gen_info["hub_height"],
        )

    # --- Attach static features ---
    print("Attaching static features...")
    merged["capacity_mw"] = merged["generator_id"].map(
        {k: v["nameplate_capacity"] for k, v in gen_lookup.items()}
    )
    merged["hub_height"] = merged["generator_id"].map(
        {k: v["hub_height"] for k, v in gen_lookup.items()}
    )

    # Site ID: integer index from alphabetically sorted generator names
    sorted_generators = sorted(merged["generator_id"].unique())
    site_id_map = {name: idx for idx, name in enumerate(sorted_generators)}
    merged["site_id"] = merged["generator_id"].map(site_id_map)

    # Generator name for debugging (original IESO name with spaces)
    merged["generator_name"] = merged["generator"]

    # --- Select and order output columns ---
    output_cols = [
        "datetime",
        "generator_name",
        "generator_id",
        "site_id",
        "output_mwh",
        "available_capacity_mw",
        "wind_speed_hub",
        "temperature_2m",
        "surface_pressure",
        "capacity_mw",
        "hub_height",
    ]
    merged = merged[output_cols].sort_values(["generator_id", "datetime"])

    # --- Write output ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"\nWrote {len(merged)} rows to {output_path}")

    # --- Summary ---
    n_generators = merged["generator_id"].nunique()
    date_min = merged["datetime"].min()
    date_max = merged["datetime"].max()
    n_nan_output = merged["output_mwh"].isna().sum()
    n_nan_capacity = merged["available_capacity_mw"].isna().sum()
    print(f"  Generators: {n_generators}")
    print(f"  Date range: {date_min} to {date_max}")
    print(f"  Remaining NaN in output_mwh: {n_nan_output}")
    print(f"  Remaining NaN in available_capacity_mw: {n_nan_capacity}")


if __name__ == "__main__":
    main()
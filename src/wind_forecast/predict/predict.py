"""
Run inference for all wind farm sites.

Determines the prediction window based on IESO data availability:
the encoder uses the last 48 hours of available IESO data, and the
decoder predicts the 24 hours immediately following. This allows the
pipeline to run at any time — the prediction window adjusts to whatever
data is available.

Reads from:
    - <MODELS_ROOT>/cf/best_model.pt, norm_stats.pt, config.pt
    - <DATA_ROOT>/processed/ieso/*.csv                             (preprocessed IESO generation data)
    - <DATA_ROOT>/predictions/weather/{GENERATORID}_{YYYYMMDD_HHMM}.csv  (weather snapshot)
    - <DATA_ROOT>/mapping.csv

Writes to:
    - <DATA_ROOT>/predictions/lstm/{YYYYMMDD_HHMM}.csv

Usage:
    python -m wind_forecast.predict.predict
    python -m wind_forecast.predict.predict --run-timestamp 20260528_1100
"""

import argparse
import csv
import io
import math
import sys
from datetime import datetime, timedelta

import pandas as pd
import torch

from wind_forecast import storage
from wind_forecast.model import WindPowerLSTM

# Surface roughness for open farmland (m)
Z0 = 0.03

# Open-Meteo wind speed measurement heights (m)
WIND_HEIGHT_LOW = 80
WIND_HEIGHT_HIGH = 120

# Snap threshold (m)
SNAP_THRESHOLD = 10

# Sequence lengths
ENCODER_STEPS = 48
DECODER_STEPS = 24


# ============================================================
# Data loading
# ============================================================

def load_mapping(mapping_path: str) -> dict[str, dict]:
    """Load generator mapping CSV.

    Returns dict keyed by IESO name (with spaces) with values:
        {generator_id, hub_height, nameplate_capacity}
    """
    generators = {}
    with storage.open_file(mapping_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["IESO name"]
            generators[name] = {
                "generator_id": name.replace(" ", ""),
                "hub_height": float(row["Hub Height"]),
                "nameplate_capacity": float(row["Nameplate Capacity"]),
            }
    return generators


def build_site_id_map(mapping: dict[str, dict]) -> dict[str, int]:
    """Build site_id lookup from alphabetically sorted generator_ids."""
    gen_ids = sorted(info["generator_id"] for info in mapping.values())
    return {gid: idx for idx, gid in enumerate(gen_ids)}


def load_ieso_actuals(ieso_dir: str) -> pd.DataFrame:
    """Load IESO preprocessed CSVs, returning all available data.

    Loads all CSVs found directly under ieso_dir (flat layout). The exact
    encoder window is determined later based on the latest available hour.

    Returns DataFrame with columns:
        [datetime, generator, output_mwh, available_capacity_mw]
    """
    all_frames = []
    csv_files = sorted(storage.glob(f"{ieso_dir.rstrip('/')}/*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No IESO CSV files found in {ieso_dir}")

    for f in csv_files:
        df = storage.read_csv(f)

        # Filter to WIND fuel type only
        df = df[df["Fuel Type"] == "WIND"].copy()

        # Melt hour columns
        hour_cols = [c for c in df.columns if c.startswith("Hour ")]
        id_cols = [c for c in df.columns if not c.startswith("Hour ")]

        melted = df.melt(
            id_vars=id_cols,
            value_vars=hour_cols,
            var_name="hour_col",
            value_name="value",
        )

        melted["hour"] = melted["hour_col"].str.extract(r"(\d+)").astype(int) - 1
        melted["datetime"] = pd.to_datetime(melted["Delivery Date"]) + pd.to_timedelta(
            melted["hour"], unit="h"
        )
        melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
        melted = melted[["datetime", "Generator", "Measurement", "value"]]
        melted.columns = ["datetime", "generator", "measure_type", "value"]

        all_frames.append(melted)

    combined = pd.concat(all_frames, ignore_index=True)

    # Pivot to get output_mwh and available_capacity_mw columns
    combined = combined[
        combined["measure_type"].isin(["Output", "Available Capacity"])
    ]
    pivoted = combined.pivot_table(
        index=["datetime", "generator"],
        columns="measure_type",
        values="value",
        aggfunc="first",
    ).reset_index()
    pivoted.columns.name = None
    pivoted = pivoted.rename(columns={
        "Output": "output_mwh",
        "Available Capacity": "available_capacity_mw",
    })

    return pivoted


def load_weather_forecast(
    weather_dir: str, generator_id: str, run_timestamp: str
) -> pd.DataFrame | None:
    """Load a single site's weather snapshot CSV.

    The snapshot was written by fetch_forecast_all under the matching
    run_timestamp (YYYYMMDD_HHMM). Returns None if the file is missing
    (a site whose ingest failed upstream).
    """
    filename = f"{generator_id}_{run_timestamp}.csv"
    path = f"{weather_dir.rstrip('/')}/{filename}"

    if not storage.exists(path):
        return None

    df = storage.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def find_latest_run_timestamp(weather_dir: str, date_str: str | None = None) -> str | None:
    """Find the most recent run_timestamp under weather_dir, optionally for a given date.

    Scans filenames like {GENERATORID}_{YYYYMMDD_HHMM}.csv and returns the
    maximum YYYYMMDD_HHMM. Used when --run-timestamp isn't passed explicitly.
    """
    pattern = f"{weather_dir.rstrip('/')}/*_*.csv"
    paths = storage.glob(pattern)
    timestamps = set()
    for p in paths:
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


# ============================================================
# Feature derivation
# ============================================================

def compute_wind_speed_hub(
    ws_low: pd.Series,
    ws_high: pd.Series,
    hub_height: float,
) -> pd.Series:
    """Compute hub-height wind speed using log wind profile interpolation."""
    if abs(hub_height - WIND_HEIGHT_LOW) <= SNAP_THRESHOLD:
        return ws_low.copy()
    if abs(hub_height - WIND_HEIGHT_HIGH) <= SNAP_THRESHOLD:
        return ws_high.copy()

    log_hub = math.log(hub_height / Z0)
    log_low = math.log(WIND_HEIGHT_LOW / Z0)
    log_high = math.log(WIND_HEIGHT_HIGH / Z0)

    weight = (log_hub - log_low) / (log_high - log_low)
    return ws_low + weight * (ws_high - ws_low)


# ============================================================
# Tensor assembly
# ============================================================

def assemble_site_tensors(
    ieso_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    encoder_start: pd.Timestamp,
    encoder_end: pd.Timestamp,
    decoder_start: pd.Timestamp,
    decoder_end: pd.Timestamp,
    hub_height: float,
    capacity_mw: float,
    site_id: int,
    generator_id: str = "",
) -> dict[str, torch.Tensor] | None:
    """Assemble encoder/decoder/static tensors for one site.

    The encoder/decoder windows are passed in, determined by IESO
    data availability in main().

    Returns dict with keys: encoder_input, decoder_input, static,
        decoder_datetimes, encoder_end
    or None if data is incomplete.
    """
    tag = f"[{generator_id}]" if generator_id else ""

    # Filter weather to encoder and decoder windows
    weather_enc = weather_df[
        (weather_df["datetime"] >= encoder_start)
        & (weather_df["datetime"] <= encoder_end)
    ].copy()
    weather_dec = weather_df[
        (weather_df["datetime"] >= decoder_start)
        & (weather_df["datetime"] <= decoder_end)
    ].copy()

    if len(weather_enc) < ENCODER_STEPS:
        print(f"    {tag} weather encoder rows: {len(weather_enc)}, need {ENCODER_STEPS}")
        print(f"    {tag}   requested range: {encoder_start} to {encoder_end}")
        print(f"    {tag}   weather range: {weather_df['datetime'].min()} to {weather_df['datetime'].max()}")
        return None
    if len(weather_dec) < DECODER_STEPS:
        print(f"    {tag} weather decoder rows: {len(weather_dec)}, need {DECODER_STEPS}")
        print(f"    {tag}   requested range: {decoder_start} to {decoder_end}")
        print(f"    {tag}   weather range: {weather_df['datetime'].min()} to {weather_df['datetime'].max()}")
        return None

    # Take exactly the required number of rows
    weather_enc = weather_enc.tail(ENCODER_STEPS).reset_index(drop=True)
    weather_dec = weather_dec.head(DECODER_STEPS).reset_index(drop=True)

    # Derive hub-height wind speed
    weather_enc["wind_speed_hub"] = compute_wind_speed_hub(
        weather_enc["wind_speed_80m"], weather_enc["wind_speed_120m"], hub_height
    )
    weather_dec["wind_speed_hub"] = compute_wind_speed_hub(
        weather_dec["wind_speed_80m"], weather_dec["wind_speed_120m"], hub_height
    )

    # Filter IESO to encoder window
    site_ieso_enc = ieso_df[
        (ieso_df["datetime"] >= encoder_start)
        & (ieso_df["datetime"] <= encoder_end)
    ].sort_values("datetime")

    # Debug: uncomment to inspect IESO/weather alignment for K2WIND.
    # Useful when the encoder window is short and you suspect a time-mismatch
    # between IESO actuals and the weather snapshot.
    # if generator_id == "K2WIND":
    #     print(f"DEBUG K2WIND IESO last 4 rows before join:")
    #     print(site_ieso_enc.tail(4)[["datetime", "output_mwh"]].to_string())
    #     print(f"DEBUG K2WIND weather encoder last 4 datetimes:")
    #     print(weather_enc.tail(4)[["datetime"]].to_string())

    # Join IESO actuals with encoder weather on datetime
    encoder_data = pd.merge(
        weather_enc[["datetime", "wind_speed_hub", "temperature_2m", "surface_pressure"]],
        site_ieso_enc[["datetime", "output_mwh", "available_capacity_mw"]],
        on="datetime",
        how="inner",
    ).sort_values("datetime")

    # Debug: uncomment to confirm the inner-join kept the expected rows.
    # if generator_id == "K2WIND":
    #     print(f"DEBUG K2WIND encoder_data last 4 rows after join:")
    #     print(encoder_data.tail(4)[["datetime", "output_mwh"]].to_string())

    if len(encoder_data) < ENCODER_STEPS:
        ieso_min = site_ieso_enc["datetime"].min() if len(site_ieso_enc) > 0 else "N/A"
        ieso_max = site_ieso_enc["datetime"].max() if len(site_ieso_enc) > 0 else "N/A"
        print(f"    {tag} encoder join: {len(encoder_data)}/{ENCODER_STEPS} rows matched")
        print(f"    {tag}   encoder window: {encoder_start} to {encoder_end}")
        print(f"    {tag}   IESO range:     {ieso_min} to {ieso_max}")
        return None

    # Convert output_mwh to capacity factor
    encoder_data["output_cf"] = encoder_data["output_mwh"] / capacity_mw

    # Encoder input: (48, 5)
    encoder_input = torch.tensor(
        encoder_data[
            ["output_cf", "available_capacity_mw", "wind_speed_hub",
             "temperature_2m", "surface_pressure"]
        ].values,
        dtype=torch.float32,
    )

    # Decoder input: (24, 3)
    decoder_input = torch.tensor(
        weather_dec[
            ["wind_speed_hub", "temperature_2m", "surface_pressure"]
        ].values,
        dtype=torch.float32,
    )

    # Static: (3,)
    static = torch.tensor(
        [capacity_mw, hub_height, site_id],
        dtype=torch.float32,
    )

    return {
        "encoder_input": encoder_input,
        "decoder_input": decoder_input,
        "static": static,
        "decoder_datetimes": weather_dec["datetime"].tolist(),
        "encoder_end": encoder_end,
    }


# ============================================================
# Inference
# ============================================================

def normalize(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Apply z-score normalization."""
    return (tensor - mean) / std


def run_inference(
    model: WindPowerLSTM,
    norm_stats: dict,
    site_tensors: dict[str, dict],
) -> dict[str, torch.Tensor]:
    """Run batch inference for all sites.

    Returns:
        Dict keyed by generator_id with predicted capacity factor tensor (24,).
    """
    gen_ids = sorted(site_tensors.keys())

    # Stack into batches
    encoder_batch = torch.stack([site_tensors[g]["encoder_input"] for g in gen_ids])
    decoder_batch = torch.stack([site_tensors[g]["decoder_input"] for g in gen_ids])
    static_batch = torch.stack([site_tensors[g]["static"] for g in gen_ids])

    # Normalize using pretrained stats
    encoder_batch = normalize(encoder_batch, norm_stats["encoder_mean"], norm_stats["encoder_std"])
    decoder_batch = normalize(decoder_batch, norm_stats["decoder_mean"], norm_stats["decoder_std"])
    static_batch_norm = static_batch.clone()
    static_batch_norm[:, :2] = normalize(
        static_batch[:, :2], norm_stats["static_mean"], norm_stats["static_std"]
    )

    # Run model
    model.eval()
    with torch.no_grad():
        predictions_norm = model(encoder_batch, decoder_batch, static_batch_norm)

    # Denormalize predictions (capacity factor)
    predictions_cf = (
        predictions_norm * norm_stats["target_std"] + norm_stats["target_mean"]
    )

    # Clamp to valid range
    predictions_cf = predictions_cf.clamp(min=0.0, max=1.0)

    return {gid: predictions_cf[i] for i, gid in enumerate(gen_ids)}


# ============================================================
# Main
# ============================================================

def _load_torch(path: str):
    """Load a torch artifact from a local or gs:// path.

    Routes through storage.read_bytes + BytesIO so the same call works on
    either backend (torch.load reads a file-like object).
    """
    return torch.load(io.BytesIO(storage.read_bytes(path)), weights_only=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run wind power predictions for all sites."
    )
    parser.add_argument(
        "--model-dir", default=storage.models_path("cf"),
        help="Directory with best_model.pt, norm_stats.pt, config.pt "
        "(local or gs://). Defaults to <MODELS_ROOT>/cf",
    )
    parser.add_argument(
        "--ieso-dir", default=storage.data_path("processed", "ieso"),
        help="Preprocessed IESO data directory (local or gs://). "
        "Defaults to <DATA_ROOT>/processed/ieso",
    )
    parser.add_argument(
        "--weather-dir", default=storage.data_path("predictions", "weather"),
        help="Weather forecast directory (local or gs://). "
        "Defaults to <DATA_ROOT>/predictions/weather",
    )
    parser.add_argument(
        "--mapping-csv", default=storage.data_path("mapping.csv"),
        help="Generator mapping CSV (local or gs://). Defaults to <DATA_ROOT>/mapping.csv",
    )
    parser.add_argument(
        "--output-dir", default=storage.data_path("predictions", "lstm"),
        help="Output directory for prediction CSVs (local or gs://). "
        "Defaults to <DATA_ROOT>/predictions/lstm",
    )
    parser.add_argument(
        "--run-timestamp",
        default=None,
        help="YYYYMMDD_HHMM batch identifier shared with fetch_forecast_all. "
        "Defaults to the latest snapshot for today found under --weather-dir.",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.rstrip("/")
    ieso_dir = args.ieso_dir
    weather_dir = args.weather_dir
    mapping_path = args.mapping_csv
    output_dir = args.output_dir.rstrip("/")

    # --- Load model ---
    print("Loading model...")
    config = _load_torch(f"{model_dir}/config.pt")
    norm_stats = _load_torch(f"{model_dir}/norm_stats.pt")

    model = WindPowerLSTM(
        encoder_input_size=config["encoder_input_size"],
        decoder_input_size=config["decoder_input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_sites=config["num_sites"],
        site_embedding_dim=config["site_embedding_dim"],
    )
    model.load_state_dict(_load_torch(f"{model_dir}/best_model.pt"))
    print(f"  Loaded model from {model_dir}")

    # --- Load mapping ---
    mapping = load_mapping(mapping_path)
    site_id_map = build_site_id_map(mapping)

    # --- Load IESO actuals ---
    print("Loading IESO actuals...")
    ieso_df = load_ieso_actuals(ieso_dir)
    n_generators = ieso_df["generator"].nunique()
    print(f"  {n_generators} generators loaded")

    # --- Determine encoder/decoder windows from IESO availability ---
    ieso_last_hour = ieso_df["datetime"].max()
    encoder_end = ieso_last_hour
    encoder_start = encoder_end - pd.Timedelta(hours=ENCODER_STEPS - 1)
    decoder_start = encoder_end + pd.Timedelta(hours=1)
    decoder_end = decoder_start + pd.Timedelta(hours=DECODER_STEPS - 1)

    staleness = datetime.now() - encoder_end.to_pydatetime()
    staleness_hours = staleness.total_seconds() / 3600

    print(f"  IESO last hour:    {encoder_end}")
    print(f"  Encoder window:    {encoder_start} to {encoder_end} ({ENCODER_STEPS}h)")
    print(f"  Decoder window:    {decoder_start} to {decoder_end} ({DECODER_STEPS}h)")
    print(f"  Encoder staleness: {staleness_hours:.1f} hours")

    # --- Resolve run_timestamp ---
    # When called from the orchestrator, --run-timestamp is passed explicitly
    # to pair this prediction with the matching fetch_forecast_all batch.
    # For standalone runs, fall back to the latest snapshot found on disk.
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

    # --- Output filename pairs with the weather batch ---
    output_path = f"{output_dir}/{run_timestamp}.csv"

    # --- Assemble tensors per site ---
    print("Assembling inputs...")
    site_tensors = {}
    skipped = []

    for ieso_name, info in mapping.items():
        gen_id = info["generator_id"]
        site_id = site_id_map.get(gen_id)
        if site_id is None:
            skipped.append((gen_id, "no site_id"))
            continue

        # Load weather snapshot for this site
        weather_df = load_weather_forecast(weather_dir, gen_id, run_timestamp)
        if weather_df is None:
            skipped.append((gen_id, "no weather forecast"))
            continue

        # Get IESO actuals for this generator
        site_ieso = ieso_df[ieso_df["generator"] == ieso_name].copy()
        if len(site_ieso) < ENCODER_STEPS:
            skipped.append((gen_id, f"insufficient IESO data ({len(site_ieso)} hours)"))
            continue

        tensors = assemble_site_tensors(
            ieso_df=site_ieso,
            weather_df=weather_df,
            encoder_start=encoder_start,
            encoder_end=encoder_end,
            decoder_start=decoder_start,
            decoder_end=decoder_end,
            hub_height=info["hub_height"],
            capacity_mw=info["nameplate_capacity"],
            site_id=site_id,
            generator_id=gen_id,
        )

        if tensors is None:
            skipped.append((gen_id, "tensor assembly failed"))
            continue

        site_tensors[gen_id] = tensors

    # Debug: uncomment to dump K2WIND tensor shape and value range. Useful when
    # diagnosing normalization issues or unexpected model inputs.
    # if "K2WIND" in site_tensors:
    #     enc = site_tensors["K2WIND"]["encoder_input"]
    #     print(f"DEBUG K2WIND encoder shape: {enc.shape}")
    #     print(f"DEBUG K2WIND encoder last 4 rows:\n{enc[-4:]}")
    #     print(f"DEBUG K2WIND encoder feature 0 (output_cf) stats: "
    #           f"min={enc[:,0].min():.4f}, max={enc[:,0].max():.4f}, mean={enc[:,0].mean():.4f}")
    #     dec = site_tensors["K2WIND"]["decoder_input"]
    #     print(f"DEBUG K2WIND decoder shape: {dec.shape}")
    #     print(f"DEBUG K2WIND decoder first 4 rows:\n{dec[:4]}")

    print(f"  Ready: {len(site_tensors)} sites")
    if skipped:
        print(f"  Skipped: {len(skipped)} sites")
        for gen_id, reason in skipped:
            print(f"    {gen_id}: {reason}")

    if not site_tensors:
        raise RuntimeError("No sites ready for prediction.")


    # --- Run inference ---
    print("Running inference...")
    predictions = run_inference(model, norm_stats, site_tensors)

    # --- Build output CSV ---
    # Map stripped internal gen_id -> IESO canonical name (spaced) for output.
    # Written value matches IESO's Generator identifier; model/internal keys
    # (site_id embedding, weather filenames) remain stripped via gen_id.
    id_to_name = {info["generator_id"]: name for name, info in mapping.items()}

    rows = []
    for gen_id in sorted(predictions.keys()):
        pred_cf = predictions[gen_id]
        datetimes = site_tensors[gen_id]["decoder_datetimes"]

        # Find capacity for MWh conversion
        capacity_mw = None
        for info in mapping.values():
            if info["generator_id"] == gen_id:
                capacity_mw = info["nameplate_capacity"]
                break

        for hour_idx in range(DECODER_STEPS):
            cf_val = pred_cf[hour_idx].item()
            mwh_val = cf_val * capacity_mw if capacity_mw else None

            rows.append({
                "datetime": datetimes[hour_idx],
                "generator_id": id_to_name[gen_id],
                "predicted_cf": round(cf_val, 4),
                "predicted_mwh": round(mwh_val, 2) if mwh_val is not None else None,
                "encoder_end": encoder_end,
                "staleness_hours": round(staleness_hours, 1),
            })

    output_df = pd.DataFrame(rows)
    storage.write_csv(output_df, output_path)
    print(f"\nPredictions saved to {output_path}")
    print(f"  {len(site_tensors)} sites x {DECODER_STEPS} hours = {len(rows)} rows")
    print(f"  Prediction target: {decoder_start.strftime('%Y-%m-%d %H:%M')} to {decoder_end.strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()

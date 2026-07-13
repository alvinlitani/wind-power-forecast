"""
Build sliding window sequences from features.csv for the seq2seq LSTM.

Constructs encoder/decoder/target tensors with 1-hour stride per generator,
splits by date into train/val/test, and saves as .pt files.

Encoder input (48 timesteps x 5 features):
    output_mwh, available_capacity_mw, wind_speed_hub, temperature_2m, surface_pressure

Decoder input (24 timesteps x 3 features):
    wind_speed_hub, temperature_2m, surface_pressure

Target (24 timesteps):
    output_mwh

Static features (per window):
    capacity_mw, hub_height, site_id

Split boundaries:
    Train:      2023-01-01 to 2024-12-31
    Validation: 2025-01-01 to 2025-12-31
    Test:       2026-01-01 to 2026-04-30

Windows that contain any NaN are skipped.

Usage:
    python build_sequences.py
    python build_sequences.py --input data/processed/features.csv --output-dir data/processed/sequences
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Sequence lengths
ENCODER_LEN = 48
DECODER_LEN = 24
WINDOW_LEN = ENCODER_LEN + DECODER_LEN  # 72 hours total

# Feature definitions
ENCODER_FEATURES = [
    "output_mwh",
    "available_capacity_mw",
    "wind_speed_hub",
    "temperature_2m",
    "surface_pressure",
]

DECODER_FEATURES = [
    "wind_speed_hub",
    "temperature_2m",
    "surface_pressure",
]

TARGET_COL = "output_mwh"

STATIC_FEATURES = [
    "capacity_mw",
    "hub_height",
    "site_id",
]

# Temporal split boundaries (inclusive)
SPLITS = {
    "train": ("2023-01-01", "2024-12-31"),
    "val": ("2025-01-01", "2025-12-31"),
    "test": ("2026-01-01", "2026-04-30"),
}


def build_windows_for_generator(df: pd.DataFrame) -> dict:
    """Build sliding windows from a single generator's time series.

    Args:
        df: DataFrame for one generator, sorted by datetime, with no
            duplicate timestamps.

    Returns:
        Dict with keys: encoder_input, decoder_input, target, static, datetime_start
        Each value is a list of numpy arrays (one per valid window).
    """
    df = df.sort_values("datetime").reset_index(drop=True)

    # Check for duplicate timestamps
    if df["datetime"].duplicated().any():
        gen = df["generator_id"].iloc[0]
        n_dup = df["datetime"].duplicated().sum()
        print(f"  WARNING: {gen} has {n_dup} duplicate timestamps, dropping")
        df = df.drop_duplicates(subset="datetime", keep="first").reset_index(drop=True)

    # Columns needed for NaN checking
    all_feature_cols = list(set(ENCODER_FEATURES + DECODER_FEATURES))

    # Pre-extract numpy arrays for speed
    encoder_data = df[ENCODER_FEATURES].values
    decoder_data = df[DECODER_FEATURES].values
    target_data = df[TARGET_COL].values
    static_data = df[STATIC_FEATURES].iloc[0].values  # Same for all rows
    datetimes = df["datetime"].values

    # Check which rows have NaN in any feature or target
    nan_mask = df[all_feature_cols + [TARGET_COL]].isna().any(axis=1).values

    windows = {
        "encoder_input": [],
        "decoder_input": [],
        "target": [],
        "static": [],
        "datetime_start": [],
    }

    n_rows = len(df)
    for i in range(n_rows - WINDOW_LEN + 1):
        # Skip if any NaN in the window
        if nan_mask[i : i + WINDOW_LEN].any():
            continue

        enc = encoder_data[i : i + ENCODER_LEN]
        dec = decoder_data[i + ENCODER_LEN : i + WINDOW_LEN]
        tgt = target_data[i + ENCODER_LEN : i + WINDOW_LEN]

        windows["encoder_input"].append(enc)
        windows["decoder_input"].append(dec)
        windows["target"].append(tgt)
        windows["static"].append(static_data)
        windows["datetime_start"].append(datetimes[i])

    return windows


def assign_split(datetime_start: np.datetime64) -> str | None:
    """Determine which split a window belongs to based on encoder start time.

    Returns split name ('train', 'val', 'test') or None if outside all ranges.
    """
    ts = pd.Timestamp(datetime_start)
    for split_name, (start, end) in SPLITS.items():
        if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
            return split_name
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Build sequence tensors from features.csv."
    )
    parser.add_argument(
        "--input",
        default="data/processed/features.csv",
        help="Input features CSV (default: data/processed/features.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/sequences",
        help="Output directory for .pt files (default: data/processed/sequences)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load features ---
    print(f"Loading features from {input_path}...")
    df = pd.read_csv(input_path, parse_dates=["datetime"])
    print(f"  {len(df)} rows, {df['generator_id'].nunique()} generators")

    # --- Build windows per generator ---
    print("Building sliding windows...")

    split_data = {
        split: {
            "encoder_input": [],
            "decoder_input": [],
            "target": [],
            "static": [],
        }
        for split in SPLITS
    }

    n_skipped = 0
    n_total = 0

    for gen_id, gen_df in df.groupby("generator_id"):
        windows = build_windows_for_generator(gen_df)
        n_windows = len(windows["encoder_input"])

        for j in range(n_windows):
            n_total += 1
            split = assign_split(windows["datetime_start"][j])
            if split is None:
                n_skipped += 1
                continue

            split_data[split]["encoder_input"].append(windows["encoder_input"][j])
            split_data[split]["decoder_input"].append(windows["decoder_input"][j])
            split_data[split]["target"].append(windows["target"][j])
            split_data[split]["static"].append(windows["static"][j])

    # --- Convert to tensors and save ---
    print(f"\nTotal windows built: {n_total}")
    print(f"Windows outside split ranges: {n_skipped}")

    for split_name, data in split_data.items():
        n = len(data["encoder_input"])
        if n == 0:
            print(f"  {split_name}: 0 windows (skipping)")
            continue

        tensors = {
            "encoder_input": torch.tensor(
                np.array(data["encoder_input"]), dtype=torch.float32
            ),
            "decoder_input": torch.tensor(
                np.array(data["decoder_input"]), dtype=torch.float32
            ),
            "target": torch.tensor(
                np.array(data["target"]), dtype=torch.float32
            ),
            "static": torch.tensor(
                np.array(data["static"]), dtype=torch.float32
            ),
        }

        out_path = output_dir / f"{split_name}.pt"
        torch.save(tensors, out_path)

        print(f"  {split_name}: {n} windows")
        print(f"    encoder_input: {tensors['encoder_input'].shape}")
        print(f"    decoder_input: {tensors['decoder_input'].shape}")
        print(f"    target:        {tensors['target'].shape}")
        print(f"    static:        {tensors['static'].shape}")
        print(f"    -> {out_path}")


if __name__ == "__main__":
    main()
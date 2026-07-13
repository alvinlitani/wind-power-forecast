"""
Evaluate the trained wind power LSTM model.

Loads the best model checkpoint, runs inference on a specified split,
and reports detailed metrics: overall MAE, per-generator MAE (absolute
and as % of nameplate capacity), and per-hour-ahead MAE.

Auto-detects whether the model was trained on raw MWh or capacity factor
based on the norm_stats file.

Usage:
    python evaluate.py
    python evaluate.py --split val --model-dir models
    python evaluate.py --split val --model-dir models_cf
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

# Add train directory to path for model import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from model import WindPowerLSTM

def normalize(
    tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Apply z-score normalization."""
    return (tensor - mean) / std


def denormalize(tensor: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Reverse z-score normalization for a single feature."""
    return tensor * std + mean


def load_mapping(mapping_path: Path) -> dict[int, dict]:
    """Load generator mapping, keyed by site_id (alphabetical order of sanitized names).

    Returns dict: site_id -> {name, nameplate_capacity}
    """
    df = pd.read_csv(mapping_path)
    df["generator_id"] = df["IESO name"].str.replace(" ", "", regex=False)
    df = df.sort_values("generator_id").reset_index(drop=True)

    lookup = {}
    for idx, row in df.iterrows():
        lookup[idx] = {
            "name": row["IESO name"],
            "nameplate_capacity": row["Nameplate Capacity"],
        }
    return lookup


def main():
    parser = argparse.ArgumentParser(description="Evaluate wind power LSTM model.")
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="Which split to evaluate (default: val)",
    )
    parser.add_argument(
        "--data-dir",
        default="../../data/processed/sequences",
        help="Directory with .pt split files (default: data/processed/sequences)",
    )
    parser.add_argument(
        "--model-dir",
        default="../models",
        help="Directory with model checkpoint and stats (default: models)",
    )
    parser.add_argument(
        "--mapping",
        default="../../data/mapping.csv",
        help="Generator mapping CSV (default: data/mapping.csv)",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    # --- Load model ---
    config = torch.load(model_dir / "config.pt", weights_only=True)
    norm_stats = torch.load(model_dir / "norm_stats.pt", weights_only=True)

    # Detect model type
    is_cf = norm_stats.get("target_is_capacity_factor", torch.tensor(False)).item()
    model_type = "capacity factor" if is_cf else "MWh"
    print(f"Model type: {model_type}")

    model = WindPowerLSTM(**config).to(device)
    model.load_state_dict(
        torch.load(model_dir / "best_model.pt", map_location=device, weights_only=True)
    )
    model.eval()

    # --- Load data ---
    split_path = data_dir / f"{args.split}.pt"
    if not split_path.exists():
        print(f"Split file not found: {split_path}")
        sys.exit(1)

    data = torch.load(split_path, weights_only=True)
    print(f"Evaluating on {args.split} split: {data['encoder_input'].shape[0]} windows")

    # --- Prepare inputs ---
    # Original MWh target and capacity for denormalization
    target_mwh = data["target"]  # (N, 24) in MWh
    capacity = data["static"][:, 0]  # (N,) capacity_mw
    site_ids = data["static"][:, 2].long()

    # For capacity factor models, convert encoder output feature to CF
    encoder_input = data["encoder_input"].clone()
    if is_cf:
        encoder_input[:, :, 0] = encoder_input[:, :, 0] / capacity.unsqueeze(1)

    # Normalize inputs
    encoder_input = normalize(
        encoder_input, norm_stats["encoder_mean"], norm_stats["encoder_std"]
    )
    decoder_input = normalize(
        data["decoder_input"], norm_stats["decoder_mean"], norm_stats["decoder_std"]
    )
    static = data["static"].clone()
    static[:, :2] = normalize(static[:, :2], norm_stats["static_mean"], norm_stats["static_std"])

    # --- Run inference in batches ---
    print("Running inference...")
    batch_size = 512
    n_samples = encoder_input.shape[0]
    pred_chunks = []

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            pred_normalized = model(
                encoder_input[start:end].to(device),
                decoder_input[start:end].to(device),
                static[start:end].to(device),
            )
            pred_chunks.append(pred_normalized.cpu())

    pred_all_normalized = torch.cat(pred_chunks, dim=0)

    # Denormalize predictions
    target_mean = norm_stats["target_mean"].item()
    target_std = norm_stats["target_std"].item()

    if is_cf:
        # Predictions are in capacity factor space -> denorm to CF -> multiply by capacity to get MWh
        pred_cf = denormalize(pred_all_normalized, target_mean, target_std)
        pred_mwh = pred_cf * capacity.unsqueeze(1)
    else:
        # Predictions are in MWh space -> denorm directly
        pred_mwh = denormalize(pred_all_normalized, target_mean, target_std)

    # --- Overall MAE ---
    overall_mae = nn.L1Loss()(pred_mwh, target_mwh).item()
    print(f"\nOverall MAE: {overall_mae:.2f} MWh")

    # --- Naive baseline: repeat last encoder hour for all 24 decoder hours ---
    last_encoder_output = data["encoder_input"][:, -1, 0]  # (N,) in MWh
    naive_pred = last_encoder_output.unsqueeze(1).expand(-1, 24)
    naive_mae = nn.L1Loss()(naive_pred, target_mwh).item()
    print(f"Naive baseline MAE (repeat last hour): {naive_mae:.2f} MWh")

    improvement = (1 - overall_mae / naive_mae) * 100
    print(f"Model improvement over baseline: {improvement:.1f}%")

    # --- Per-generator MAE ---
    gen_lookup = load_mapping(Path(args.mapping))
    unique_sites = site_ids.unique().sort().values

    print(f"\n{'Generator':<25} {'MAE (MWh)':>10} {'Capacity':>10} {'MAE %':>8}")
    print("-" * 55)

    gen_results = []
    for site_id in unique_sites:
        sid = site_id.item()
        mask = site_ids == site_id
        gen_pred = pred_mwh[mask]
        gen_target = target_mwh[mask]
        gen_mae = nn.L1Loss()(gen_pred, gen_target).item()

        info = gen_lookup.get(sid, {"name": f"Site {sid}", "nameplate_capacity": 0})
        cap = info["nameplate_capacity"]
        mae_pct = (gen_mae / cap * 100) if cap > 0 else float("nan")

        gen_results.append({
            "name": info["name"],
            "mae_mwh": gen_mae,
            "capacity": cap,
            "mae_pct": mae_pct,
        })

        print(f"{info['name']:<25} {gen_mae:>10.2f} {cap:>10.1f} {mae_pct:>7.1f}%")

    # Sort by MAE % to highlight worst performers
    gen_results.sort(key=lambda x: x["mae_pct"], reverse=True)
    print(f"\n{'Top 5 worst (by MAE %)':<25}")
    print("-" * 55)
    for r in gen_results[:5]:
        print(f"{r['name']:<25} {r['mae_mwh']:>10.2f} {r['capacity']:>10.1f} {r['mae_pct']:>7.1f}%")

    print(f"\n{'Top 5 best (by MAE %)':<25}")
    print("-" * 55)
    for r in gen_results[-5:]:
        print(f"{r['name']:<25} {r['mae_mwh']:>10.2f} {r['capacity']:>10.1f} {r['mae_pct']:>7.1f}%")

    # --- Per-hour-ahead MAE ---
    print(f"\n{'Hour Ahead':>10} {'MAE (MWh)':>10}")
    print("-" * 22)
    for h in range(24):
        hour_mae = nn.L1Loss()(pred_mwh[:, h], target_mwh[:, h]).item()
        print(f"{h + 1:>10} {hour_mae:>10.2f}")


if __name__ == "__main__":
    main()

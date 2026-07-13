"""
Train the seq2seq LSTM model using capacity factor as target.

Same as train.py but converts the target from raw MWh to capacity factor
(output_mwh / capacity_mw) before training. This makes the target
scale-invariant across generators of different sizes.

The encoder input's output_mwh feature (index 0) is also converted to
capacity factor for consistency.

Usage:
    python train_cf.py
    python train_cf.py --output-dir models_cf
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import WindPowerLSTM


def compute_norm_stats(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-feature mean and std from a tensor.

    Args:
        tensor: (N, timesteps, features) or (N, features)

    Returns:
        (mean, std) each of shape (features,)
    """
    if tensor.dim() == 3:
        flat = tensor.reshape(-1, tensor.shape[-1])
    else:
        flat = tensor

    mean = flat.mean(dim=0)
    std = flat.std(dim=0)
    std = std.clamp(min=1e-8)

    return mean, std


def normalize(
    tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Apply z-score normalization."""
    return (tensor - mean) / std


def denormalize(
    tensor: torch.Tensor, mean: float, std: float
) -> torch.Tensor:
    """Reverse z-score normalization for a single feature."""
    return tensor * std + mean


def load_split(data_dir: Path, split: str) -> dict[str, torch.Tensor]:
    """Load a .pt split file."""
    path = data_dir / f"{split}.pt"
    if not path.exists():
        print(f"Split file not found: {path}")
        sys.exit(1)
    return torch.load(path, weights_only=True)


def convert_to_capacity_factor(data: dict[str, torch.Tensor]) -> None:
    """Convert output_mwh to capacity factor in-place.

    Divides:
        - target (N, 24) by capacity_mw
        - encoder_input feature at index 0 (output_mwh) by capacity_mw

    capacity_mw is at static[:, 0].
    """
    capacity = data["static"][:, 0]  # (N,)

    # Target: (N, 24) / (N, 1)
    data["target"] = data["target"] / capacity.unsqueeze(1)

    # Encoder input feature 0 (output_mwh): (N, 48, 5) -> divide only index 0
    data["encoder_input"][:, :, 0] = (
        data["encoder_input"][:, :, 0] / capacity.unsqueeze(1)
    )


def make_dataloader(
    data: dict[str, torch.Tensor], batch_size: int, shuffle: bool
) -> DataLoader:
    """Create a DataLoader from a split dict."""
    dataset = TensorDataset(
        data["encoder_input"],
        data["decoder_input"],
        data["target"],
        data["static"],
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for enc, dec, tgt, static in dataloader:
        enc = enc.to(device)
        dec = dec.to(device)
        tgt = tgt.to(device)
        static = static.to(device)

        optimizer.zero_grad()
        pred = model(enc, dec, static)
        loss = criterion(pred, tgt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate on a dataset. Returns average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for enc, dec, tgt, static in dataloader:
        enc = enc.to(device)
        dec = dec.to(device)
        tgt = tgt.to(device)
        static = static.to(device)

        pred = model(enc, dec, static)
        loss = criterion(pred, tgt)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(
        description="Train wind power LSTM model with capacity factor target."
    )
    parser.add_argument(
        "--data-dir",
        default="../../data/processed/sequences",
        help="Directory with train.pt, val.pt (default: data/processed/sequences)",
    )
    parser.add_argument(
        "--output-dir",
        default="../models_cf",
        help="Directory to save model and stats (default: models_cf)",
    )
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-sites", type=int, default=45)
    parser.add_argument("--site-embedding-dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--stride",
        type=int,
        default=6,
        help="Subsample training windows every N steps to reduce overlap (default: 6)",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print("Loading data...")
    train_data = load_split(data_dir, "train")
    val_data = load_split(data_dir, "val")

    print(f"  Train: {train_data['encoder_input'].shape[0]} windows")
    print(f"  Val:   {val_data['encoder_input'].shape[0]} windows")

    # --- Convert to capacity factor ---
    print("Converting target and encoder output to capacity factor...")
    convert_to_capacity_factor(train_data)
    convert_to_capacity_factor(val_data)

    # --- Compute normalization stats from full training set (before subsampling) ---
    print("Computing normalization stats...")

    encoder_mean, encoder_std = compute_norm_stats(train_data["encoder_input"])
    decoder_mean, decoder_std = compute_norm_stats(train_data["decoder_input"])
    target_mean, target_std = compute_norm_stats(
        train_data["target"].unsqueeze(-1)
    )
    target_mean = target_mean.squeeze()
    target_std = target_std.squeeze()

    static_numeric = train_data["static"][:, :2]
    static_mean, static_std = compute_norm_stats(static_numeric)

    # --- Subsample training windows by stride to reduce overlap ---
    if args.stride > 1:
        n_full = train_data["encoder_input"].shape[0]
        idx = torch.arange(0, n_full, args.stride)
        for key in train_data:
            train_data[key] = train_data[key][idx]
        print(f"  Train after stride {args.stride}: {train_data['encoder_input'].shape[0]} windows")

    # Save normalization stats for inference
    norm_stats = {
        "encoder_mean": encoder_mean,
        "encoder_std": encoder_std,
        "decoder_mean": decoder_mean,
        "decoder_std": decoder_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "static_mean": static_mean,
        "static_std": static_std,
        "target_is_capacity_factor": torch.tensor(True),
    }

    # --- Normalize data ---
    print("Normalizing data...")

    for split_data in [train_data, val_data]:
        split_data["encoder_input"] = normalize(
            split_data["encoder_input"], encoder_mean, encoder_std
        )
        split_data["decoder_input"] = normalize(
            split_data["decoder_input"], decoder_mean, decoder_std
        )
        split_data["target"] = normalize(
            split_data["target"], target_mean, target_std
        )
        split_data["static"] = split_data["static"].clone()
        split_data["static"][:, :2] = normalize(
            split_data["static"][:, :2], static_mean, static_std
        )

    # --- Create dataloaders ---
    train_loader = make_dataloader(train_data, args.batch_size, shuffle=True)
    val_loader = make_dataloader(val_data, args.batch_size, shuffle=False)

    # --- Initialize model ---
    model = WindPowerLSTM(
        encoder_input_size=train_data["encoder_input"].shape[-1],
        decoder_input_size=train_data["decoder_input"].shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_sites=args.num_sites,
        site_embedding_dim=args.site_embedding_dim,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    # --- Training setup ---
    criterion = nn.L1Loss()  # MAE on capacity factor
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    # --- Training loop ---
    print(f"\nTraining (max {args.max_epochs} epochs, patience {args.patience})...\n")
    print(f"{'Epoch':>5}  {'Train MAE':>10}  {'Val MAE':>10}  {'Time':>6}  {'Status':>10}")
    print("-" * 50)

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            status = "* best"

            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
            status = ""

        print(
            f"{epoch:>5}  {train_loss:>10.4f}  {val_loss:>10.4f}  {elapsed:>5.1f}s  {status:>10}"
        )

        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    # --- Save normalization stats and config ---
    torch.save(norm_stats, output_dir / "norm_stats.pt")

    config = {
        "encoder_input_size": train_data["encoder_input"].shape[-1],
        "decoder_input_size": train_data["decoder_input"].shape[-1],
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "num_sites": args.num_sites,
        "site_embedding_dim": args.site_embedding_dim,
    }
    torch.save(config, output_dir / "config.pt")

    # --- Summary ---
    best_val_cf = best_val_loss * target_std.item()
    print(f"\nTraining complete.")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val MAE (normalized): {best_val_loss:.4f}")
    print(f"  Best val MAE (capacity factor): {best_val_cf:.4f}")
    print(f"  Best val MAE (%): {best_val_cf * 100:.2f}%")
    print(f"\nSaved to {output_dir}/:")
    print(f"  best_model.pt  - model weights")
    print(f"  norm_stats.pt  - normalization statistics")
    print(f"  config.pt      - model configuration")


if __name__ == "__main__":
    main()

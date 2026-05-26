"""
Fine-tune the pretrained seq2seq LSTM model for a specific wind farm site.

Loads the pretrained model weights and normalization stats, filters
training/validation data to a single site, converts to capacity factor,
and continues training with a lower learning rate.

Usage:
    python finetune_cf.py --site-id 24 --site-name K2WIND
    python finetune_cf.py --site-id 24 --site-name K2WIND --lr 0.0001 --max-epochs 50
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import WindPowerLSTM


def normalize(
    tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Apply z-score normalization."""
    return (tensor - mean) / std


def load_split(data_dir: Path, split: str) -> dict[str, torch.Tensor]:
    """Load a .pt split file."""
    path = data_dir / f"{split}.pt"
    if not path.exists():
        print(f"Split file not found: {path}")
        sys.exit(1)
    return torch.load(path, weights_only=True)


def filter_site(data: dict[str, torch.Tensor], site_id: int) -> dict[str, torch.Tensor]:
    """Filter windows to a single site by site_id (static[:, 2])."""
    mask = data["static"][:, 2] == site_id
    n_total = data["static"].shape[0]
    n_site = mask.sum().item()

    if n_site == 0:
        print(f"No windows found for site_id={site_id}")
        sys.exit(1)

    filtered = {key: tensor[mask] for key, tensor in data.items()}
    print(f"  Filtered site_id={site_id}: {n_site}/{n_total} windows")
    return filtered


def convert_to_capacity_factor(data: dict[str, torch.Tensor]) -> None:
    """Convert output_mwh to capacity factor in-place."""
    capacity = data["static"][:, 0]  # (N,)
    data["target"] = data["target"] / capacity.unsqueeze(1)
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
        description="Fine-tune pretrained wind power LSTM for a specific site."
    )
    parser.add_argument(
        "--site-id", type=int, required=True,
        help="Integer site_id to fine-tune on (e.g. 24 for K2WIND)",
    )
    parser.add_argument(
        "--site-name", type=str, required=True,
        help="Site name for output directory naming (e.g. K2WIND)",
    )
    parser.add_argument(
        "--data-dir",
        default="../../data/processed/sequences",
        help="Directory with train.pt, val.pt (default: data/processed/sequences)",
    )
    parser.add_argument(
        "--pretrained-dir",
        default="../models_cf",
        help="Directory with pretrained best_model.pt, norm_stats.pt, config.pt",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: {pretrained-dir}/finetuned/{site_name})",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Subsample training windows every N steps (default: 1, no subsampling)",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    pretrained_dir = Path(args.pretrained_dir)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = pretrained_dir / "finetuned" / args.site_name.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load pretrained model config and norm stats ---
    print("Loading pretrained model...")
    config = torch.load(pretrained_dir / "config.pt", weights_only=True)
    norm_stats = torch.load(pretrained_dir / "norm_stats.pt", weights_only=True)

    model = WindPowerLSTM(
        encoder_input_size=config["encoder_input_size"],
        decoder_input_size=config["decoder_input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_sites=config["num_sites"],
        site_embedding_dim=config["site_embedding_dim"],
    ).to(device)

    state_dict = torch.load(pretrained_dir / "best_model.pt", weights_only=True)
    model.load_state_dict(state_dict)
    print(f"  Loaded weights from {pretrained_dir / 'best_model.pt'}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    # --- Load and filter data ---
    print(f"\nLoading data for site_id={args.site_id} ({args.site_name})...")
    train_data = load_split(data_dir, "train")
    val_data = load_split(data_dir, "val")

    train_data = filter_site(train_data, args.site_id)
    val_data = filter_site(val_data, args.site_id)

    # --- Convert to capacity factor ---
    print("Converting to capacity factor...")
    convert_to_capacity_factor(train_data)
    convert_to_capacity_factor(val_data)

    # --- Subsample training windows by stride ---
    if args.stride > 1:
        n_full = train_data["encoder_input"].shape[0]
        idx = torch.arange(0, n_full, args.stride)
        for key in train_data:
            train_data[key] = train_data[key][idx]
        print(f"  Train after stride {args.stride}: {train_data['encoder_input'].shape[0]} windows")

    # --- Normalize using pretrained stats ---
    print("Normalizing with pretrained stats...")
    for split_data in [train_data, val_data]:
        split_data["encoder_input"] = normalize(
            split_data["encoder_input"],
            norm_stats["encoder_mean"],
            norm_stats["encoder_std"],
        )
        split_data["decoder_input"] = normalize(
            split_data["decoder_input"],
            norm_stats["decoder_mean"],
            norm_stats["decoder_std"],
        )
        split_data["target"] = normalize(
            split_data["target"],
            norm_stats["target_mean"],
            norm_stats["target_std"],
        )
        split_data["static"] = split_data["static"].clone()
        split_data["static"][:, :2] = normalize(
            split_data["static"][:, :2],
            norm_stats["static_mean"],
            norm_stats["static_std"],
        )

    print(f"  Train: {train_data['encoder_input'].shape[0]} windows")
    print(f"  Val:   {val_data['encoder_input'].shape[0]} windows")

    # --- Create dataloaders ---
    train_loader = make_dataloader(train_data, args.batch_size, shuffle=True)
    val_loader = make_dataloader(val_data, args.batch_size, shuffle=False)

    # --- Evaluate pretrained model before fine-tuning ---
    criterion = nn.L1Loss()
    pretrained_val_loss = evaluate(model, val_loader, criterion, device)
    pretrained_val_cf = pretrained_val_loss * norm_stats["target_std"].item()
    print(f"\nPretrained val MAE (capacity factor): {pretrained_val_cf:.4f} ({pretrained_val_cf * 100:.2f}%)")

    # --- Training setup ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    # --- Training loop ---
    print(f"\nFine-tuning (max {args.max_epochs} epochs, patience {args.patience}, lr {args.lr})...\n")
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

    # --- Summary ---
    best_val_cf = best_val_loss * norm_stats["target_std"].item()

    print(f"\nFine-tuning complete for {args.site_name} (site_id={args.site_id}).")
    print(f"  Pretrained val MAE (capacity factor): {pretrained_val_cf:.4f} ({pretrained_val_cf * 100:.2f}%)")
    print(f"  Fine-tuned val MAE (capacity factor): {best_val_cf:.4f} ({best_val_cf * 100:.2f}%)")
    improvement = pretrained_val_cf - best_val_cf
    print(f"  Improvement: {improvement:.4f} ({improvement * 100:.2f} pp)")
    print(f"  Best epoch: {best_epoch}")
    print(f"\nSaved to {output_dir}/:")
    print(f"  best_model.pt  - fine-tuned model weights")
    print(f"  (norm_stats.pt and config.pt inherited from {pretrained_dir}/)")


if __name__ == "__main__":
    main()

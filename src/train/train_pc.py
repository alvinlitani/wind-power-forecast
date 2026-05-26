"""
Train per-site XGBoost power curve models.

Trains one XGBoost regressor per generator on 4 raw weather features
(wind_speed_80m, wind_speed_120m, temperature_2m, surface_pressure)
to predict output_mwh. Predictions are clipped to [0, nameplate_capacity].

Train period: 2023-2024
Test period:  2025

Reads from:
    - data/processed/features_pc.csv
    - data/mapping.csv

Writes to:
    - models_pc/power_curves.pkl   (dict of {generator_id: trained XGBRegressor})
    - models_pc/test_results.csv   (per-site test metrics)

Usage:
    python train_pc.py
    python train_pc.py --features data/processed/features_pc.csv --mapping data/mapping.csv
"""

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb


FEATURE_COLS = [
    "wind_speed_80m",
    "wind_speed_120m",
    "temperature_2m",
    "surface_pressure",
]
TARGET_COL = "output_mwh"

TRAIN_YEARS = [2023, 2024]
TEST_YEAR = 2025


def load_nameplate_capacity(mapping_path: Path) -> dict[str, float]:
    """Load nameplate capacity per generator from mapping.csv.

    Returns dict keyed by generator_id (spaces removed) -> capacity in MW.
    """
    df = pd.read_csv(mapping_path)
    return {
        row["IESO name"].replace(" ", ""): row["Nameplate Capacity"]
        for _, row in df.iterrows()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train per-site XGBoost power curve models."
    )
    parser.add_argument(
        "--features",
        default="../../data/processed/features_pc.csv",
        help="Features CSV path (default: data/processed/features_pc.csv)",
    )
    parser.add_argument(
        "--mapping",
        default="../../data/mapping.csv",
        help="Generator mapping CSV (default: data/mapping.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="../../models_pc",
        help="Output directory for models (default: models_pc)",
    )
    args = parser.parse_args()

    # --- Load data ---
    print("Loading features...")
    df = pd.read_csv(args.features, parse_dates=["datetime"])
    df["year"] = df["datetime"].dt.year
    print(f"  {len(df)} rows, {df['generator_id'].nunique()} generators")

    nameplate = load_nameplate_capacity(Path(args.mapping))

    # --- Split by year ---
    train_df = df[df["year"].isin(TRAIN_YEARS)].copy()
    test_df = df[df["year"] == TEST_YEAR].copy()
    print(f"  Train: {len(train_df)} rows ({TRAIN_YEARS})")
    print(f"  Test:  {len(test_df)} rows ({TEST_YEAR})")

    # --- Train per-site models ---
    models = {}
    results = []
    generators = sorted(df["generator_id"].unique())

    print(f"\nTraining {len(generators)} models...\n")
    print(f"{'Generator':<30} {'Train MAE':>10} {'Test MAE':>10} {'Nameplate':>10} {'Test MAE%':>10}")
    print("-" * 75)

    for gen_id in generators:
        cap = nameplate.get(gen_id)
        if cap is None:
            print(f"WARNING: No nameplate capacity for {gen_id}, skipping.")
            continue

        gen_train = train_df[train_df["generator_id"] == gen_id]
        gen_test = test_df[test_df["generator_id"] == gen_id]

        if len(gen_train) == 0:
            print(f"WARNING: No training data for {gen_id}, skipping.")
            continue

        X_train = gen_train[FEATURE_COLS]
        y_train = gen_train[TARGET_COL]
        X_test = gen_test[FEATURE_COLS]
        y_test = gen_test[TARGET_COL]

        # Train XGBoost with defaults
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
        )
        model.fit(X_train, y_train, verbose=False)

        # Predict and clip to [0, nameplate]
        train_pred = model.predict(X_train).clip(0, cap)
        train_mae = (y_train - train_pred).abs().mean()

        test_mae = float("nan")
        test_mae_pct = float("nan")
        if len(gen_test) > 0:
            test_pred = model.predict(X_test).clip(0, cap)
            test_mae = (y_test - test_pred).abs().mean()
            test_mae_pct = test_mae / cap * 100

        models[gen_id] = model

        print(f"{gen_id:<30} {train_mae:>10.2f} {test_mae:>10.2f} {cap:>10.2f} {test_mae_pct:>9.2f}%")

        results.append({
            "generator_id": gen_id,
            "train_rows": len(gen_train),
            "test_rows": len(gen_test),
            "nameplate_mw": cap,
            "train_mae": round(train_mae, 3),
            "test_mae": round(test_mae, 3),
            "test_mae_pct": round(test_mae_pct, 2),
        })

    # --- Summary ---
    results_df = pd.DataFrame(results)
    avg_test_mae_pct = results_df["test_mae_pct"].mean()
    print("-" * 75)
    print(f"{'Average':<30} {'':>10} {'':>10} {'':>10} {avg_test_mae_pct:>9.2f}%")

    # --- Save ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "power_curves.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(models, f)
    print(f"\nSaved {len(models)} models to {model_path}")

    results_path = output_dir / "test_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved test results to {results_path}")


if __name__ == "__main__":
    main()

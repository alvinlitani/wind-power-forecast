"""
Hyperparameter tuning for XGBoost power curve models.

Finds one shared set of best hyperparameters across all 45 sites using
time-based cross-validation (2023 train / 2024 val). Selects the params
that minimize average validation MAE% across all sites, then retrains
all models on 2023-2024 with those params and evaluates on 2025.

Reads from:
    - data/processed/features_pc.csv
    - data/mapping.csv

Writes to:
    - models_pc/tuning_results.csv    (per-site metrics with tuned params)
    - models_pc/power_curves.pkl      (retrained models with best shared params)

Usage:
    python tune_pc.py
    python tune_pc.py --features data/processed/features_pc.csv --mapping data/mapping.csv
"""

import argparse
import itertools
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

# Grid search space
PARAM_GRID = {
    "n_estimators": [100, 200, 500],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1],
}


def load_nameplate_capacity(mapping_path: Path) -> dict[str, float]:
    df = pd.read_csv(mapping_path)
    return {
        row["IESO name"].replace(" ", ""): row["Nameplate Capacity"]
        for _, row in df.iterrows()
    }


def grid_combinations(param_grid: dict) -> list[dict]:
    """Generate all combinations from a parameter grid."""
    keys = param_grid.keys()
    values = param_grid.values()
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def main():
    parser = argparse.ArgumentParser(
        description="Tune XGBoost power curve models (shared hyperparameters)."
    )
    parser.add_argument(
        "--features",
        default="../../data/processed/features_pc.csv",
        help="Features CSV path",
    )
    parser.add_argument(
        "--mapping",
        default="../../data/mapping.csv",
        help="Generator mapping CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="../../models_pc",
        help="Output directory",
    )
    args = parser.parse_args()

    print("Loading features...")
    df = pd.read_csv(args.features, parse_dates=["datetime"])
    df["year"] = df["datetime"].dt.year

    nameplate = load_nameplate_capacity(Path(args.mapping))

    # Time-based CV: train on 2023, validate on 2024, test on 2025
    train_df = df[df["year"] == 2023].copy()
    val_df = df[df["year"] == 2024].copy()
    test_df = df[df["year"] == 2025].copy()
    full_train_df = df[df["year"].isin([2023, 2024])].copy()

    generators = sorted(df["generator_id"].unique())
    combos = grid_combinations(PARAM_GRID)
    print(f"Grid: {len(combos)} param combinations × {len(generators)} sites = {len(combos) * len(generators)} fits\n")

    # =========================================================
    # Phase 1: Find best shared params (train 2023, val 2024)
    # =========================================================
    print("Phase 1: Grid search for best shared hyperparameters...")
    print(f"  Training on 2023, validating on 2024\n")

    # Pre-split data per site for efficiency
    site_data = {}
    for gen_id in generators:
        cap = nameplate.get(gen_id)
        if cap is None:
            continue
        gt = train_df[train_df["generator_id"] == gen_id]
        gv = val_df[val_df["generator_id"] == gen_id]
        if len(gt) == 0 or len(gv) == 0:
            continue
        site_data[gen_id] = {
            "cap": cap,
            "X_train": gt[FEATURE_COLS],
            "y_train": gt[TARGET_COL],
            "X_val": gv[FEATURE_COLS],
            "y_val": gv[TARGET_COL],
        }

    # Evaluate each param combo across all sites
    combo_scores = []
    for i, params in enumerate(combos):
        site_val_pcts = []
        for gen_id, sd in site_data.items():
            model = xgb.XGBRegressor(random_state=42, **params)
            model.fit(sd["X_train"], sd["y_train"], verbose=False)
            pred = model.predict(sd["X_val"]).clip(0, sd["cap"])
            val_mae = (sd["y_val"] - pred).abs().mean()
            val_pct = val_mae / sd["cap"] * 100
            site_val_pcts.append(val_pct)

        avg_val_pct = sum(site_val_pcts) / len(site_val_pcts)
        combo_scores.append((avg_val_pct, params))
        params_str = f"n={params['n_estimators']} d={params['max_depth']} lr={params['learning_rate']}"
        print(f"  [{i+1:>2}/{len(combos)}] {params_str:<35} avg val MAE%: {avg_val_pct:.2f}%")

    # Sort by validation score
    combo_scores.sort(key=lambda x: x[0])
    best_val_pct, best_params = combo_scores[0]

    print(f"\n  Best params: n_estimators={best_params['n_estimators']}, "
          f"max_depth={best_params['max_depth']}, "
          f"learning_rate={best_params['learning_rate']}")
    print(f"  Best avg val MAE%: {best_val_pct:.2f}%")

    # Show top 5 for context
    print(f"\n  Top 5 combinations:")
    for rank, (score, params) in enumerate(combo_scores[:5], 1):
        params_str = f"n={params['n_estimators']} d={params['max_depth']} lr={params['learning_rate']}"
        print(f"    {rank}. {params_str:<35} {score:.2f}%")

    # =========================================================
    # Phase 2: Retrain all sites with best params on 2023-2024,
    #           evaluate on 2025, compare to defaults
    # =========================================================
    print(f"\n\nPhase 2: Retraining all sites with best params, evaluating on 2025...")
    print(f"\n{'Generator':<30} {'Default MAE%':>13} {'Tuned MAE%':>12} {'Δ':>8}")
    print("-" * 67)

    models = {}
    results = []

    for gen_id in generators:
        cap = nameplate.get(gen_id)
        if cap is None:
            continue

        gen_full = full_train_df[full_train_df["generator_id"] == gen_id]
        gen_test = test_df[test_df["generator_id"] == gen_id]

        if len(gen_full) == 0:
            continue

        X_full = gen_full[FEATURE_COLS]
        y_full = gen_full[TARGET_COL]
        X_test = gen_test[FEATURE_COLS]
        y_test = gen_test[TARGET_COL]

        # Default model
        default_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        default_model.fit(X_full, y_full, verbose=False)
        default_pred = default_model.predict(X_test).clip(0, cap)
        default_mae = (y_test - default_pred).abs().mean()
        default_pct = default_mae / cap * 100

        # Tuned model
        tuned_model = xgb.XGBRegressor(random_state=42, **best_params)
        tuned_model.fit(X_full, y_full, verbose=False)
        tuned_pred = tuned_model.predict(X_test).clip(0, cap)
        tuned_mae = (y_test - tuned_pred).abs().mean()
        tuned_pct = tuned_mae / cap * 100

        delta = tuned_pct - default_pct
        models[gen_id] = tuned_model

        print(f"{gen_id:<30} {default_pct:>12.2f}% {tuned_pct:>11.2f}% {delta:>+7.2f}%")

        results.append({
            "generator_id": gen_id,
            "nameplate_mw": cap,
            "default_test_mae_pct": round(default_pct, 2),
            "tuned_test_mae_pct": round(tuned_pct, 2),
            "delta_pct": round(delta, 2),
        })

    # --- Summary ---
    results_df = pd.DataFrame(results)
    avg_default = results_df["default_test_mae_pct"].mean()
    avg_tuned = results_df["tuned_test_mae_pct"].mean()
    avg_delta = results_df["delta_pct"].mean()
    improved = (results_df["delta_pct"] < 0).sum()
    worsened = (results_df["delta_pct"] > 0).sum()
    unchanged = (results_df["delta_pct"] == 0).sum()

    print("-" * 67)
    print(f"{'Average':<30} {avg_default:>12.2f}% {avg_tuned:>11.2f}% {avg_delta:>+7.2f}%")
    print(f"\nSites improved: {improved}, worsened: {worsened}, unchanged: {unchanged}")
    print(f"\nBest shared params: n_estimators={best_params['n_estimators']}, "
          f"max_depth={best_params['max_depth']}, "
          f"learning_rate={best_params['learning_rate']}")

    # --- Save ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "power_curves.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(models, f)
    print(f"\nSaved {len(models)} tuned models to {model_path}")

    # Also save the best params for reference
    params_path = output_dir / "best_params.txt"
    with open(params_path, "w") as f:
        for k, v in best_params.items():
            f.write(f"{k}={v}\n")
    print(f"Saved best params to {params_path}")

    results_path = output_dir / "tuning_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved tuning results to {results_path}")


if __name__ == "__main__":
    main()

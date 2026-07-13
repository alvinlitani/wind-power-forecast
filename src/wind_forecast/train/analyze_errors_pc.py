"""
Error analysis for XGBoost power curve models.

Analyzes where the model fails: by wind speed regime, season, time of day,
and identifies systematic error patterns per site.

Reads from:
    - data/processed/features_pc.csv
    - data/mapping.csv
    - models_pc/power_curves.pkl

Writes to:
    - models_pc/error_analysis.txt   (printed report)

Usage:
    python analyze_errors_pc.py
"""

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
import numpy as np


FEATURE_COLS = [
    "wind_speed_80m",
    "wind_speed_120m",
    "temperature_2m",
    "surface_pressure",
]
TARGET_COL = "output_mwh"
TEST_YEAR = 2025


def load_nameplate_capacity(mapping_path: Path) -> dict[str, float]:
    df = pd.read_csv(mapping_path)
    return {
        row["IESO name"].replace(" ", ""): row["Nameplate Capacity"]
        for _, row in df.iterrows()
    }


def wind_speed_bin(ws: float) -> str:
    """Bin wind speed into regimes."""
    if ws < 3:
        return "0-3 (below cut-in)"
    elif ws < 6:
        return "3-6 (low)"
    elif ws < 10:
        return "6-10 (mid)"
    elif ws < 15:
        return "10-15 (high)"
    elif ws < 25:
        return "15-25 (rated)"
    else:
        return "25+ (above cut-out)"


def main():
    parser = argparse.ArgumentParser(
        description="Analyze XGBoost power curve model errors."
    )
    parser.add_argument(
        "--features",
        default="data/processed/features_pc.csv",
        help="Features CSV path",
    )
    parser.add_argument(
        "--mapping",
        default="data/mapping.csv",
        help="Generator mapping CSV",
    )
    parser.add_argument(
        "--model-path",
        default="models_pc/power_curves.pkl",
        help="Trained models pickle",
    )
    parser.add_argument(
        "--output",
        default="models_pc/error_analysis.txt",
        help="Output report path",
    )
    args = parser.parse_args()

    # --- Load everything ---
    print("Loading data...")
    df = pd.read_csv(args.features, parse_dates=["datetime"])
    df["year"] = df["datetime"].dt.year
    test_df = df[df["year"] == TEST_YEAR].copy()

    nameplate = load_nameplate_capacity(Path(args.mapping))

    with open(args.model_path, "rb") as f:
        models = pickle.load(f)

    # --- Generate predictions for all test data ---
    all_preds = []

    for gen_id, model in models.items():
        cap = nameplate.get(gen_id, float("inf"))
        gen_test = test_df[test_df["generator_id"] == gen_id].copy()
        if len(gen_test) == 0:
            continue

        gen_test["pred"] = model.predict(gen_test[FEATURE_COLS]).clip(0, cap)
        gen_test["error"] = gen_test["pred"] - gen_test[TARGET_COL]
        gen_test["abs_error"] = gen_test["error"].abs()
        gen_test["nameplate"] = cap
        all_preds.append(gen_test)

    preds = pd.concat(all_preds, ignore_index=True)
    preds["month"] = preds["datetime"].dt.month
    preds["hour"] = preds["datetime"].dt.hour
    preds["season"] = preds["month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall",
    })
    # Use average of 80m and 120m as representative wind speed for binning
    preds["ws_avg"] = (preds["wind_speed_80m"] + preds["wind_speed_120m"]) / 2
    preds["ws_bin"] = preds["ws_avg"].apply(wind_speed_bin)

    lines = []

    def report(text=""):
        lines.append(text)
        print(text)

    report("=" * 80)
    report("XGBoost Power Curve Model — Error Analysis (2025 Test Set)")
    report("=" * 80)

    # --- 1. Overall metrics ---
    report("\n1. OVERALL METRICS")
    report("-" * 40)
    overall_mae = preds["abs_error"].mean()
    overall_rmse = np.sqrt((preds["error"] ** 2).mean())
    overall_bias = preds["error"].mean()
    report(f"   MAE:  {overall_mae:.2f} MWh")
    report(f"   RMSE: {overall_rmse:.2f} MWh")
    report(f"   Bias: {overall_bias:+.2f} MWh (positive = overprediction)")

    # --- 2. Error by wind speed regime ---
    report("\n2. ERROR BY WIND SPEED REGIME")
    report("-" * 70)
    report(f"   {'Wind Speed Bin':<25} {'MAE':>8} {'Bias':>8} {'RMSE':>8} {'Count':>10} {'% of data':>10}")
    report("   " + "-" * 67)

    ws_order = ["0-3 (below cut-in)", "3-6 (low)", "6-10 (mid)",
                "10-15 (high)", "15-25 (rated)", "25+ (above cut-out)"]
    total = len(preds)
    for ws_bin in ws_order:
        subset = preds[preds["ws_bin"] == ws_bin]
        if len(subset) == 0:
            continue
        mae = subset["abs_error"].mean()
        bias = subset["error"].mean()
        rmse = np.sqrt((subset["error"] ** 2).mean())
        pct = len(subset) / total * 100
        report(f"   {ws_bin:<25} {mae:>8.2f} {bias:>+8.2f} {rmse:>8.2f} {len(subset):>10} {pct:>9.1f}%")

    # --- 3. Error by season ---
    report("\n3. ERROR BY SEASON")
    report("-" * 60)
    report(f"   {'Season':<12} {'MAE':>8} {'MAE%':>8} {'Bias':>8} {'Count':>10}")
    report("   " + "-" * 47)

    avg_cap = preds.groupby("generator_id")["nameplate"].first().mean()
    for season in ["Winter", "Spring", "Summer", "Fall"]:
        subset = preds[preds["season"] == season]
        if len(subset) == 0:
            continue
        mae = subset["abs_error"].mean()
        # MAE% computed per-site then averaged
        site_maes = subset.groupby("generator_id").apply(
            lambda g: g["abs_error"].mean() / g["nameplate"].iloc[0] * 100
        )
        mae_pct = site_maes.mean()
        bias = subset["error"].mean()
        report(f"   {season:<12} {mae:>8.2f} {mae_pct:>7.2f}% {bias:>+8.2f} {len(subset):>10}")

    # --- 4. Error by hour of day ---
    report("\n4. ERROR BY HOUR OF DAY")
    report("-" * 50)
    report(f"   {'Hour':<6} {'MAE':>8} {'Bias':>8} {'Count':>10}")
    report("   " + "-" * 33)

    for hour in range(24):
        subset = preds[preds["hour"] == hour]
        if len(subset) == 0:
            continue
        mae = subset["abs_error"].mean()
        bias = subset["error"].mean()
        report(f"   {hour:<6} {mae:>8.2f} {bias:>+8.2f} {len(subset):>10}")

    # --- 5. Error by month ---
    report("\n5. ERROR BY MONTH")
    report("-" * 50)
    report(f"   {'Month':<8} {'MAE':>8} {'Bias':>8} {'Count':>10}")
    report("   " + "-" * 35)

    for month in range(1, 13):
        subset = preds[preds["month"] == month]
        if len(subset) == 0:
            continue
        mae = subset["abs_error"].mean()
        bias = subset["error"].mean()
        report(f"   {month:<8} {mae:>8.2f} {bias:>+8.2f} {len(subset):>10}")

    # --- 6. Worst sites ---
    report("\n6. PER-SITE TEST MAE (sorted worst to best)")
    report("-" * 70)
    report(f"   {'Generator':<30} {'MAE':>8} {'MAE%':>8} {'Bias':>8} {'Nameplate':>10}")
    report("   " + "-" * 67)

    site_stats = preds.groupby("generator_id").agg(
        mae=("abs_error", "mean"),
        bias=("error", "mean"),
        nameplate=("nameplate", "first"),
    )
    site_stats["mae_pct"] = site_stats["mae"] / site_stats["nameplate"] * 100
    site_stats = site_stats.sort_values("mae_pct", ascending=False)

    for gen_id, row in site_stats.iterrows():
        report(f"   {gen_id:<30} {row['mae']:>8.2f} {row['mae_pct']:>7.2f}% {row['bias']:>+8.2f} {row['nameplate']:>10.2f}")

    # --- 7. High-error hours analysis ---
    report("\n7. HIGH-ERROR ANALYSIS (top 1% absolute errors)")
    report("-" * 60)
    threshold = preds["abs_error"].quantile(0.99)
    high_err = preds[preds["abs_error"] >= threshold]
    report(f"   Threshold: {threshold:.2f} MWh ({len(high_err)} hours)")
    report(f"   Mean wind speed (avg 80/120m): {high_err['ws_avg'].mean():.1f} m/s")
    report(f"   Mean temperature: {high_err['temperature_2m'].mean():.1f} °C")

    report(f"\n   Season distribution of high-error hours:")
    season_dist = high_err["season"].value_counts(normalize=True) * 100
    for season in ["Winter", "Spring", "Summer", "Fall"]:
        pct = season_dist.get(season, 0)
        report(f"     {season:<12} {pct:.1f}%")

    report(f"\n   Wind speed bin distribution of high-error hours:")
    ws_dist = high_err["ws_bin"].value_counts(normalize=True) * 100
    for ws_bin in ws_order:
        pct = ws_dist.get(ws_bin, 0)
        report(f"     {ws_bin:<25} {pct:.1f}%")

    report(f"\n   Direction of error:")
    over = (high_err["error"] > 0).sum()
    under = (high_err["error"] < 0).sum()
    report(f"     Overprediction: {over} ({over/len(high_err)*100:.1f}%)")
    report(f"     Underprediction: {under} ({under/len(high_err)*100:.1f}%)")

    # --- 8. Capacity factor analysis ---
    report("\n8. ERROR BY ACTUAL CAPACITY FACTOR")
    report("-" * 60)
    preds["cf_actual"] = preds[TARGET_COL] / preds["nameplate"]
    cf_bins = [(0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0)]
    report(f"   {'CF Range':<15} {'MAE':>8} {'Bias':>8} {'Count':>10} {'% of data':>10}")
    report("   " + "-" * 52)

    for low, high in cf_bins:
        subset = preds[(preds["cf_actual"] >= low) & (preds["cf_actual"] < high)]
        if len(subset) == 0:
            continue
        mae = subset["abs_error"].mean()
        bias = subset["error"].mean()
        pct = len(subset) / total * 100
        report(f"   {low:.0%}-{high:.0%}{'':>9} {mae:>8.2f} {bias:>+8.2f} {len(subset):>10} {pct:>9.1f}%")

    # --- Save report ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()

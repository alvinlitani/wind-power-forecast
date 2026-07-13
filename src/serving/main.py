"""
FastAPI serving app for wind-forecast-ontario (MVP-C).

Serves the latest power-curve (XGBoost) predictions written to GCS by the
prediction flow. Reads predictions per-request from GCS (no in-memory caching
of predictions), with a serve-side completeness guard that falls back to prior
runs and fails closed rather than serving a partial/mid-write file.

Endpoints:
    GET /health               -> liveness, no GCS dependency
    GET /predictions/latest   -> all sites, grouped by site (showpiece)
    GET /predictions/{site}   -> single site horizon
    GET /predictions/ontario  -> fleet MWh aggregate by datetime
    GET /docs                 -> FastAPI auto Swagger (demo surface)

Roster: trusted from the upstream fetch/predict gates for *which* sites are
required. This app re-checks only *completeness* (distinct site count) of the
file it actually reads, to guard against partial reads / mid-write races.
"""

import os
from functools import lru_cache

import gcsfs
import pandas as pd
from fastapi import FastAPI, HTTPException

# --- Config -----------------------------------------------------------------
# Twelve-factor hybrid: overridable via env, defaults to the production bucket.
DATA_ROOT = os.environ.get("DATA_ROOT", "gs://wind-forecast-ontario-data")
PREDICTIONS_PREFIX = f"{DATA_ROOT}/predictions/pc"
MAPPING_PATH = f"{DATA_ROOT}/mapping.csv"

# Sites expected to be legitimately absent (mirror pipeline's EXPECTED_EXCLUSIONS).
# Kept here as an empty set to match current pipeline state; if the pipeline's
# set changes, update here too (duplicated by design, not imported, since this
# is a standalone app).
EXPECTED_EXCLUSIONS: set[str] = set()

# How many runs back to try before failing closed.
MAX_RUNS_BACK = 2

app = FastAPI(
    title="Ontario Wind Power Forecast",
    description="Hourly wind power forecasts across IESO-regulated Ontario wind sites.",
    version="0.1.0",
)

# Single filesystem handle; gcsfs is safe to reuse.
_fs = gcsfs.GCSFileSystem()


# --- Roster (loaded once at startup) ----------------------------------------
@lru_cache(maxsize=1)
def _expected_roster() -> frozenset[str]:
    """Distinct generator_ids expected per run, from mapping.csv minus exclusions.

    Loaded once and cached for the process lifetime; the roster does not change
    mid-run. Raises at first call (startup warmup) if mapping.csv is unreadable
    or lacks generator_id, so misconfiguration fails fast rather than silently.

    Note: mapping.csv keys sites by the 'IESO name' column in spaced canonical
    form (e.g. "PORT BURWELL"), matching the generator_id now written into
    prediction files. Only leading/trailing whitespace is stripped here; internal
    spaces are preserved so the comparison is like-for-like against predictions.
    """
    with _fs.open(MAPPING_PATH, "r") as f:
        mapping = pd.read_csv(f)
    if "IESO name" not in mapping.columns:
        raise RuntimeError(
            f"mapping.csv at {MAPPING_PATH} has no 'IESO name' column; "
            f"found {list(mapping.columns)}"
        )
    roster = set(mapping["IESO name"].astype(str).str.strip().unique())
    roster -= EXPECTED_EXCLUSIONS
    if not roster:
        raise RuntimeError(f"Empty roster derived from {MAPPING_PATH}")
    return frozenset(roster)


# --- Prediction file discovery & loading ------------------------------------
def _list_run_stems() -> list[str]:
    """Filename stems (run timestamps) under the predictions prefix, newest first.

    Stems are 'YYYYMMDD_HHMM'. Lexicographic sort equals chronological sort for
    this fixed-width format, so we sort the strings directly without parsing.
    """
    paths = _fs.glob(f"{PREDICTIONS_PREFIX}/*.csv")
    stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    return sorted(stems, reverse=True)


def _load_complete_run() -> tuple[str, pd.DataFrame]:
    """Return (run_stem, df) for the newest *complete* run within MAX_RUNS_BACK.

    Completeness = distinct generator_id count in the file matches the expected
    roster. A short file (partial read / mid-write / dropped sites that somehow
    bypassed the upstream gate) is rejected and we fall back to the prior run.
    Fails closed with 503 if no complete run is found within the bound.
    """
    roster = _expected_roster()
    stems = _list_run_stems()
    if not stems:
        raise HTTPException(
            status_code=503,
            detail="No prediction runs available.",
        )

    tried = []
    for stem in stems[:MAX_RUNS_BACK]:
        path = f"{PREDICTIONS_PREFIX}/{stem}.csv"
        with _fs.open(path, "r") as f:
            df = pd.read_csv(f)
        sites = set(df["generator_id"].astype(str).str.strip().unique())
        sites -= EXPECTED_EXCLUSIONS
        if sites == roster:
            return stem, df
        missing = sorted(roster - sites)
        tried.append(
            f"{stem}: {len(sites)}/{len(roster)} sites"
            + (f" (missing {missing[:5]}{'...' if len(missing) > 5 else ''})" if missing else "")
        )

    raise HTTPException(
        status_code=503,
        detail=(
            "No complete prediction run within "
            f"{MAX_RUNS_BACK} runs back. Tried: {'; '.join(tried)}"
        ),
    )


# --- Endpoints --------------------------------------------------------------
@app.get("/health")
def health():
    """Liveness check. No GCS dependency by design."""
    return {"status": "healthy"}


@app.get("/predictions/latest")
def predictions_latest():
    """All sites for the latest complete run, grouped by site."""
    stem, df = _load_complete_run()
    sites: dict[str, list[dict]] = {}
    for gid, group in df.groupby("generator_id"):
        sites[str(gid)] = (
            group[["datetime", "predicted_mwh", "predicted_cf"]]
            .sort_values("datetime")
            .to_dict("records")
        )
    return {"run_timestamp": stem, "sites": sites}


@app.get("/predictions/ontario")
def predictions_ontario():
    """Fleet MWh aggregate by datetime for the latest complete run.

    CF is intentionally omitted: per-site capacity factors are not summable and
    a fleet CF would require capacity weighting that isn't meaningful here.
    """
    stem, df = _load_complete_run()
    fleet = (
        df.groupby("datetime")["predicted_mwh"]
        .sum()
        .reset_index()
        .rename(columns={"predicted_mwh": "total_mwh"})
        .sort_values("datetime")
        .to_dict("records")
    )
    return {"run_timestamp": stem, "fleet": fleet}


@app.get("/predictions/{site}")
def predictions_site(site: str):
    """Single site's horizon for the latest complete run.

    The whole-file completeness guard still applies: a corrupt run is rejected
    even if it happens to contain the requested site. Matching normalizes
    leading/trailing whitespace and case, but preserves internal spaces so
    spaced canonical names (e.g. "PORT BURWELL") resolve correctly.
    """
    site_key = site.strip().upper()
    stem, df = _load_complete_run()
    df["_gid"] = df["generator_id"].astype(str).str.strip().str.upper()
    group = df[df["_gid"] == site_key]
    if group.empty:
        known = sorted(df["generator_id"].astype(str).str.strip().unique())
        raise HTTPException(
            status_code=404,
            detail=f"Unknown site '{site}'. Known sites: {known}",
        )
    horizon = (
        group[["datetime", "predicted_mwh", "predicted_cf"]]
        .sort_values("datetime")
        .to_dict("records")
    )
    return {"site": str(group["generator_id"].iloc[0]), "run_timestamp": stem, "horizon": horizon}

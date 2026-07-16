"""Fire-path tests for predict_pc's fail-closed gates.

These prove the gates FIRE on bad data — the half that live green runs can't
verify (clean data never populates the gate accumulators). The last-site /
single-site cases are regression tests for the wiring bug where gate checks
sat at the top of the loop and read accumulators populated at the bottom, so
a bad site processed last was flagged but never checked.

run_batch takes an injected weather_loader, so no storage, model pickle, or
network is touched. Stub models predict a flat value — the gates test data
quality, not prediction values.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from wind_forecast.predict.predict_pc import (
    run_batch,
    FEATURE_COLS,
    FORECAST_HOURS_OUTPUT,
)

# Fixed "now" so weather datetimes are deterministically in the future window.
NOW = datetime(2026, 5, 28, 8, 15)


class StubModel:
    """Minimal stand-in for a trained XGBoost model: predicts a flat value."""
    def predict(self, X):
        return np.full(len(X), 5.0)


def make_weather(n_hours=FORECAST_HOURS_OUTPUT, nan_feature_rows=0):
    """Build a clean future weather frame; optionally NaN the first N rows."""
    base = NOW.replace(minute=0, second=0, microsecond=0)
    times = pd.to_datetime([base + timedelta(hours=h) for h in range(1, n_hours + 1)])
    df = pd.DataFrame({
        "datetime": times,
        "wind_speed_80m": np.full(n_hours, 8.0),
        "wind_speed_120m": np.full(n_hours, 10.0),
        "temperature_2m": np.full(n_hours, 15.0),
        "surface_pressure": np.full(n_hours, 1013.0),
    })
    if nan_feature_rows > 0:
        df.loc[df.index[:nan_feature_rows], "wind_speed_80m"] = np.nan
    return df


def make_mapping(gen_ids):
    return {
        gid: {"ieso_name": gid, "latitude": 43.0, "longitude": -80.0,
              "nameplate_capacity": 100.0}
        for gid in gen_ids
    }


def make_models(gen_ids):
    return {gid: StubModel() for gid in gen_ids}


def loader_from(frames):
    """weather_loader closing over a {gen_id: DataFrame|None} dict."""
    return lambda gen_id: frames.get(gen_id)


def _run(models_ids, mapping_ids, frames):
    return run_batch(
        models=make_models(models_ids),
        mapping=make_mapping(mapping_ids),
        run_timestamp="20260528_0815",
        now_local=NOW,
        code_sha="test",
        weather_loader=loader_from(frames),
    )


# --- Control: clean batch returns rows, stamps provenance ---

def test_clean_batch_returns_rows():
    ids = ["AAA", "BBB", "CCC"]
    frames = {g: make_weather() for g in ids}
    rows = _run(ids, ids, frames)
    assert len(rows) == len(ids) * FORECAST_HOURS_OUTPUT
    assert all(r["code_sha"] == "test" for r in rows)
    assert all(r["run_timestamp"] == "20260528_0815" for r in rows)


# --- NaN gate ---

def test_nan_in_middle_site_raises():
    ids = ["AAA", "BBB", "CCC"]
    frames = {g: make_weather() for g in ids}
    frames["BBB"] = make_weather(nan_feature_rows=1)
    with pytest.raises(RuntimeError, match="Degraded weather"):
        _run(ids, ids, frames)


def test_nan_in_last_site_raises():
    """Regression: last-processed bad site must still trip the gate."""
    ids = ["AAA", "BBB", "CCC"]
    frames = {g: make_weather() for g in ids}
    frames["CCC"] = make_weather(nan_feature_rows=1)   # last in sorted order
    with pytest.raises(RuntimeError, match="Degraded weather"):
        _run(ids, ids, frames)


def test_nan_in_single_site_raises():
    """Regression: a lone site is always 'last', so pre-fix it leaked silently."""
    frames = {"AAA": make_weather(nan_feature_rows=1)}
    with pytest.raises(RuntimeError, match="Degraded weather"):
        _run(["AAA"], ["AAA"], frames)


# --- Short-window gate ---

def test_short_window_in_last_site_raises():
    """Regression: same wiring hole, short-window path."""
    ids = ["AAA", "BBB", "CCC"]
    frames = {g: make_weather() for g in ids}
    frames["CCC"] = make_weather(n_hours=10)   # < FORECAST_HOURS_OUTPUT
    with pytest.raises(RuntimeError, match="Short prediction window"):
        _run(ids, ids, frames)


# --- Roster gate ---

def test_missing_site_raises_roster_gate():
    ids = ["AAA", "BBB", "CCC"]
    frames = {g: make_weather() for g in ids}
    frames["CCC"] = None   # snapshot absent -> produces nothing
    with pytest.raises(RuntimeError, match="Incomplete prediction batch"):
        _run(ids, ids, frames)


def test_model_without_mapping_raises_drift():
    frames = {g: make_weather() for g in ["AAA", "BBB"]}
    with pytest.raises(RuntimeError, match="Model/mapping drift"):
        _run(["AAA", "BBB", "CCC"], ["AAA", "BBB"], frames)   # CCC model, no mapping
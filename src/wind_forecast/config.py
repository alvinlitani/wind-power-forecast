"""
Central configuration for storage roots.

Two roots, each independently switchable between local disk and a GCS bucket
via environment variables. Scripts never hardcode paths — they build them with
`storage.data_path(...)` / `storage.models_path(...)`, which read these roots.

    DATA_ROOT   raw + processed + predictions + evaluations
                local default:  "data"
                cloud:          "gs://wind-power-forecast-data"

    MODELS_ROOT trained model artifacts (LSTM .pt files, XGBoost .pkl)
                local default:  "models"
                cloud:          "gs://wind-power-forecast-models"

Keeping data and models in separate roots lets them live in separate GCS
buckets with separate lifecycle/IAM policies — model artifacts are written
rarely (after training) and read every run, while data churns daily.
"""

from __future__ import annotations

import os

DATA_ROOT = os.environ.get("DATA_ROOT", "data")
MODELS_ROOT = os.environ.get("MODELS_ROOT", "models")

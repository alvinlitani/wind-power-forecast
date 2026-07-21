# Wind Power Forecasting

End-to-end ML pipeline forecasting hourly wind power output for 45 IESO-regulated Ontario wind sites. Two production models (LSTM and per-site XGBoost) serve predictions via a daily flow that ingests weather forecasts, runs inference, and evaluates against IESO actuals.

The pipeline is built for production discipline: cloud-native storage abstraction, scheduled orchestration, infrastructure as code, and reproducible local development. It is deliberately *not* a notebook-only project — the goal is to demonstrate what shipping ML to a renewable-energy operator actually requires.

---

## At a glance

|  |  |
|---|---|
| **Domain** | Wind power generation forecasting for grid operations (Ontario, Canada) |
| **Sites covered** | 45 IESO-regulated wind farms, ~3.5 GW combined nameplate capacity |
| **Forecast horizon** | 24 hours, hourly resolution |
| **Models** | Seq2seq LSTM (multi-site); per-site XGBoost power curve |
| **Refresh** | LSTM 1×/day; XGBoost 4×/day on NWP cadence |
| **Training data** | 2023–2024 (~17,500 hours per site) |
| **Test set** | Jan–Apr 2026 |
| **LSTM test MAE** | ~11.66 MWh aggregate |
| **XGBoost test MAE** | ~11.84% (capacity factor) |
| **Stack** | PyTorch · XGBoost · pandas · Prefect · FastAPI · GCS · Terraform · GitHub Actions · W&B · Grafana |

---

## Why this problem

Ontario operates a competitive wholesale electricity market with a day-ahead bidding window closing around 10:00 ET. Wind generators that can produce accurate next-day forecasts earn better revenue and reduce balancing costs for the grid operator. Forecast errors of 1% across the 3.5 GW Ontario wind fleet translate to roughly 35 MW of unexpected generation per hour — material at grid scale.

Public forecasts exist (IESO publishes its own) but improvement against the IESO baseline is the natural benchmark for any new model. This project's models are evaluated against that baseline daily.

---

## Architecture

The production pipeline runs on GCP free-tier services with an Oracle Cloud ARM VM as the Prefect worker. Storage is split into two buckets (data and models) for independent lifecycle/IAM. The serving layer is decoupled from the prediction pipeline — predictions are pre-computed on a schedule, the API just reads CSVs from GCS.

```
                  ┌──────────────────────┐
                  │   Prefect Cloud      │  Schedules + UI
                  │   (control plane)    │
                  └──────────┬───────────┘
                             │ "run now"
                             ▼
                  ┌──────────────────────┐
                  │   Oracle ARM VM      │  Prefect worker
                  │   (compute plane)    │  Executes flows
                  └──────────┬───────────┘
                             │
                             ▼
       ┌──────────────────────────────────────────┐
       │              GCS (storage)               │
       │  ─────────────────────────────────────   │
       │  wind-power-forecast-data/               │
       │    raw/ieso/, processed/ieso/,           │
       │    predictions/{lstm,pc,weather}/,       │
       │    evaluations/{lstm,pc,baseline}/       │
       │                                          │
       │  wind-power-forecast-models/             │
       │    cf/ (LSTM), pc/ (XGBoost)             │
       └──────────────────────────────────────────┘
                             ▲
                             │ reads
                  ┌──────────┴───────────┐
                  │   Cloud Run          │  FastAPI service
                  │   (serving)          │  /predict, /history
                  └──────────────────────┘
```

### Daily schedule (Eastern Time)

| Time  | Flow(s) | Purpose |
|-------|---------|---------|
| 02:15 | predict (XGBoost only) | Uses 00 UTC NWP run |
| 08:15 | ingest ∥ predict (both models) ∥ evaluate | LSTM here because IESO actuals just published. Predictions ready ~08:35 ET, ahead of 10:00 ET DAM bid deadline. |
| 14:15 | predict (XGBoost only) | Uses 12 UTC NWP run |
| 20:15 | predict (XGBoost only) | Uses 18 UTC NWP run |

The three 08:15 flows fire in parallel — `evaluate_flow` polls for `ingest_flow`'s output rather than waiting on a flow-to-flow dependency, so a delayed IESO publish doesn't block today's predictions.

---

## Modeling choices

### Why two models in production

The LSTM and XGBoost serve different purposes. The LSTM uses 48 hours of past site output (from IESO actuals) as encoder context, which makes it capable of learning site-specific dynamics — but it can only run once a day, after IESO publishes. The XGBoost is a stateless power curve (weather → output), so it can rerun any time fresh weather lands. Both run every weekday morning; the XGBoost also runs at 02:15, 14:15, and 20:15 for intra-day positioning.

Building the LSTM *first*, then layering XGBoost, was deliberate — the LSTM exists to validate that the pipeline is model-agnostic and to provide a benchmark before considering more architectures (Temporal Fusion Transformer is the planned next upgrade).

### Feature inclusion discipline

Features were included only if they carry signal not already captured by existing inputs. Several feature candidates were tested and excluded with explicit justification:

- **Wind direction** — yaw-controlled turbines make raw direction irrelevant; any site-specific directional effects can't generalize across the multi-site model and are captured implicitly during fine-tuning.
- **Site elevation** — effects on output are already captured by NWP wind speed at hub height and by pressure/temperature.
- **Distance to water** — the mechanism (wind speed enhancement near large water bodies) is already an input to the NWP model.
- **Air density** — derivable from temperature and pressure, both already present.
- **Boundary layer height** — variable not available pre-2025 in the Open-Meteo Historical Forecast API.

Final inputs (per site, per hour): `wind_speed_80m`, `wind_speed_120m`, `temperature_2m`, `surface_pressure`, plus static features (capacity, hub height, site embedding) for the LSTM.

### Capacity factor as the target

Both models predict capacity factor (MWh ÷ nameplate capacity) rather than raw MWh. Across 45 sites of vastly different sizes (60 MW to 270 MW) this is essential for the LSTM's multi-site generalization — a single model with a shared output head would otherwise overweight large sites in the loss. Predictions are converted back to MWh at serving time using the per-site nameplate.

### Train on forecast weather, not actuals

Training uses Open-Meteo's Historical Forecast API — i.e., what the forecast *was* for each historical hour — rather than reanalysis (what the weather actually *did*). This matches the conditions the model sees at inference time and avoids the silent domain-shift bug where a model trained on perfect weather catastrophically underperforms when given forecast weather in production.

---

## Pipeline structure

```
wind-power-forecast/
├── src/wind_forecast/
│   ├── config.py        # DATA_ROOT / MODELS_ROOT env-var resolution
│   ├── storage.py       # I/O helper: routes between local disk and gs://
│   ├── model.py         # LSTM network definition (shared by train + predict)
│   ├── ingest/          # IESO download + weather fetch
│   ├── features/        # Feature engineering (training)
│   ├── predict/         # LSTM + XGBoost inference
│   ├── evaluate/        # MAE vs IESO actuals, IESO baseline comparison
│   └── train/           # Training, fine-tuning, hyperparameter tuning
├── flows/
│   ├── tasks.py         # Prefect @task wrappers around package functions
│   ├── ingest_flow.py   # IESO actuals → preprocess
│   ├── predict_flow.py  # Weather fetch → LSTM and/or XGBoost
│   └── evaluate_flow.py # Eval yesterday's predictions vs IESO actuals
├── api/                 # FastAPI service (Cloud Run) — TBD
├── infra/               # Terraform (GCS, IAM, Cloud Run) — TBD
└── pyproject.toml       # Package + extras [pipeline], [api], [dev]
```

The storage helper is the load-bearing piece. Every script reads/writes through `wind_forecast.storage`, which routes `data/...` to local disk and `gs://bucket/...` to GCS based on the path prefix. The same code runs unchanged on a laptop (`DATA_ROOT=data`) and inside the Prefect worker (`DATA_ROOT=gs://wind-power-forecast-data`).

---

## Local development

Requirements: Python 3.10+, CPU is fine (training the LSTM on CPU takes a few hours; inference is fast).

```bash
# Clone and install in editable mode with the pipeline extras
git clone https://github.com/alvinlitani/wind-power-forecast.git
cd wind-power-forecast
pip install -e ".[pipeline]"

# Configure storage roots (defaults point at local ./data and ./models)
cp .env.example .env

# Run a flow once, locally — Prefect runs in ephemeral mode, no cloud needed
python -m flows.predict_flow
python -m flows.ingest_flow
python -m flows.evaluate_flow
```

Individual scripts also still run as plain modules:

```bash
python -m wind_forecast.ingest.fetch_forecast_all
python -m wind_forecast.predict.predict_pc --run-timestamp 20260528_0815
python -m wind_forecast.evaluate.evaluate_daily \
    --prediction data/predictions/lstm/20260528_0815.csv
```

---

## Data sources

- **IESO Generator Output and Capability Report** — hourly per-generator actuals + IESO's own forecast. CSV, published ~06:00 ET daily for the prior month. Used as ground truth and as the baseline benchmark.
- **Open-Meteo Historical Forecast API** — past forecast weather used for training. Mirrors production conditions (forecast, not reanalysis) to avoid distribution shift.
- **Open-Meteo Forecast API** — live forecasts used at inference time. Best-match model selection per coordinate (deterministic — same model used at training and inference for any given site).
- **Wind Turbine Database FGP** — per-site nameplate, hub height, rotor diameter.

---

## Roadmap

- [x] Local pipeline (ingest, features, train, predict, evaluate) for both models
- [x] Storage abstraction (local ↔ GCS)
- [x] Python package + Prefect flows
- [ ] Terraform: GCS buckets, Artifact Registry, Cloud Run, IAM
- [ ] Docker image for the Prefect worker
- [ ] Prefect Cloud + Oracle ARM VM worker
- [ ] FastAPI on Cloud Run serving both models
- [ ] GitHub Actions: test + image build + Cloud Run deploy
- [ ] Grafana dashboard: daily MAE, per-site, LSTM-vs-XGBoost, drift alerts
- [ ] W&B integration for daily metrics logging
- [ ] Temporal Fusion Transformer (next model upgrade)
- [ ] National scope expansion (sites beyond Ontario)

---

## Known limitations

- Training and backfill scripts still use the pre-package import style and local filesystem paths. They run on the development laptop; they will need converting if/when training moves to the cloud.
- The 02:15 ET XGBoost slot runs against potentially-stale weather data (the 00 UTC NWP run may not be fully ingested by Open-Meteo at that hour). The other three slots are fresh.
- Anomaly handling is rule-based (e.g., BOW LAKE excluded for months with mean output exactly zero while available capacity is positive). A statistical anomaly detector is out of scope for v1.

---
# Evaluation metrics — note for README

## Why these metrics

Normalized error metrics (nMAE, nRMSE), normalized by rated capacity, are the
standard reporting metrics in the wind power forecasting literature — the
normalization is what makes error comparable across farms of different sizes.
Because this project predicts capacity factor (output ÷ nameplate), CF-space
errors are already the normalized quantities.

A naive persistence forecast is the conventional reference model, and skill
score (1 − model_error / reference_error) is the recommended accompanying
metric. Skill score is not standalone — it is defined relative to a reference,
which is why persistence is logged alongside it.

Persistence here = actual output at the same hour on the previous day. This is
information-fair: when the 02:00 batch is issued, D-1 actuals are the most
recent data available.

MAPE is deliberately NOT used — it degrades badly at small or zero generation,
and ~21% of site-hours in the IESO data have zero output.

## Forecast horizon labelling

All reported metrics are for **24h-ahead, hourly** forecasts. The literature
stresses stating the horizon explicitly; metrics are meaningless without it.

## Why the IESO GOCR Forecast column is NOT used as a baseline

The Forecast column in the Generator Output and Capability Report is not a
day-ahead product. Measured against the Output column over July 2026 it shows
~1.6% MAE of capacity, with error scaling by ramp size (1.5 MWh MAE on flat
hours vs 4.5 MWh on 20+ MW ramps) and near-zero bias — the profile of a
very-short-lead forecast informed by live telemetry, not a 24h-ahead one. The
archived monthly report also contains no forward-looking rows, so a day-ahead
IESO forecast cannot be recovered from it retroactively.

Comparing a 24h-ahead model against it would be a lead-time mismatch in either
direction. The GOCR remains ground truth via the Output column; only the
Forecast column is unsuitable as a comparator.

IESO's Variable Generation Forecast Summary (48h ahead) would be lead-time
comparable, but publishes provincial/zonal totals rather than per-generator
values — usable only against the Ontario aggregate. Noted as future work.

---
## License

TBD

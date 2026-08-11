# Wind Power Forecasting (Ontario)

Daily hourly forecasts of wind generation output for all 45 IESO-reporting wind generator plants in Ontario. Per-site XGBoost models turns forecasts for the next 24 hours of weather conditions into predicted energy output which is written to cloud storage. Access is provided through a small FastAPI service hosted on Google Cloud Run.

You can test the API at the following endpoints:
- [24-hour per-site output forecast starting at 3 AM](https://wind-forecast-api-654769911920.us-central1.run.app/predictions/latest)
- [24-hour total Ontario output forecast starting at 3 AM](https://wind-forecast-api-654769911920.us-central1.run.app/predictions/ontario)

My goal is to have this project built using production environment practices rather than demo standards. There is scheduling that runs on Prefect Cloud. Accuracy is logged into Weights and Biases (W&B) against a persistence baseline which is the conventional reference for wind forecast skill scores.

## Why this problem

To maintain grid stability, the baseload power must be able to cover energy demand if the renewable energy sites with variable output produce less than expected. In Ontario, baseload power is provided by nuclear, hydro, and gas. According to [IESO data](https://www.ieso.ca/Learn/Ontario-Electricity-Grid/Supply-Mix-and-Generation), the share of total capacity provided by wind is 13% while the combined baseload is about 82%. 

Wind energy output is variable and dependent on weather. However, the output is not all absorbed by the Ontario grid. The IESO issues dispatch instructions meaning its scheduling algorithm sets the expected energy output of each generator sites. The plant report the amount that it can produce in current condition and the IESO schedules the output taking into account the grid condition. 

For renewable energy like solar and wind, dispatch instructions can only runs downwards (curtailment) since you can not instruct the generator sites to produce more if the weather conditions do not allow it while the output can be restricted. The reason for curtailment is that during times of surplus baseload generation, the market already clears and does not need additional capacity provided by the renewable plants.

The IESO already has made variable energy forecasting available to market participants. The [forecast summary](https://reports-public.ieso.ca/public/VGForecastSummary/PUB_VGForecastSummary.xml) is done for the next 48 hours and refreshes hourly. The forecast summary is not granular since it is broken down into electrical zones (not per-site) and not all the zones are covered.  The reason that some zones are not covered may be that there are too few plants are in the region and therefore the summary can be broken down into individual sites' expected output. The per-site forecast in the [IESO Generator Output and Capability Report](https://reports-public.ieso.ca/public/GenOutputCapabilityMonth/) is given out retroactively on the next day.

This project aims to give granular per-site forecast for market participants beforehand for all IESO-covered wind generators.

---

## What is running today

A single Cloud Run Job (`wind-forecast-job` running the `wf-daily` entry point) is triggered daily at 02:00 ET. It fetches weather for 45 sites, runs 45 XGBoost models, and writes predictions as CSV to Google Cloud Storage. Prefect Cloud have the schedule and triggers the job. The job is deliberately made lightweight and usually executes under a minute each run.

A separate Cloud Run service reads the CSVs and serves the results at API (`wind-forecast-api`) endpoints. 

Evaluation runs on my local machine and logs to Weights & Biases. 

## Summarized numbers

| | |
|---|---|
| Sites | 45, totalling 4,943 MW |
| Coverage | ~90% of Ontario's ~5.5 GW of installed wind |
| Site Output | 20 MW to 270 MW |
| Horizon | Rolling 24 hours ahead, hourly |
| Model | Per-site XGBoost power curve, capacity factor target |
| Training | 2023–2024 forecast weather, 17,520 hours per site |
| Validation | None with hyperparameters untuned (see below) |
| Test | Full-year 2025, 394,006 site-hours (~8,756 per site).   |
| Offline nMAE | 11.84% per-site-equal, 11.31% capacity-weighted |
| Live nMAE | 6.2% to 19.5% capacity-weighted (median 8.0%), across 13 full-window batches, 18 Jul – 2 Aug 2026 |
| Stack | XGBoost, pandas, Prefect Cloud, FastAPI, Cloud Run, GCS, W&B, OpenMeteo |

The roster is every site included in the [IESO Generator Output and Capability Report](https://reports-public.ieso.ca/public/GenOutputCapabilityMonth/) which covers market-participant generators of 20 MW or more. Embedded and sub-20 MW wind generators do not appear in the report which explains the ~560 MW gap against the provincial total. The [IESO Active Contracted Generation List](https://www.ieso.ca/-/media/Files/IESO/Document-Library/power-data/supply/IESO-Active-Contracted-Generation-List.xlsx) have 59 wind generator sites that are sub-20 MW with total capacity of around ~533 MW.

I did not perform validation split. A separate validation set exists to choose from candidate models using different hyperparameter settings, feature sets, stopping points. This project is trained on one configuration that is chosen in advance from values in common use: n_estimators=100 and max_depth=6 are XGBoost's defaults, learning_rate=0.1 for conservative learning rate, random_state=42 for reproducibility. The 2025 test set was scored once.

---

## Architecture

Prefect is deliberately thin. It has the schedule and triggers a named Cloud Run Job. The Job itself does not import Prefect nor torch. It is also orchestration-agnostic as it can be triggered by Prefect, by `gcloud', or by anything else that can hit the Cloud Run API. 

Serving is made separate with the API reading pre-computed CSV files and never runs a model.

```
      ┌──────────────────────┐
      │   Prefect Cloud      │  Schedule (02:00 ET) + UI
      │   (orchestrator)     │  
      └──────────┬───────────┘
                 │ triggers
                 v
      ┌──────────────────────┐
      │  Cloud Run Job       │  `wind-forecast-job`
      │  `wf-daily`          │  fetch weather → XGBoost predict
      └──────────┬───────────┘
                 │ writes to
                 v
┌──────────────────────────────────────────┐
│        Google Cloud Storage              │
│                                          │
│  wind-forecast-ontario-data/             │
│    mapping.csv                           │
│    raw/ieso/, processed/ieso/,           │
│    predictions/{pc,lstm,weather}/,       │
│    evaluations/{pc,lstm}/                │
│                                          │
│  wind-forecast-ontario-models/           │
│    pc/ (XGBoost), cf/ (LSTM, offline)    │
└──────────────────────────────────────────┘
                 ^
                 │ reads
      ┌──────────┴───────────┐
      │   Cloud Run Service  │  `wind-forecast-api`
      │                      │  /health
      │                      │  /predictions/latest
      │                      │  /predictions/{site}
      │                      │  /predictions/ontario
      └──────────────────────┘
```

## Why fire-and-forget

The trigger flow fires and does not wait for response. This avoids making the trigger a state machine and keeps the trigger a single atomic API call. The advantages are that no need to tune timeout period and no polling failure leading to re-triggering a job that was still running. The run does not depend on the connection between the orchestrator and the job staying up for its duration. The job currently executes about less than 5 minutes per run and the connection is honestly fine for that short period of time. However, I want to make this architecture more robust by reducing the chance of failure. 

The main tradeoff is that a Prefect run is recorded successful when the trigger is accepted even when the job fails. Runtime failures appear in the Cloud Run's execution history but not in Prefect's history so Prefect does not send alert on them.

## Why a Managed pool for Prefect

There are two ways Prefect can run jobs on Google Cloud Run. A push pool creates a brand new Cloud Run Job for each run and deletes it afterwards. It can not call a job that already exists. I wanted the opposite with one job that is already deployed and can be triggered in multiple ways. Therefore I use a managed pool where my flow code calls the Cloud Run API itself.

## Schedule

| Time (ET) | Job | Contents |
|---|---|---|
| 02:00 | `wf-daily` | Weather fetch → XGBoost predict → write CSV to GCS |

02:00 ET was chosen to clear Open-Meteo's 00Z publish lag (~3–5 hours). Earlier
slots risk being served the previous 18Z cycle.

The prediction window runs 03:00 on the run date through 02:00 the following day,
so it straddles midnight. Because IESO actuals publish with roughly a one-day
lag, a batch becomes fully evaluable about **two days** after it runs.

---

## Modeling choices

### Capacity factor as the target

The model predicts capacity factor (output ÷ nameplate) rather than raw MWh.
Across sites spanning 20 MW to 270 MW — a 13.5× range — this is what makes error
comparable between sites and prevents large sites from dominating any pooled
metric. Predictions are converted back to MWh at serving time using per-site
nameplate.

### Train on forecast weather, not reanalysis

Training uses Open-Meteo's Historical Forecast API — what the forecast *was* for
each historical hour — rather than reanalysis of what the weather actually did.
This matches the conditions the model sees at inference and avoids the silent
domain-shift failure where a model trained on near-perfect weather degrades
sharply when handed real forecasts in production.

### Feature inclusion discipline

Features were included only where they carry signal not already present in
existing inputs. Candidates tested and excluded, with reasons:

- **Wind direction** — yaw-controlled turbines make raw direction largely
  irrelevant to output.
- **Site elevation** — already reflected in NWP wind speed at height and in
  pressure/temperature.
- **Distance to water** — the mechanism (wind enhancement near large water
  bodies) is already an input to the NWP model itself.
- **Air density** — derivable from temperature and pressure, both present.
- **Boundary layer height** — not available pre-2025 in the Open-Meteo
  Historical Forecast API.

Inputs to the live XGBoost path, per site per hour: `wind_speed_80m`,
`wind_speed_120m`, `temperature_2m`, `surface_pressure`.

### Hub-height handling

The XGBoost path deliberately does **not** extrapolate wind speed to hub height.
Each per-site model receives the 80 m and 120 m levels directly and learns the
mapping for its own site, which avoids committing to a fixed shear coefficient
across 45 sites with different terrain.

One caveat: fleet hub heights span 78–132 m, so for the tallest sites the two
levels sit below hub height rather than bracketing it. The model extrapolates
from below in those cases, which is a modelling choice rather than an
interpolation.

The offline LSTM path handles this differently — it interpolates to hub height
using a log wind profile. The two approaches are not interchangeable, and the
feature description above applies to the live XGBoost path only.

---

## Evaluation

### Metrics and why these

Normalized error metrics — nMAE and nRMSE, normalized by rated capacity — are the
standard in wind power forecasting literature; the normalization is what makes
error comparable across farms of different sizes. Because this project predicts
capacity factor, CF-space errors are already the normalized quantities.

A naive **persistence** forecast is the conventional reference model, and skill
score (1 − model error ÷ reference error) is the recommended accompanying metric.
Skill score is not standalone — it is defined relative to a reference, which is
why persistence is logged beside it. Persistence here is actual output at the
same hour on the previous day, which is information-fair: when the 02:00 batch is
issued, D-1 actuals are the most recent data available.

MAPE is deliberately not used. It degrades badly at small or zero generation, and
roughly 21% of site-hours in the IESO data have zero output.

All reported metrics are for **24-hour-ahead, hourly** forecasts. The horizon is
stated explicitly everywhere because normalized metrics are not interpretable
without it.

### Two aggregations, two questions

Error is reported under two weightings because they answer different questions:

- **Capacity-weighted nMAE** — Σ|error| ÷ Σ capacity. The portfolio-normalized
  figure standard in the literature. *How much error per MW installed?*
- **Fleet-aggregate nMAE** — generation summed across sites first, then error
  measured. Over- and under-prediction at different sites cancel. *How much
  error does the portfolio as a whole bear?*

A third weighting — every site-hour counted equally — is retained for per-site
analysis, where it surfaces which sites are weakest. It is not used as a headline
figure.

Fleet-aggregate error is substantially lower than per-site error and the two are
not interchangeable. Both are logged; neither is presented alone.

### Why the IESO GOCR Forecast column is not a baseline

The Generator Output and Capability Report contains a Forecast column, but it is
not a day-ahead product. Measured against the Output column over July 2026 it
shows ~1.6% MAE of capacity with near-zero bias, error that scales with ramp size
(1.5 MWh on flat hours versus 4.5 MWh on 20+ MW hour-over-hour changes), and
occasional catastrophic misses at outage events. That is the profile of a
very-short-lead forecast informed by live telemetry, not a 24-hour-ahead one.

The archived monthly report also contains no forward-looking rows, so a
day-ahead IESO forecast cannot be recovered from it retroactively, and daily
snapshotting would not help — each snapshot only ever adds another completed day.

Comparing a 24-hour-ahead model against it would be a lead-time mismatch in
either direction, so no such comparison is made. The GOCR remains ground truth
via its Output column; only the Forecast column is unsuitable as a comparator.

IESO's Variable Generation Forecast Summary (48 hours ahead) would be
lead-time-comparable, but publishes provincial and zonal totals rather than
per-generator values — usable against the fleet aggregate only. Noted as future
work.

### Offline results — full-year 2025 held-out test

| Metric | Value |
|---|---|
| nMAE, per-site-equal | 11.84% |
| nMAE, capacity-weighted | 11.31% |
| MAE | 12.43 MWh |
| Bias | +0.13 MWh |

Seasonal breakdown shows substantial variation: winter 13.70%, spring 13.81%,
summer 9.16%, fall 10.70%. Any single-season result should be read against the
corresponding season, not the annual figure.

### Live results — July 2026

Scored batches, capacity-weighted unless noted:

| Batch | Per-site-equal | Capacity-wtd | Fleet-aggregate | Fleet bias |
|---|---|---|---|---|
| 2026-07-18 | 17.98% | 19.49% | 6.68% | +3.83% |
| 2026-07-19 | 8.28% | 7.89% | 4.56% | +4.56% |
| 2026-07-20 | 11.79% | 11.21% | 7.01% | −5.40% |

Skill score against persistence has ranged from roughly +0.36 to +0.79 across
scored batches — consistently better than naive, but varying enough that skill
score should not be read as fully normalizing out day difficulty.

**These are not comparable to the offline figures.** The offline result is a
full year across all seasons; these are consecutive summer days in a single
weather regime. Offline summer alone is 9.16%. The live and offline numbers are
computed identically but drawn from different populations, and no agreement
between them is claimed.

### What fleet aggregation reveals

Fleet-aggregate error runs 40–63% below per-site error across scored batches.
More informatively, fleet *bias* accounts for most of the fleet error that
remains — on 2026-07-19 the two were identical to two decimals, meaning the
hourly fleet error never changed sign across 24 hours.

The interpretation: spatial aggregation cancels the site-idiosyncratic component
of error, and what survives is largely common-mode — 45 sites sharing one NWP
source being wrong in the same direction at the same hour. This points at bias
correction as the highest-value next improvement, since better per-site modelling
attacks the component that already cancels.

Caveat: four days, one season, one weather regime.

---

## Known limitations

### Conditional bias: regression to the mean

Aggregate bias is near zero (+0.13 MWh on the 2025 test set), but that figure
masks large opposing biases. Binning by *actual* capacity factor:

| Actual CF | MAE (MWh) | Bias (MWh) | % of hours |
|---|---|---|---|
| 0–5% | 9.94 | **+9.83** | 24.2% |
| 5–20% | 9.49 | +6.16 | 23.0% |
| 20–50% | 13.14 | +0.19 | 24.4% |
| 50–80% | 16.24 | **−10.16** | 15.0% |
| 80–100% | 16.41 | **−16.05** | 13.4% |

The model over-predicts when the fleet is generating weakly and under-predicts
when it is generating hard — classic regression toward the mean. The headline
bias is near zero only because the bins cancel: count-weighting these values
reproduces the reported aggregate to rounding.

The distribution matters. Ontario wind sits below 20% capacity factor 47% of the
time and above 50% only 28% of the time, so the model spends most of its
operating hours in its over-prediction regime.

**This pattern was confirmed live.** On 2026-07-21 — an atypical high-wind day
where 51% of site-hours exceeded 50% CF, against an annual rate of 28% — the same
monotone structure reproduced on data eleven months after the test set: strongly
positive bias in the low-CF bins, negative in the high. Live magnitudes ran
roughly 2–3× the offline ones and the zero-crossing shifted upward, so the
structure generalized while the calibration did not.

Correcting this conditional bias is the clearest available improvement and is not
yet implemented.

### Other limitations

- **Outages and curtailment are invisible to the model.** Inputs are weather
  only, so a site that is offline for non-weather reasons produces large errors
  the model cannot anticipate. One site on 2026-07-21 recorded 52% nMAE against a
  fleet figure of 20%. The GOCR publishes an Available Capacity column that would
  support filtering these cases; this is not yet implemented.
- **Anomaly handling is rule-based.** One site is excluded for months where mean
  output is exactly zero while available capacity is positive. A statistical
  anomaly detector is out of scope for v1.
- **Evaluation is manual.** The scheduled pipeline fetches weather and predicts;
  it does not ingest IESO actuals, so scoring a batch requires running the
  ingest flow by hand first. See [Evaluating a batch](#evaluating-a-batch).
- **Training and backfill scripts predate the package layout.** They use the
  older import style and local filesystem paths, and run on a development machine
  rather than in the cloud.
- **Unit convention.** Both the training and inference fetchers use Open-Meteo's
  default km/h wind speed. Training and inference are therefore consistent and
  the model is unaffected, but the codebase does not state its units explicitly.
  Standardizing on m/s requires a retrain and is deferred to the next model
  upgrade, where a backfill happens anyway.
- **Provenance column is not yet discriminating.** Every prediction row carries a
  `code_sha` column, which falls back to the literal `local` when the environment
  variable is unset. Because CI is not yet wired up, all rows currently read
  `local` — the mechanism is in place and the fallback behaves correctly, but it
  will only distinguish scheduled from manual runs once CI injects the SHA.

---

## Data quality gates

I want to ensure that the runtime fails completely rather than having corrupted data. There are several failure gates:
- Fetch stage: all generator sites must have weather forecasts for the next 24 hours. Recovery is done by retrying fetch for particular sites but missing sites are not tolerated.
- Predict stage: all generator sites must have predictions for the next 24 hours.
- Serving stage: all the sites must be in the pre-computed CSV file. This is to ensure failures happening in mid-write process is caught.


The pipeline fails closed rather than emitting a plausible-looking but incomplete
batch. Each gate raises rather than exiting, so failures are visible to the
orchestrator and eligible for retry.

| Gate | Location | Fires when |
|---|---|---|
| Roster completeness | `fetch_forecast_all.py` | Expected sites missing from the weather fetch |
| Roster completeness | `predict_pc.py` | Any of the 45 sites absent from the prediction batch |
| NaN features | `predict_pc.py` | Any NaN feature-hour (`MAX_NAN_FEATURE_HOURS=0`) |
| Short window | `predict_pc.py` | A site's forecast window is shorter than expected |

All three `predict_pc` gates are evaluated **after** the per-site loop completes.
An earlier version evaluated them at the top of the loop against dictionaries
populated at the bottom, so a bad site processed last — or a single-site run —
was recorded but never checked, and a green run could write a bad CSV. The
regression tests for that case are in `tests/test_predict_pc_gates.py`, which
covers seven fire paths including a clean-batch control, NaN in the middle, last,
and only site, a short window in the last site, a missing site, and model/mapping
drift.

Gates that have only been verified to pass on clean data have not been verified
at all. Each of these has a negative test that injects the failure it is meant to
catch.

---

## Repository structure

```
wind-power-forecast/
├── src/wind_forecast/
│   ├── config.py          # DATA_ROOT / MODELS_ROOT resolution
│   ├── storage.py         # I/O router: local disk vs gs://
│   ├── model.py           # LSTM network definition (offline path)
│   ├── ingest/            # IESO download + weather fetch
│   ├── features/          # Feature engineering (training)
│   ├── predict/           # predict_pc.py (live), predict.py (LSTM, offline)
│   ├── evaluate/          # evaluate_daily.py, evaluate_and_log.py
│   ├── train/             # Training, fine-tuning, hyperparameter search
│   └── pipeline/
│       └── daily.py       # `wf-daily`: fetch → predict, one process
├── serving/               # FastAPI service deployed to Cloud Run
├── orchestration/
│   └── trigger.py         # Prefect flow: triggers the Cloud Run Job
├── flows/                 # Local-development orchestration (see note)
├── tests/
├── docs/
└── pyproject.toml         # extras: [dl] (torch), [pipeline] (prefect, wandb)
```

`storage.py` is the load-bearing piece. Every read and write goes through it, and
it routes on path prefix: `data/...` to local disk, `gs://bucket/...` to Cloud
Storage. The same code runs unchanged on a laptop with `DATA_ROOT=data` and in
the Cloud Run Job with `DATA_ROOT=gs://wind-forecast-ontario-data`.

**A note on `flows/`.** These are Prefect flows for local development and
iteration — they run end-to-end and are useful for exercising the pipeline
interactively, but they are not what runs in production. The deployed schedule
triggers the Cloud Run Job via `orchestration/trigger.py`; `flows/` is not
deployed anywhere. Note that a local flow run inherits whatever `DATA_ROOT`
points at, so running one with `DATA_ROOT=gs://...` will write a real
off-schedule batch into the production bucket.

---

## Local development

Requires Python 3.10+. CPU is sufficient; the live XGBoost path has no GPU
dependency and the Job image does not install torch.

```bash
git clone https://github.com/alvinlitani/wind-power-forecast.git
cd wind-power-forecast
pip install -e ".[dl,pipeline]"

# Storage roots default to ./data and ./models
cp .env.example .env
```

Run the production pipeline locally:

```bash
python -m wind_forecast.pipeline.daily
```

Or individual stages:

```bash
python -m wind_forecast.ingest.fetch_forecast_all
python -m wind_forecast.predict.predict_pc --run-timestamp 20260721_0200
```

Or the development flows:

```bash
python -m flows.ingest_flow
python -m flows.predict_flow
python -m flows.evaluate_flow
```

### Evaluating a batch

Scoring is currently a three-step manual workflow, because the scheduled
pipeline does not ingest IESO actuals:

```bash
# 1. Push IESO actuals to the bucket
DATA_ROOT=gs://wind-forecast-ontario-data python -m flows.ingest_flow

# 2. Choose a batch at least two days old (GOCR lag + midnight-straddling window)

# 3. Score it, with W&B logging
python -m wind_forecast.evaluate.evaluate_and_log --run-timestamp 20260718_0202

# ...or without
python -m wind_forecast.evaluate.evaluate_and_log --run-timestamp 20260718_0202 --no-wandb
```

---

## Data sources

- **IESO Generator Output and Capability Report** — hourly per-generator output
  and available capacity for market-participant generators of 20 MW or greater.
  Published with roughly a one-day lag. Used as ground truth. Its Forecast column
  is not used as a baseline; see [Evaluation](#why-the-ieso-gocr-forecast-column-is-not-a-baseline).
- **Open-Meteo Historical Forecast API** — past *forecast* weather, used for
  training. Mirrors production conditions rather than reanalysis.
- **Open-Meteo Forecast API** — live forecasts at inference time. Best-match
  model selection per coordinate, deterministic across training and inference for
  any given site.
- **Wind Turbine Database (FGP)** — per-site nameplate, hub height, rotor
  diameter.

---

## Roadmap

Shipped:

- [x] Local pipeline: ingest, features, train, predict, evaluate
- [x] Storage abstraction (local ↔ GCS)
- [x] Python package + Prefect flows
- [x] Cloud Run Job running the daily fetch-and-predict pipeline
- [x] Prefect Cloud scheduling the Job (Managed pool, invoke-only credentials)
- [x] FastAPI on Cloud Run serving predictions from GCS
- [x] Fail-closed data quality gates with negative tests
- [x] W&B evaluation logging with persistence reference and skill score

Next:

- [ ] GitHub Actions CI with Workload Identity Federation, plus `CODE_SHA`
      injection at deploy so the provenance column becomes discriminating
- [ ] Conditional bias correction (see [Known limitations](#conditional-bias-regression-to-the-mean))
- [ ] Availability-based outage and curtailment filtering using the GOCR
      Available Capacity column
- [ ] Scheduled evaluation — requires adding an IESO ingest stage to the
      scheduled pipeline
- [ ] Day-ahead horizon (00:00–23:00 on D+1, 22–46 hours ahead) to align with
      DAM bidding. The power curve carries no lead-time features, so this is a
      configuration change rather than a retrain; accuracy will degrade with
      lead time and would need re-baselining.
- [ ] Fleet-aggregate benchmark against IESO's Variable Generation Forecast
      Summary, the only lead-time-comparable public reference available
- [ ] Sequence models in the live path. An LSTM with site embeddings is trained
      and evaluated offline but is not served — its roster gate is fail-open,
      which is a prerequisite for wiring it in. A Temporal Fusion Transformer is
      the intended upgrade beyond that.
- [ ] Standardize wind speed units on m/s (retrain; batched with the next model
      upgrade)
- [ ] Terraform for buckets, IAM, and Cloud Run
- [ ] `/history` endpoint for served evaluation history
- [ ] National scope expansion beyond Ontario

---

## MIT License

Copyright (c) <2026> 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
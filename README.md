# Wind Power Forecasting (Ontario)

Daily hourly forecasts of wind generation output for all 45 IESO-reporting wind generator plants in Ontario. Per-site XGBoost models turns forecasts for the next 24 hours of weather conditions into predicted energy output which is written to cloud storage. Access is provided through a small FastAPI service hosted on Google Cloud Run.

You can test the API at the following endpoints:
- [24-hour per-site output forecast starting at 3 AM](https://wind-forecast-api-654769911920.us-central1.run.app/predictions/latest)
- [24-hour total Ontario output forecast starting at 3 AM](https://wind-forecast-api-654769911920.us-central1.run.app/predictions/ontario)

My goal is to have this project built using production environment practices rather than demo standards. There is scheduling that runs on Prefect Cloud. The accuracy is logged into Weights and Biases (W&B) against a persistence baseline which is the conventional reference for wind forecast skill scores.

## Why this problem

To maintain grid stability, the baseload power must be able to cover energy demand if the renewable energy sites with variable output produce less than expected. In Ontario, baseload power is provided by nuclear, hydro, and gas. According to [IESO data](https://www.ieso.ca/Learn/Ontario-Electricity-Grid/Supply-Mix-and-Generation), the share of total capacity provided by wind is 13% while the combined baseload is about 82%. 

Wind energy output is variable and dependent on weather. However, the output is not all absorbed by the Ontario grid. The IESO issues dispatch instructions meaning its scheduling algorithm sets the expected energy output of each generator sites. The plant report the amount that it can produce in current condition and the IESO schedules the output taking into account the grid condition. 

For renewable energy like solar and wind, dispatch instructions can only runs downwards (curtailment) since you can not instruct the generator sites to produce more if the weather conditions do not allow it while the output can be restricted. The reason for curtailment is that during times of surplus baseload generation, the market already clears and does not need additional capacity provided by the renewable plants.

The IESO already has made variable energy forecasting available to market participants. The [forecast summary](https://reports-public.ieso.ca/public/VGForecastSummary/PUB_VGForecastSummary.xml) is done for the next 48 hours and refreshes hourly. The forecast summary is not granular since it is broken down into electrical zones (not per-site) and not all the zones are covered.  The reason that some zones are not covered may be that there are too few plants are in the region and therefore the summary can be broken down into individual sites' expected output. The per-site forecast in the [IESO Generator Output and Capability Report](https://reports-public.ieso.ca/public/GenOutputCapabilityMonth/) is given out retroactively on the next day.

This project aims to give granular per-site forecast for market participants beforehand for all IESO-covered wind generators.

---

## What is running now

A single Cloud Run Job (`wind-forecast-job` running the `wf-daily` entry point) is triggered daily at 02:00 ET. It fetches weather for 45 sites, runs 45 XGBoost models, and writes predictions as CSV to Google Cloud Storage. Prefect Cloud have the schedule and triggers the job. The job is deliberately made lightweight and usually executes under a minute each run.

A separate Cloud Run service reads the CSVs and serves the results at API (`wind-forecast-api`) endpoints. 

Evaluation runs on my local machine and logs to Weights & Biases. 

## Summarized numbers

| Parameters | Value |
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
| 02:00 | `wf-daily` | Weather fetch -> XGBoost predict -> write CSV to GCS |

Many weather models used by Open-Meteo have a [daily schedule](https://wethr.net/model-schedule) of running at 00, 06, 12, 18 UTC and they publish a few hours after that. This project aims to fetch the fresh weather forecast after the 00 run. 

---

## Modeling

## Capacity factor as the target

The model predicts capacity factor (output ÷ nameplate) rather than raw MWh. The sites' capacity range between 20 MW to 270 MW which is a large 13.5× range. Predicting CF keeps error comparable between them and stops the big sites dominating any pooled number. Conversion back to MWh happens at serving time.

## Train on forecast weather, not actual weather data

In production, the model will have to use weather forecast for inference and not actual weather data. Therefore, I trained the model using Open-Meteo's Historical Forecast API instead of Historical Weather API. This also avoids the potential failure mode of the model trained on near-perfect data performing sharply worse in production.

## Feature selection 

Turbine power is calculated by the equation: 1/2 × ρ (air density) × A (rotor swept area) × V^3 (wind speed) × Cp (power coefficient).

Two features relating to air density which are temperature and surface pressure is included. Humidity is excluded as it contributes very little to the variability of air density compared to the other two features.

Other excluded features which were considered:

- **Wind direction**: yaw-controlled turbines make raw direction largely irrelevant to output.
- **Site elevation**:  already reflected in wind speed at height and in pressure/temperature.
- **Distance to water**: the phenomenon of increased wind near large water bodies is already reflected in wind speed.
- **Air density**: derivable from already included temperature and pressure.

Features for the live model per site per hour: `wind_speed_80m`, `wind_speed_120m`, `temperature_2m`, `surface_pressure`.

## Hub-height handling

I considered extrapolating/interpolating the two wind speed features to calculate hub-height wind speed per site but decided against it.

Each per-site model receives the 80m and 120m speeds directly and learns the mapping for its own site. The alternative is using a shear coefficient and applying it to calculate hub-height wind speed across 45 sites with very different terrain. 

Some sites also have turbines at different heights which make it impossible to use only one height to represent it. Hub heights across the fleet run 78m to 132m with the data taken from [Canadian Wind Turbine Database](https://open.canada.ca/data/en/dataset/79fdad93-9025-49ad-ba16-c26d718cc070).

---

## Evaluation

## Metrics

The standard error metrics used in wind energy forecasting are normalized because it allows us to compare error across plants of different sizes. The normalized mean absolute error (nMAE) and normalized root mean square error (nRMSE) values are used to compare the error rate between predicted and actual values of the plants' capacity factor. 

Mean absolute percentage error (MAPE) is deliberately not used. It degrades badly at small or zero generation and roughly a fifth of site-hours in the IESO data have zero output.

A baseline is needed for comparison and a naive persistence model is the conventional reference for renewable energy outputs. Persistence model assumes the predicted value of the previous time step (yesterday) will be the same for the next time step (today). This means the energy output of a particular site at 1 PM today will be the same as yesterday. The accompanying skill score value (1 - (forecast / reference)) is also logged with it.

The reported metrics are for hourly forecasts for the next 24 hours starting from 3 AM.

## Three ways of aggregation

Error is reported in three ways because they answer different questions:

- **Fleet-aggregate nMAE** : sums generation and prediction separately across all sites then measures the difference/error. Over-prediction at one site cancels under-prediction at another one. How accurate is the prediction for the whole fleet?
- **Capacity-weighted per hour**: sums up the error sizes and divide it by the sum of capacity. Negative signs are not kept. Big sites dominate because they contribute more MW to both sides. This is the standard figure in the literature. How much error per MW installed?
- **Per-site-equal**: calculate each site's error as a percentage of its own capacity then average all the percentages. Which sites have the best and worst predictions? How wrong is the forecast at a typical site?


## Why not the Generator Output and Capability Report Forecast column as the baseline

The Generator Output and Capability Report have a Forecast column but I am not using it as the baseline. The reason is lead-time mismatch.

The Forecast numbers in the Report is short-lead forecast using live telemetry instead of a day-ahead forecast. It can be seen by the way of its near-perfect accuracy which shows ~1.6% MAE with no bias when measuring against the Output column in July 2026. A study conducted by [Miettinen et al.](https://doi.org/10.1002/we.2410) studying wind power error forecast distributions states that "The average site‐specific MAE is 10.7 % of the installed capacity." for day-ahead forecasting. For short-term forecasting, another study by [Würth et al.](https://www.mdpi.com/1996-1073/12/4/712) states that "the background mean absolute error (MAE) just under 4% of installed capacity".

The Variable Generation Forecast Summary runs 48 hours ahead and have comparable lead-time. However, it publishes only zonal totals and no per-site breakdown.

---

## Offline results for full-year 2025 test

| Metric | Value |
|---|---|
| nMAE, per-site-equal | 11.84% |
| nMAE, capacity-weighted | 11.31% |
| MAE | 12.43 MWh |
| Bias | +0.13 MWh |
| winter nMAE | 13.70% |
| spring nMAE | 13.81% |
| summer nMAE | 9.16% |
| fall nMAE | 10.70% |

The nMAE values are in line with the values from the quoted study above. Season matters a lot with nMAE going up on more windy seasons.

## Live results 

The 02:00 ET job has run daily since 16 July to 19 August 2026 with 35 batches and no missed days. Each batch covers 45 sites over 24 hours. It then gets scored against IESO Generator Output and Capability Report actual values.

A full batch is 1080 site-hours = 45 sites × 24 hours. Four batches have way less site-hours than 1080 site-hours because of missing actual values in the report. Therefore I left them out: 21 July, 29 July, 31 July, and 3 August. There are 25 full batches and 6 almost-full batches. The median values below are from those 31 batches.

| Metric | Median value | 
|---|---|
| nMAE, per-site-equal | 8.71% |
| nMAE, capacity-weighted | 8.17% |
| nMAE, fleet-aggregate | 4.86% | 
| MAE | 8.97 MWh | 
| Skill score (MAE) vs. persistence | 0.366 |
| Skill score (RMSE) vs. persistence | 0.457 |
| Persistence reference, nMAE | 14.69% | 

Persistence model means using the actual output from the same hour yesterday as the forecast. An example is that the output at 10 AM on 24 July is assumed to be similar to the output at 10 AM on 23 July.

Median per-site nMAE of 8.71% is in line with published day-ahead results. According to the survey of forecasting models done by [Piotrowski et al.](https://www.mdpi.com/1996-1073/15/24/9657), reported nMAE has a median of ~8.9% for different models and ~7.3% for best models. Those studies mostly cover single wind farms over a time period compared to this project which is a 45-site fleet scored daily.

Fleet-aggregate nMAE of 4.86% is comparable to published regional forecast errors. A study of Nordic wind power models by [Miettinen and Holttinen](https://cris.vtt.fi/en/publications/characteristics-of-day-ahead-wind-power-forecast-errors-in-nordic/) shows an average day-ahead MAE of 5.7% of installed capacity across individual Nordic regions. The smallest areas have about 8% MAE while dropping to 2.5% when all four Nordic countries are aggregated together. Ontario's wind capacity sits in between those two levels as it is larger than one of their regions but smaller than the whole Nordic system. 

## Positive bias

There are small biases on the run batches with positive value on 26 batches. Overall, the model predicts more output than what is actually produced. This is in line with earlier observation that the model over-predicts when output is low. Ontario wind runs below 20% capacity factor about 47% of the time so most hours are over-predicted and the net bias comes out positive.

## Skill score against persistence

The skill_score_mae value is calculated by the formula 1 − (pc/nmae_pct  / persistence/nmae_pct). There are 30 of 31 batches beating persistence on MAE with the one loss on 10 August. Skill score of 0 means the two values are exact while negative means persistence beats the model.

Beating persistence is a low bar as this shows the weather inputs add information beyond yesterday's output. 

## Errors cancel out across the fleet

Fleet-aggregate error (4.86%) is about 40% lower than per-site error (8.17%). Sites are spread across the province with varying geographical conditions so some have higher error rates than others. The sites that are forecast too high and the sites that are forecast too low partly cancel out each other in the provincial total. A 40% reduction is in line with what the literature reports (40–63%). 

Aggregate error is what matters to a system operator; per-site error is what matters to an asset owner.

## Caveat

All live days are July and August during summer which is Ontario's low-wind/low-output season. The model has not been tested live against winter weather when the winds are stronger and more variable. The low temperature and icing on the blades will also affects output.

Published day-ahead nMAE for similar gradient-boosted models is around 10–12% (Miettinen et al. study). The 8.71% value is around the same ballpark figure. 

---

## Known limitations

## Misleading bias value

The aggregate bias is near zero (+0.13 MWh on the 2025 test set) which looks good at first glance. However, if we break down the numbers by capacity factor:

| Actual CF | MAE (MWh) | Bias (MWh) | % of hours |
|---|---|---|---|
| 0–5% | 9.94 | **+9.83** | 24.1% |
| 5–20% | 9.49 | +6.16 | 22.9% |
| 20–50% | 13.14 | +0.19 | 24.3% |
| 50–80% | 16.24 | **−10.16** | 15.0% |
| 80–100% | 16.41 | **−16.05** | 13.4% |

When the wind is weak, it predicts over actual output. The opposite happens with the model predicts under actual output when the wind is strong. The bias is near zero as those cancel each other out. 

Ontario wind generators are below 20% CF around 47% of the time, and above 50% around 28% of the time. Most of the bias occurs by over-predicting the output meaning less actual power is generated than the forecast.

Several possible causes are: 
- the model is underfit
- curtailment
- the weather forecast is inaccurate for wind speed especially for 24 hours ahead.

## Other limitations

**Curtailment are invisible to the model**: When IESO algorithm instructs a wind farm to produce less electricity for market or grid stability reason, the output drops not because of the weather. The curtailment hours are not indicated in the Generator Output and Capability Report. Therefore, the model will produce error during curtailment as dispatch decisions are invisible to it. This may also be part of the reson for under-prediction during high CF times when the wind is strong. Most curtailment happens on windy nights and demand is low which is exactly when the model under-predicts.

**Outages are invisible to the model**: Maintenance decisions are not published beforehand therefore this is another possible source of error. A particular site may have some of its turbines offline for a time and it will produce less output than expected. 

**Training weather is better than production weather**: Open-Meteo's historical forecast archive stores short forecasts made close to the time of the weather models run. This means the wind speed data used for training is close to when it actually happened. In production, I use weather forecast up to 24 hours ahead where the wind speed used as input is considerably less accurate. 

**Anomaly handling is hand-written**: I excluded on purpose a site (BOWLAKE) where the mean output is exactly zero while there is available capacity. An anomaly detector is out of scope for this initial version.   

**Evaluation is manual**: Currently, the scheduled job only downloads the weather and predicts the output. It never automatically downloads actual IESO data. Evaluating a prediction batch is done manually by running the ingest flow then the evaluation flow. Steps below for [evaluating a batch](#evaluating-a-batch).

**Training scripts use local paths**: The script for training a new model is outdated since it still  use local filesystem paths. It runs on local machine instead of in the cloud. 

**CI feature not available yet**: The prediction CSV have a column 'code_sha' which stores the particular git commit of the prediction code running. It is not working yet and currently only prints 'local'.

---

## Data quality gates

I want to ensure that the runtime fails completely rather than having corrupted data output. There are several checks that raise exceptions so it is recorded as failure instead of silently stopping. Those are:
- Fetch stage: all generator sites must have weather forecasts for the next 24 hours. Recovery is done by retrying fetch for particular sites but missing sites are not tolerated.
- Predict stage: all generator sites must have predictions for the next 24 hours.
- Serving stage: all the sites must be in the pre-computed CSV file. This is to ensure failures happening in mid-write process is caught.

---

## Local development

Requires Python 3.10+. CPU is fine. The live XGBoost model has no GPU dependency and the Job image does not install torch.

```bash
git clone https://github.com/alvinlitani/wind-power-forecast.git
cd wind-power-forecast
pip install -e ".[dl,pipeline]"
```

Run the production pipeline locally:

```bash
python -m wind_forecast.pipeline.daily
```

Individual stages:

```bash
python -m wind_forecast.ingest.fetch_forecast_all
python -m wind_forecast.predict.predict_pc --run-timestamp 20260721_0200
```

The flows in flows/ are for local testing only. In production, the Prefect schedule triggers the Cloud Run Job through orchestration/trigger.py and flows/ is not used anywhere. Development flows:

```bash
python -m flows.ingest_flow
python -m flows.predict_flow
python -m flows.evaluate_flow
```

### Evaluating a batch

Scoring is currently a three-step manual workflow because the scheduled pipeline does not ingest IESO actuals:

```bash
# 1. Push IESO actuals to the bucket
DATA_ROOT=gs://wind-forecast-ontario-data python -m flows.ingest_flow

# 2. Choose a batch at least two days old (GOCR lag + midnight-straddling window)

# 3. Score it with W&B logging
python -m wind_forecast.evaluate.evaluate_and_log --run-timestamp 20260718_0202

# Without logging
python -m wind_forecast.evaluate.evaluate_and_log --run-timestamp 20260718_0202 --no-wandb
```

---

## Data sources

**[IESO Output and Capability Report](https://reports-public.ieso.ca/public/GenOutputCapabilityMonth/)**: The report contains hourly output, available capacity, and forecast for IESO plants with capacity of 20 MW or greater. This project uses the report as ground truth. The Forecast column is not used as baseline for reason stated above.

**[Canadian Wind Turbine Database](https://open.canada.ca/data/en/dataset/79fdad93-9025-49ad-ba16-c26d718cc070)**: The database has detailed information about the wind turbine locations such as: number of turbines in a location, year of commision, manufacturer, hub height, rotor diameter, etc. For this project, the latitude/longitude of each turbine in a site is used to calculate the centroid of that site. The centroid is then used to represent the location when requesting weather forecasts. 

**[IESO Active Contracted Generation List](https://www.ieso.ca/-/media/Files/IESO/Document-Library/power-data/supply/IESO-Active-Contracted-Generation-List.xlsx)**:  The list contains generator sites that are currently contracted with IESO. It has detailed information such as: operation starting date, contract dates, IESO zones, etc. The list is used to confirm actual locations of the wind generator sites as it has municipality locations of the sites.

**[Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api)**: The API serves past weather forecasts and not past actual weather conditions. The data is used for training the model.

**[Open-Meteo Forecast API](https://open-meteo.com/en/docs)**:  The API serves live forecasts which is used for inference. 

---

## Shipped

- Local pipeline: ingest, features, train, predict, evaluate
- Python package + Prefect flows
- Cloud Run Job running the daily fetch-and-predict pipeline
- Prefect Cloud scheduling the Job (Managed pool, invoke-only credentials)
- FastAPI on Cloud Run serving predictions from GCS
- Fail-closed data quality gates 
- W&B evaluation logging with persistence reference and skill score

---

## MIT License

Copyright (c) <2026> 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# K2 Wind Farm Energy Output Prediction

A production-grade ML pipeline that predicts hourly energy output (MWh) for wind farms in Ontario, starting with K2 Wind Farm in Goderich, Ontario. Designed to scale to multiple sites across Canada.

---

## Project Goals

- Predict the next 24 hours of hourly energy output for a wind farm, once daily
- Demonstrate a production-grade ML pipeline with daily data ingestion, experiment tracking, and monitoring
- Build a generalizable multi-site model architecture that transfers knowledge across wind farms

---

## Target Site: K2 Wind Farm

- **Location:** Goderich, Ontario (43.74°N, 81.71°W)
- **Capacity:** 270 MW
- **Turbines:** 140 × Siemens SWT-2.3-101
- **Hub height:** ~80m
- **Notable:** ~10km from eastern shore of Lake Huron, which creates local microclimate effects

---

## Data Sources

### IESO Generator Output and Capability Report
- **URL:** https://www.ieso.ca/power-data
- **Frequency:** Daily XML, ~1-day lag
- **Parsed with:** `xml.etree.ElementTree` with explicit namespace `http://www.ieso.ca/schema`
- **Note:** `pd.read_xml()` fails to capture nested context; manual three-loop parsing required

### Open-Meteo Historical Forecast API
- **URL:** https://historical-forecast-api.open-meteo.com
- **Frequency:** Hourly
- **Note:** Returns what the forecast model *would have predicted*, not perfect hindsight actuals. This is intentional — training data should mirror production conditions where only forecasts are available.

---

## Prediction Setup

- **Prediction frequency:** Once daily, late at night (~11pm)
- **Horizon:** Next 24 hours (hourly resolution)
- **Rationale for late-night polling:** Minimizes the gap between forecast issue time and actual output, improving forecast accuracy while still predicting a full day ahead

---

## Target Variable

**Output (MWh)** — actual metered production as reported by IESO.

### Why Output and not Available Capacity or Forecast

The IESO report provides three measurements for wind generators:

| Measurement | Definition |
|---|---|
| Output | Actual metered production. Includes all real-world effects: low wind, curtailment, outages. ±10 MW metering variance. |
| Available Capacity | Maximum potential output minus turbine derates and outages. Does **not** account for wind availability. Reflects turbine health only. |
| Forecast | IESO's own wind-aware prediction. Accounts for Available Capacity plus forecasted wind availability. |

**Output** is chosen as the target because:
- It is the ground truth of what actually hits the grid
- Available Capacity has no wind signal — a gap between Available Capacity and Output is primarily explained by wind conditions, not curtailment
- Using Forecast as a target would mean predicting IESO's own prediction, which conflates this model's independent value with theirs

### Noise and irreducible error floor

Output includes curtailment — when IESO dispatches K2 down due to grid surplus or transmission constraints. Curtailment is indistinguishable from low wind in the Output values alone. This creates an irreducible error floor that the model cannot predict around, since curtailment decisions depend on real-time grid conditions not available as forecast inputs.

The gap between Output and Forecast (IESO's prediction) is bidirectional — Output can exceed Forecast under optimal wind conditions — confirming this is forecast error rather than a clean curtailment signal.

---

## Feature Selection

### Time-Varying Encoder Features (past observations fed to encoder)

| Feature | Source | Rationale |
|---|---|---|
| Output (MWh) | IESO | Historical actual production as temporal context for the encoder |
| Available Capacity (MW) | IESO | Turbine health indicator. Low Available Capacity signals derates or outages independent of wind conditions |
| Wind speed at 80m (m/s) | Open-Meteo | Primary driver of wind power output. 80m matches K2 hub height |
| Temperature (°C) | Open-Meteo | Affects air density and turbine performance |
| Surface pressure (hPa) | Open-Meteo | Affects air density |
| Air density (kg/m³) | Derived | Computed from temperature and pressure: ρ = P / (R × T), where R = 287.05 J/(kg·K). Directly affects power output and avoids making the model discover this relationship implicitly |
| Wind direction deviation (°) | Derived | See wind direction section below |

### Time-Varying Decoder Features (future forecasts fed to decoder)

| Feature | Source | Rationale |
|---|---|---|
| Wind speed at 80m (m/s) | Open-Meteo forecast | Primary weather driver, available as forecast |
| Temperature (°C) | Open-Meteo forecast | Available as forecast |
| Surface pressure (hPa) | Open-Meteo forecast | Available as forecast |
| Air density (kg/m³) | Derived | Computed from forecast temperature and pressure |
| Wind direction deviation (°) | Derived | Computed from forecast wind direction and site prevailing direction |
| Available Capacity (MW) | IESO | Planned outages and derates are known ahead of time |

**Note:** Historical Output is not available as a decoder feature — it is the prediction target.

### Static Site Features (fed once as site context)

| Feature | Rationale |
|---|---|
| Elevation (m) | Captures terrain effects on local wind patterns |
| Distance to nearest large water body (km) | Captures microclimate effects. For K2, proximity to Lake Huron (~10km) creates lake-effect wind patterns |
| Prevailing wind direction (°) | Site-specific historical prevailing direction, used to compute wind direction deviation feature |
| Capacity (MW) | Sets the scale of expected output |
| Hub height (m) | Determines the relevant wind speed measurement height |
| site_id | One-hot encoded site identifier. Captures residual site-specific quirks not explained by other static features. Only meaningful for known sites — new sites rely on physical features until fine-tuned |

### Features Considered and Rejected

**Raw latitude and longitude**
- Raw coordinates are arbitrary numbers with no physical meaning to the model
- Replaced by elevation and distance to water, which carry actual physical signal about terrain and microclimate

**Raw wind direction**
- Wind direction is physically meaningful but its *effect* on output is site-specific — it depends on turbine orientation and layout, which is not publicly available and cannot be generalized across sites
- Replaced by wind direction deviation from prevailing (see below)

**Time features (hour of day, day of year)**
- Season is already captured by temperature and pressure — adding month/day of year would be redundant
- Time features would bias the model against freak weather events: if an unusual wind condition occurs in July that normally only happens in January, weather features correctly represent it while time features would push predictions toward July-typical output
- Dropped in favor of features with direct physical causality

**IESO Forecast measurement**
- Using IESO's own forecast as a feature conflates this model's predictions with theirs
- Does not add independent signal

---

## Wind Direction Handling

### The Problem
Wind direction is one of the most important drivers of wind farm output. However, the effect of any given wind direction depends on turbine orientation and layout, which varies per site and is not publicly available. A westerly wind at K2 may be highly productive while the same direction is suboptimal at a site with different turbine layout.

Including raw wind direction would allow the model to learn site-specific directional response for known sites, but would not generalize to unseen sites.

### Solution: Deviation from Prevailing Direction
Wind direction is encoded as the angular deviation from the site's historical prevailing wind direction:

```python
deviation = (wind_direction - prevailing_direction + 180) % 360 - 180
```

This normalizes wind direction to a site-agnostic scale:
- `0°` = wind is coming from the most historically productive direction
- Large deviation = wind is coming from an atypical direction

Prevailing direction is a static site feature computed from historical data and stored per site.

### Why This Generalizes
A new site can compute its own prevailing direction from historical Open-Meteo data before any fine-tuning. The model then sees deviation from prevailing as a consistent signal across all sites regardless of turbine orientation.

---

## Model Architecture

### Overview
Sequence-to-sequence (seq2seq) LSTM with transfer learning and per-site fine-tuning.

### Why Seq2Seq

Three architectures were considered:

| Architecture | Description | Problem |
|---|---|---|
| Sliding window | Predict one hour at a time, repeat 24 times | Prediction errors compound across 24 steps |
| Direct multi-output | Single forward pass, dense layer outputs all 24 values | No inter-hour dependencies in output sequence |
| Seq2Seq | Encoder reads past, decoder generates 24 steps with future weather as input | Best fit — future weather forecasts are known and can be exploited at each decoder step |

Seq2seq is chosen because future weather forecasts for the full 24-hour horizon are available at prediction time. The decoder can condition each hourly prediction on the forecast for that specific hour, avoiding error compounding and leveraging the full forecast.

### Architecture Details

```
Encoder
  Input: 48-hour window of past observations (time-varying encoder features + static site features)
  Output: Hidden state summarizing recent wind and output history

Decoder
  Input: Future weather forecasts (24 steps) + static site features + encoder hidden state
  Output: 24 hourly MWh predictions
```

**Encoder window: 48 hours**
- Captures yesterday's full diurnal cycle plus the day before
- Long enough to capture ramp-up/ramp-down patterns without excessive sequence length

### Why LSTM over XGBoost

XGBoost with MultiOutputRegressor treats each of the 24 output hours as independent targets and does not model temporal dependencies between them. Wind output is autocorrelated across hours — a ramp-up at hour 3 informs hour 4. LSTM natively captures this through its hidden state.

XGBoost is used for pipeline validation on limited data before committing to the full LSTM training.

### Why LSTM before TFT (Temporal Fusion Transformer)

TFT extends LSTM with gating mechanisms, multi-head attention, and purpose-built static covariate encoders. It is state of the art for multi-horizon forecasting but significantly more complex to implement.

The progression is:
1. **XGBoost** — validate data pipeline and feature engineering
2. **LSTM seq2seq** — validate sequence modeling and production pipeline
3. **TFT** — upgrade model once pipeline is stable and multiple sites are available

TFT's static covariate encoder maps directly onto the static/time-varying feature split already established, making the upgrade straightforward once the LSTM pipeline is working.

---

## Multi-Site Strategy

### Pretraining (general model)
Train on all available sites combined. The model learns shared wind-to-power physics across sites. Static site features provide the context for the model to learn how geography and site characteristics modify the general relationship.

All features including static site features and wind direction deviation are used during pretraining. site_id is included but carries limited signal for generalization — physical features do the heavy lifting for unseen sites.

### Fine-tuning (per-site adaptation)
After pretraining, fine-tune per site on site-specific historical data:
- Freeze shared physics layers
- Unfreeze and retrain site-specific layers

Fine-tuning captures residual site-specific effects that engineered features cannot fully explain:
- Local terrain turbulence not captured by Open-Meteo grid resolution
- Site-specific curtailment patterns due to local transmission constraints
- Turbine model-specific power curve characteristics
- Lake-effect or other microclimate patterns (e.g. K2's Lake Huron proximity)

### Generalization to Unseen Sites
A new site provides its static features (elevation, distance to water, prevailing direction, capacity, hub height). The pretrained model uses physical geography to make an informed initial prediction before any fine-tuning. Fine-tuning on local historical data then adapts to site-specific quirks.

This is the key advantage of replacing raw lat/lon with physically meaningful features — a new site at coordinates the model has never seen can still be characterized meaningfully through elevation and water proximity.

---

## MLOps Stack

| Component | Tool | Rationale |
|---|---|---|
| Infrastructure | Terraform | IaC, run locally |
| CI/CD | GitHub Actions | Triggered by code changes |
| Orchestration | Prefect Cloud | Daily scheduled pipeline execution |
| Experiment tracking | Weights & Biases | Model versioning and metric tracking |
| Monitoring | Grafana Cloud | Production dashboards |
| Prediction API | FastAPI on GCP Cloud Run | Serves 24-hour predictions |
| Artifact storage | GCP Cloud Storage | Model artifacts and processed data |
| Prefect worker | Oracle Cloud ARM VM | Always-free, avoids GCP Cloud Run Jobs complexity |

**Note:** Prefect handles scheduled daily execution independently of GitHub Actions. GitHub Actions handles CI/CD triggered by code changes. These are complementary, not redundant.

---

## Domain Notes for Interviews

- **IESO curtailment:** IESO can dispatch wind farms down during grid surplus or transmission congestion. Curtailment is indistinguishable from low wind in Output values — both appear as lower-than-expected output given wind conditions. This creates an irreducible model error floor.
- **Metering variance:** IESO Output values have ±10 MW operational metering variance. Small gaps between Output and Forecast are within this noise range.
- **Available Capacity vs wind availability:** Available Capacity reflects turbine health only — it does not drop when wind is low. A large gap between Available Capacity and Output is primarily explained by wind conditions, not curtailment.
- **Historical Forecast API:** Training uses the Historical Forecast API (not the Archive API) to match production conditions — the model will always run on forecasts, never on perfect hindsight actuals.
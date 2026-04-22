# K2 Wind Farm Energy Output Prediction

A production-grade ML pipeline that predicts hourly energy output (MWh) for wind farms in Ontario. Project starting with K2 Wind Farm in Ashfield-Colborne-Wawanosh, Ontario. Designed to scale to multiple sites across Canada.

---

## Project Goals

- Predict the next 24 hours of hourly energy output for a wind farm, once daily
- Demonstrate a production-grade ML pipeline with daily data ingestion, experiment tracking, and monitoring
- Build a generalizable multi-site model architecture that transfers knowledge across wind farms

---

## Target Site: Ontario wind farms

- **Location:** Ashfield-Colborne-Wawanosh, Huron County, Ontario (43.89, -81.62)
- **Commissioned:** 2015
- **Capacity:** 270 MW
- **Turbines:** 140 × Siemens SWT-2.3-101 (1824-2300 kW rated capacity), 101 m rotor diameter
- **Hub height:** 99.5 m 
- **Notable:** situated 2 - 12km from eastern shore of Lake Huron which creates local microclimate effects (mainly lake-land breeze circulation)

### Ontario Fleet Hub Height Distribution
Some of the turbines listed in the Canadian Wind Turbine Database (FGP) does not have its output recorded in the IESO report. For this analysis, only generators listed in both shall be taken into account:

| Open-Meteo level | Hub height range | Projects |
|---|---|---|
| Snap to 80m | ≤90m | ~61 projects |
| Interpolate 80m–120m | 90–110m | ~33 projects |
| Snap to 120m or interpolate 120m–180m | >110m | ~6 projects |


---

## Data Sources

### IESO (Independent Electricity System Operator) Generator Output and Capability Report
- **URL:** https://reports-public.ieso.ca/public/GenOutputCapabilityMonth/
- **Update Frequency:** Daily CSV with 1-day lag 

### Open-Meteo Historical Forecast API
- **URL:** https://historical-forecast-api.open-meteo.com
- **Note:** The forecast model uses predictions from the day before and not the actual condition on the day itself. This is intentional as the training data should mirror production conditions where only forecasts are available.

### Canadian Wind Turbine Database
- **URL:** https://open.canada.ca/data/en/dataset/79fdad93-9025-49ad-ba16-c26d718cc070
- **Update Frequency:** No set frequency

---

## Prediction Setup

- **Prediction frequency:** Once daily late at night (11.15 pm)
- **Horizon:** Next 24 hours (hourly resolution)
- **Reason for late-night polling:** Minimizes the gap between forecast issue time and actual output to improve forecast accuracy while still predicting a full day ahead

---

## Target Variable

**Output (MWh):** actual metered production as reported by IESO. The hourly output is the facility’s five-minute outputs averaged over an hour.

The IESO report provides three measurements for wind generators:

| Measurement | Definition |
|---|---|
| Output | Actual metered production including all real-world effects: low wind, curtailment, telemetry problems, etc. Variance of ±10 MW. |
| Available Capacity | Maximum potential output minus turbine derates and outages. Reflects turbine health and availability. |
| Forecast | IESO's own output prediction accounting for forecasted wind/solar availability. |

Output is chosen as the target because:
- Measurement of how much energy actually enters the grid
- Available Capacity is not useful as a target because it is potential and not actual output.
- Forecast is not used as a target but will be used for accuracy comparison purposes for the produced model.

The output value includes curtailment. This is when IESO reduces energy intake from generators due to grid surplus, transmission constraints, or grid stability. This also creates an error floor that the model cannot predict since curtailment happens depending on unpredictable, real-time grid conditions. Low Output values due to curtailment is indistinguishable from low wind conditions. 

---

## Feature Selection

Wind power is given by following equation: P = 0.5 × Cp × ρ × A × v³
Features will be selected according to the compenents of the wind power equation.

Wind direction will also be used since it matters a lot for power generation and Ontario turbines have different directions.

### Time-Varying Encoder Features (past observations fed to encoder)

| Feature | Source | Reason |
|---|---|---|
| Output (MWh) | IESO | Historical actual metered production as prediction target |
| Available Capacity (MW) | IESO | Maximum output and turbine health indicator. Low Available Capacity indicates turbine derates or outages. |
| `wind_speed_hub` (m/s) | Derived | Wind speed at hub height interpolated or snapped from Open-Meteo levels. See hub height wind speed section. |
| `wind_direction_hub` (°) | Derived | Wind direction at hub height interpolated. See hub height wind speed section. |
| Surface temperature (°C) | Open-Meteo | Used for air density calculation. Interpolation to hub height unnecessary. See temperature and pressure section. |
| Surface pressure (hPa) | Open-Meteo | Used for air density calculation. Interpolation to hub height unnecessary. See temperature and pressure section. |
| Air density (kg/m³) | Derived | Computed from surface temperature and pressure: ρ = P / (R × T), where R = 287.05 J/(kg·K). Directly affects power output. |

### Time-Varying Decoder Features (future forecasts fed to decoder)

| Feature | Source | Reason |
|---|---|---|
| `wind_speed_hub` (m/s) | Derived | Forecast wind speed at hub height. |
| `wind_direction_hub` (°) | Derived | Forecast wind direction at hub height. |
| Surface temperature (°C) | Open-Meteo forecast | Used for air density calculation. |
| Surface pressure (hPa) | Open-Meteo forecast | Used for air density calculation. |
| Air density (kg/m³) | Derived | Computed from forecast surface temperature and pressure. |
| Available Capacity (MW) | IESO | Planned outages and derates are known ahead of time. |

**Note:** Output is not available as a decoder feature as it is the prediction target.

### Static Site Features (fed once as site context)

| Feature | Reason |
|---|---|
| Elevation (m) | Captures terrain effects on local wind patterns. |
| Distance to nearest large water body (km) | Captures microclimate effects. For K2, proximity to Lake Huron creates lake-effect wind patterns. |
| Capacity (MW) | Name plate installed capacity of the farm. Distinct from Available Capacity which varies hour by hour based on outages and derates. |
| Hub height (m) | Reference height for turbine specifications. |
| site_id | One-hot encoded site identifier. Captures other site-specific quirks not explained by other static features. Only meaningful for previously learned sites with new sites relying on physical features until fine-tuned. |

### Features Considered and Rejected

**Raw latitude and longitude**


**Wind direction deviation from prevailing**


**Dual wind speed/direction at 80m and 120m directly**


**Wind shear exponent α**


**Atmospheric stability proxies (boundary layer height, wind gusts at 10m)**


**Time features (hour of day, day of year)**


**IESO Forecast measurement**


---

## Hub Height Wind Speed and Direction

Open-Meteo provides wind speed and direction at fixed heights: 80m, 120m, and 180m. Wind farm hub heights vary across Ontario's turbines from ~80m on older projects to 132m on the tallest. A consistent `wind_speed_hub` and `wind_direction_hub` feature is needed that works correctly for any site.

Wind direction will snap to nearest level since there is no standard interpolation formula equivalent to the wind speed formula. 


---

## Temperature and Pressure

Open-Meteo provides pressure at surface level and temperatures at 80m and 120m. Correcting these to hub height was considered but rejected:

**Temperature:**
Temperature changes with height at the standard lapse rate of ~6.5°C per 1000m. For the different heights of the , this is approximately 0.65°C — negligible for air density calculations.

**Pressure:**
Using the barometric formula, the pressure difference between surface and 100m is ~1.2 hPa, representing less than 0.12% change in air density. This is well within the ±10 MW metering variance of IESO Output values.

**Conclusion:** Surface temperature and surface pressure are used directly for air density computation. The corrections are smaller than the measurement noise floor and add complexity without meaningful accuracy gain.

---

## Model Architecture

### Overview
Sequence-to-sequence (seq2seq) LSTM with transfer learning and per-site fine-tuning.

### Why Seq2Seq


### Architecture Details


### Why LSTM over XGBoost


### Why LSTM before TFT (Temporal Fusion Transformer)


---

## Multi-Site Strategy

### Pretraining (general model)


### Fine-tuning (per-site adaptation)


### Generalization to Unseen Sites

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
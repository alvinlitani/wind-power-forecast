# K2 Wind Farm Energy Output Prediction

A production-grade ML pipeline that predicts hourly energy output (MWh) for wind farms in Ontario. Project starting with K2 Wind Farm in Ashfield-Colborne-Wawanosh, Ontario. Designed to scale to multiple sites across Canada.

---

## Project Goals

- Predict the next 24 hours of hourly energy output for a wind farm, once daily
- Demonstrate a production-grade ML pipeline with daily data ingestion, experiment tracking, and monitoring
- Build a generalizable multi-site model architecture that transfers knowledge across wind farms

---

## Target Site: K2 Wind Farm

- **Location:** Ashfield-Colborne-Wawanosh, Huron County, Ontario (43.89, -81.62)
- **Commissioned:** 2015
- **Capacity:** 270 MW
- **Turbines:** 140 × Siemens SWT-2.3-101 (1824-2300 kW rated capacity), 101 m rotor diameter
- **Hub height:** 99.5 m 
- **Notable:** situated 2 - 12km from eastern shore of Lake Huron which creates local microclimate effects (mainly lake-land breeze circulation)

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

### Curtailment

The output value includes curtailment. This is when IESO reduces energy intake from generators due to grid surplus, transmission constraints, or grid stability. This also creates an error floor that the model cannot predict since curtailment happens depending on unpredictable, real-time grid conditions. Low Output values due to curtailment is indistinguishable from low wind conditions. 

---

## Feature Selection

Wind power is given by following equation: P = 0.5 × Cp × ρ × A × v³

**ρ (air density, kg/m³):** — denser air contains more mass per unit volume, so more kinetic energy passes through the rotor at the same wind speed. This is why turbines produce more power on cold days than hot days at identical wind speeds.

**A (rotor swept area, m²)** — the circular area swept by the blades, calculated as π × (rotor_diameter / 2)². Larger rotors intercept more air mass per second. This is why modern turbines have increasingly large rotors — a 10% increase in rotor diameter gives a 21% increase in swept area (area scales with diameter squared).

**v³ (wind speed cubed, m/s)** — this is the dominant term and the most important to internalize. It has two contributions:
- Kinetic energy per unit mass is proportional to v² (from KE = 0.5mv²)
- Mass flow rate through the rotor per second is proportional to v (more air passes through per second at higher wind speed)
- Combined: energy per second (power) scales as v² × v = v³

The cubic relationship is why wind speed is so disproportionately important — going from 8 m/s to 10 m/s wind speed (25% increase) produces nearly double the power (25³ ÷ 8³ ≈ 1.95x). It also explains why small wind speed forecast errors cause large output prediction errors, and why your model's worst predictions will cluster around the steep part of the power curve.

**Cp (power coefficient)** — efficiency of the turbine at converting available wind power to electrical output. Varies with wind speed and turbine design, typically 0.35–0.45 for modern turbines at optimal wind speeds. Drops sharply outside the turbine's optimal operating range.

Features will be selected according to the compenents of the wind power equation.

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
- Raw coordinates are arbitrary numbers with no physical meaning to the model.
- Replaced by elevation and distance to water that carry actual physical signal about terrain and microclimate.

**Wind direction deviation from prevailing**
- Initially considered as a site-agnostic encoding of wind direction
- Rejected for two reasons: (1) circular variance analysis of 2025 K2 data showed a value of 0.798 (close to 1 = random), meaning no single dominant prevailing direction exists across seasons — the prevailing direction is a statistical artifact, not a physically meaningful reference point; (2) 3% of hours showed physically implausible veering of >90° between 80m and 120m, making a single interpolated direction unreliable
- Replaced by `wind_direction_hub` derived consistently per site

**Dual wind speed/direction at 80m and 120m directly**
- Initially considered because K2's rotor (101m diameter, 99.5m hub) sweeps from ~49m to ~150m, encompassing both Open-Meteo measurement heights
- Rejected in favour of a consistent `wind_speed_hub` and `wind_direction_hub` feature across all sites — the dual-height approach is K2-specific and does not generalize to sites that snap to a single level

**Wind shear exponent α**
- Initially considered as an atmospheric stability signal derived from 80m and 120m wind speeds
- Rejected because it is only computable for sites where two bracketing Open-Meteo levels are available — sites that snap to a single level cannot produce α, making it inconsistent across the multi-site fleet
- Atmospheric stability dropped entirely rather than replaced with a weaker proxy

**Atmospheric stability proxies (boundary layer height, wind gusts at 10m)**
- Boundary layer height: direct NWP stability indicator but uncertain relevance to hub-height power output
- Wind gusts at 10m: captures surface turbulence but is a weak proxy for conditions at hub height
- Both dropped in favour of a lean feature set with direct physical justification

**Time features (hour of day, day of year)**
- Season is already captured by temperature and pressure so adding month/day of year would be redundant.
- Time features would bias the model against freak weather events. If an unusual wind condition occurs in July that normally only happens in January, weather features correctly represent it while time features would push predictions toward July-typical output.
- Dropped in favor of features with direct physical causality.

**IESO Forecast measurement**
- Using IESO's own forecast as a feature conflates this model's predictions with theirs
- Does not add independent signal

---

## Hub Height Wind Speed and Direction

Open-Meteo provides wind speed and direction at fixed heights: 80m, 120m, and 180m. Wind farm hub heights vary across Ontario's turbines from ~80m on older projects to 132m on the tallest. A consistent `wind_speed_hub` and `wind_direction_hub` feature is needed that works correctly for any site.

**Within 10m of an available level, snap to that level. More than 10m from the nearest level, interpolate between the two bracketing levels using the log wind profile:**
```python
def get_wind_speed_hub(hub_height, wind_speed_80m, wind_speed_120m, wind_speed_180m):
    levels = {80: wind_speed_80m, 120: wind_speed_120m, 180: wind_speed_180m}
    nearest = min(levels, key=lambda h: abs(h - hub_height))

    if abs(nearest - hub_height) <= 10:
        return levels[nearest]  # snap

    # Find bracketing levels
    lower = max(h for h in levels if h <= hub_height)
    upper = min(h for h in levels if h >= hub_height)
    return interpolate_wind_speed(levels[lower], levels[upper], lower, upper, hub_height)

def interpolate_wind_speed(v_low, v_high, h_low, h_high, h_target, z0=0.03):
    log_low = np.log(h_low / z0)
    log_high = np.log(h_high / z0)
    log_target = np.log(h_target / z0)
    weight = (log_target - log_low) / (log_high - log_low)
    return v_low + weight * (v_high - v_low)
```

Wind direction will snap to nearest level since there is no standard interpolation formula equivalent to the wind speed formula. 

### Ontario Fleet Hub Height Distribution
Some of the turbines listed in the Canadian Wind Turbine Database (FGP) does not have its output recorded in the IESO report. For this analysis, only generators listed in both shall be taken into account:

| Open-Meteo level | Hub height range | Projects |
|---|---|---|
| Snap to 80m | ≤90m | ~61 projects |
| Interpolate 80m–120m | 90–110m | ~33 projects |
| Snap to 120m or interpolate 120m–180m | >110m | ~6 projects |

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
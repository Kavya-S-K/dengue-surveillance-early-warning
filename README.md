# Dengue Surveillance & Early Warning Dashboard

A Streamlit-based GIS and machine-learning dashboard for **locality-level dengue surveillance, environmental monitoring, and one-week-ahead forecasting**.

The application combines dengue surveillance information with environmental observations to prepare a weekly modelling dataset, generate a Random Forest forecast, classify relative risk across the modelled localities, and present the results through an interactive GIS dashboard.

> **Current study configuration:** The application has been configured and tested using selected localities in **Nemom Block, Thiruvananthapuram, Kerala**. The study-area configuration is kept in the application and data-processing workflow rather than being used as the main application/site title.

---

## 1. Main capabilities

The dashboard follows the workflow:

**Upload → Validate → Process → Forecast → Review → Export**

It provides:

- Automatic processing of original health, IoT and rainfall data
- Daily dengue locality-event extraction from Kerala IDSP reports
- Locality-level aggregation of IoT observations
- IMD gridded rainfall extraction
- Weekly Monday–Sunday feature preparation
- Dengue and environmental lag variables
- Seasonality features
- Chronological Random Forest regression
- One-week-ahead dengue forecasting
- Locality-level relative-risk classification
- Interactive GIS forecast/risk map
- Locality-wise forecast and environmental indicators
- Recent dengue and environmental trend plots
- Chronological model validation using MAE, RMSE and R²
- CSV export of the generated weekly dataset and forecast results

The application builds the modelling dataset internally, so a prepared master CSV is **not required for the normal source-data workflow**.

---

## 2. Study-area configuration

The current implementation is configured for ten modelling localities:

```text
Balaramapuram
Pallichal
Maranalloor
Malayinkeezhu
Kalliyoor
Vilappil
Vilavoorkal
Peringamala
Karode
Vellayani
```

These localities represent the current study configuration used for the dengue surveillance and forecasting workflow.

The dashboard itself is intentionally given a **general application title** so that the interface can be reused or extended to another study area without changing the main identity of the application.

---

## 3. Input data

### 3.1 Health / dengue surveillance data

The preferred input is the **original Kerala IDSP daily report** data:

- Individual `.pdf` files, or
- One `.zip` containing the daily IDSP PDF reports

Example:

```text
IDSP_Reports.zip
├── IDSP-Daily-Report-01.01.2025.pdf
├── IDSP-Daily-Report-02.01.2025.pdf
├── ...
└── IDSP-Daily-Report-31.01.2025.pdf
```

The application extracts:

- Report date
- Dengue information from the IDSP disease/locality section
- The `TVM` district row
- Configured study-area locality mentions
- Explicit locality-specific case counts when they are printed in the report

When a locality is listed without an explicit locality-specific number, the application treats it as one reported locality event.

The application can also accept compatible `.csv`/`.txt` health inputs, but the intended source workflow is based on the original IDSP daily reports.

**The processed Health Summary PDF is not required as the primary input.**

---

### 3.2 IoT environmental data

Upload:

- Individual sensor `.csv` files, or
- A `.zip` containing the sensor CSV files

The application detects commonly used fields such as:

- `Date/Time`
- `Device Name`
- `Temperature (C)`
- `Humidity (%)`
- `Rainfall (mm)` — optional
- `Pressure (hPa)` — optional
- `Wind Speed (Km/h)` — optional
- Latitude/longitude — when available

Sensor/locality identification is taken from the original filename or device name where possible, with a maintained sensor-site mapping used as a fallback.

Missing environmental variables do not automatically invalidate the dataset. They remain missing and are handled during feature preparation.

---

### 3.3 IMD gridded rainfall data

Upload:

- Individual `.grd` files, or
- A `.zip` containing the `.grd` files

The current project convention supports files such as:

```text
2024.grd
2025.grd
Rainfall_ind2024_rfp25.grd
Rainfall_ind2025_rfp25.grd
```

The application reads the supplied daily float32 `129 × 135` grids at `0.25°` resolution and extracts rainfall values for the configured study-area locality coordinates.

### Rainfall source handling

IoT rainfall is optional.

When weekly sensor rainfall is available, it is used for the weekly rainfall feature. When sensor rainfall is unavailable, the application falls back to IMD gridded rainfall where available. If neither source is available, rainfall remains missing rather than being assumed to be zero.

---

## 4. Processing workflow

```text
Original IDSP daily reports
        +
Original IoT sensor data
        +
IMD gridded rainfall
        ↓
Input validation & extraction
        ↓
Daily dengue locality events
Daily/locality IoT aggregation
Daily/locality IMD rainfall extraction
        ↓
Monday–Sunday weekly dataset
        ↓
Dengue lags
Environmental lags
Seasonality features
        ↓
Chronological Random Forest
        ↓
One-week-ahead forecast
        ↓
Relative-risk classification
        ↓
GIS risk map
Locality forecast table
Trend analysis
Model validation
        ↓
CSV export
```

---

## 5. Forecasting approach

### Modelling unit

The modelling unit is:

**Locality × Week**

The week is defined as:

**Monday–Sunday**

### Forecast horizon

The dashboard generates a:

**1-week / 7-day-ahead forecast**

The selected fully observed week is used as the forecast baseline, and the model estimates dengue cases for the following Monday–Sunday week.

The application does not allow an incomplete current week to be used as a standard forecast baseline, because doing so could require imputing the current dengue value and produce a misleading operational result.

### Model

The current implementation uses:

**Random Forest Regressor**

with the following general feature groups:

- Current-week dengue cases
- Previous dengue weeks / dengue lags
- Mean temperature
- Mean humidity
- Weekly rainfall
- Wind speed
- Environmental lag variables
- Seasonal week-of-year sine/cosine features

Missing feature values are handled through median imputation with missing-value indicators.

---

## 6. Relative-risk classification

The dashboard assigns four relative-risk categories:

```text
Low
Moderate
High
Very High
```

These classes are calculated from quartiles of the **unrounded predicted case counts across the modelled localities for the selected forecast week**.

Therefore:

- `Very High` means the locality falls within the highest prediction quartile.
- It does **not** represent a clinical probability.
- It does **not** guarantee that a particular number of cases will occur.
- The prediction is a model estimate and should be interpreted together with the surveillance and validation information.

Predicted case counts are displayed as whole numbers for practical interpretation, while the underlying continuous Random Forest prediction is retained internally.

---

## 7. Dashboard outputs

### 7.1 Data coverage & readiness

After processing, the dashboard provides an overview of:

- Dengue records
- IoT observations
- IMD rainfall observations
- Weekly locality records
- Date coverage
- Number of mapped localities
- IDSP reporting coverage
- Unmapped IoT sensor names, when present

The generated weekly modelling dataset can be downloaded as CSV.

---

### 7.2 One-week forecast

The user selects a fully observed baseline week and generates the forecast.

The dashboard provides summary indicators including:

- Current reported cases
- Predicted cases
- Number of high/very-high relative-risk localities
- Validation MAE
- Validation R²

Where the subsequent target week has already occurred, the dashboard can also display the observed target-week cases for comparison with the original forecast.

---

### 7.3 Interactive GIS risk map

The forecast map displays:

- Study-area locality boundaries
- Predicted relative-risk category
- Locality markers
- Predicted case counts
- Selected-locality highlighting
- Map zooming when geometry is available
- Locality labels and pop-up information

Selecting a locality updates the individual locality panel.

---

### 7.4 Locality details

For a selected locality, the dashboard displays:

- Predicted target-week cases
- Relative-risk category
- Observed target-week cases, when available
- Current-week dengue cases
- Previous-week dengue cases
- Weekly rainfall
- Rainfall source
- Mean temperature
- Mean humidity
- Wind speed

---

### 7.5 Surveillance and environmental trends

The dashboard provides recent trend plots for the selected locality.

Available comparisons include:

- Dengue cases only
- Dengue cases vs rainfall
- Dengue cases vs humidity

This allows recent surveillance patterns to be viewed alongside environmental variables.

---

### 7.6 Model performance

The application uses a **chronological holdout** rather than a random train/test split.

Reported metrics include:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — coefficient of determination

These metrics are calculated from the uploaded dataset and therefore change when the underlying data change.

---

## 8. Export options

The dashboard provides downloadable CSV outputs for:

### Generated weekly dataset

```text
NEMOM_weekly_features.csv
```

This contains the processed locality × week modelling records.

### Forecast results

```text
NEMOM_Dengue_1Week_Forecast.csv
```

This contains the locality-level forecast output, including predicted cases, relative risk and available comparison variables.

---

## 9. GIS boundary coverage

The current reference GeoJSON contains polygon geometry for **7 of the 10 configured modelling localities**.

Before using the complete 10-locality polygon map operationally, official polygon geometry should be added for:

- Peringamala
- Karode
- Vellayani

The forecasting and locality-level outputs can still be generated, but map polygon display is dependent on the available reference geometry.

---

## 10. Running the application

### Requirements

The application is built with:

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Folium
- streamlit-folium
- Plotly
- pdfplumber

### Ubuntu / Linux

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The application will normally open at the local Streamlit address shown in the terminal.

---

## 11. Demonstration / test package

A **demonstration package** is provided with this project so that the dashboard can be opened and tested without first collecting the complete original surveillance dataset.

### Demonstration package

```text
dengue_surveillance_forecast_demo.zip
```

The package contains:

```text
app.py
requirements.txt
NEMOM_DENGUE_ML_MASTER.csv
nemom_dashboard_geojson.json
DEMO_README.txt
```

The demonstration dataset is **synthetic** and is included only to demonstrate the application interface, processing-to-forecast workflow, GIS map, trend charts and model-validation sections.

### Run the demonstration

After extracting the ZIP:

```bash
cd dengue_surveillance_forecast_demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then, in the dashboard:

1. Open the application.
2. In the sidebar, enable **Use bundled demonstration dataset (for testing only)**.
3. Review the **Data coverage & readiness** section.
4. Select a forecast baseline week.
5. Click **Generate one-week forecast**.
6. Review the forecast summary, GIS risk map, locality details, trends and validation metrics.
7. Download the generated weekly dataset or forecast CSV if required.

### Important

The demonstration data are synthetic and **must not be interpreted as actual dengue surveillance observations or operational predictions**.

For real analysis, use the original IDSP, IoT and IMD source files described above.

---

## 12. Recommended project structure

For the normal application:

```text
project/
├── app.py
├── requirements.txt
├── nemom_dashboard_geojson.json
├── README.md
└── [source data uploaded through the dashboard]
```

For the demonstration package:

```text
dengue_surveillance_forecast_demo/
├── app.py
├── requirements.txt
├── NEMOM_DENGUE_ML_MASTER.csv
├── nemom_dashboard_geojson.json
└── DEMO_README.txt
```

---

## 13. Important interpretation notes

- The health target represents **reported IDSP locality events**, not an independent estimate of all infections.
- Missing daily IDSP reports are not silently treated as zero-report days; reporting coverage is tracked separately.
- Standard forecasting is restricted to fully observed Monday–Sunday baseline weeks.
- Future weather forecasts are not currently used. The model uses environmental information available in the uploaded/current dataset.
- Relative-risk classes are comparative dashboard categories, not clinical risk probabilities.
- Forecast performance should be assessed using the validation metrics before operational interpretation.
- The current implementation is configured for the study-area localities listed above and can be adapted to another study area by updating the locality configuration, sensor mapping, coordinates and reference GIS boundaries.

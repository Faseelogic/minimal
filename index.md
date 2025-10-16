---
layout: default
title: "Indian Residential Property Price Prediction"
permalink: /
---

# Indian Residential Property Price Prediction

[![View on GitHub](https://img.shields.io/badge/Repo-India_Housing_Model-blue?logo=github)](#) <!-- Replace # with actual repository URL -->

Accurately estimating property prices in India is challenging due to heterogeneity across cities, regulatory differences (RERA compliance, circle rates), rapid infrastructure growth (metro expansions, expressways), and cultural factors like Vaastu orientation. This project builds an end‑to‑end, explainable ML pipeline for predicting listing or transaction prices (and price per sq.ft) across major Indian urban centers.

---
## 1. Problem Statement
Predict the market price (INR) and price per square foot of residential properties (apartments/flats, independent houses) given structural, locational, regulatory, and amenity features. The solution must:
- Generalize across cities (Mumbai, Bengaluru, Delhi NCR, Pune, Hyderabad, Chennai, Kolkata, Ahmedabad, etc.).
- Support micro‑market granularity (locality / sub-locality / project).
- Provide interpretable feature attributions (SHAP / PDP) to guide valuation.
- Offer calibrated uncertainty (prediction intervals / conformal quantiles) for decision support.

---
## 2. Data Sources & Schema
Potential aggregated sources (ensure licensing & ToS compliance):
- Portals: MagicBricks, 99acres, Housing.com (scraped or via permitted APIs).
- Government: State registration data (where open), RERA project registry (project start, developer, carpet vs built‑up ratios), stamp duty & registration fee schedules.
- Circle Rates / Guidance Values: Official PDFs per zone converted to structured data.
- Geo: OpenStreetMap (distance to nearest metro station, railway, arterial road, mall, school, hospital, park, lake), elevation, flood zone risk layers (where available).
- Macroeconomic: RBI CPI / housing inflation indices for temporal price normalization.

Example unified columns:
| Column | Description |
|--------|------------|
| City | Metro city name |
| Locality | Micro-market / area |
| SubLocality | Finer granularity / project cluster |
| ProjectName | Developer project name |
| Developer | Builder / brand rating |
| PropertyType | Flat / IndependentHouse / Villa / Plot |
| BHK | Bedrooms (integer) |
| Bathrooms | Total bathrooms |
| BuiltUpArea | Built-up area (sq.ft) |
| CarpetArea | Carpet area (sq.ft) |
| Price | Asking / transaction price (INR) |
| PricePerSqFt | Price / BuiltUpArea |
| Floor | Current floor number |
| TotalFloors | Total floors in building |
| AgeOfProperty | Years since completion |
| Facing | Cardinal direction (E/W/N/S/NE/SW etc.) |
| VaastuScore | Derived score (orientation + ventilation proxies) |
| Furnishing | Unfurnished/Semi/Fully |
| Parking | Covered / Open / None (count) |
| Amenities | Set of amenities (pool, gym, clubhouse, security) |
| RERARegistered | Boolean |
| OccupancyCertificate | Boolean |
| Latitude/Longitude | Geo coordinates |
| DistanceMetro | Meters to nearest metro station |
| DistanceSchool | Distance to nearest reputed school |
| CircleRate | Govt. guideline value per sq.ft for zone |
| TransactionDate | Date of listing or sale |
| InflationIndex | CPI housing factor |
| AdjustedPrice | Price adjusted for inflation & circle rate normalization |
| LegalFlags | Encumbrance / disputed / clear |

Target(s): `Price` (regression), auxiliary `PricePerSqFt`, optional quantile targets.

---
## 3. Data Cleaning (India-Specific Nuances)
1. Standardize area units (some listings use sq.meter, sq.yard). Convert everything to sq.ft using precise conversion factors.
2. Resolve BHK inconsistencies (Studio → 1, "1 RK" → 1 but mark `HasKitchen=0/1`).
3. Carpet vs built-up: If carpet missing but ratio typical for city/project (e.g., Mumbai ~65–70%), impute using learned ratio per project category.
4. Price anomalies: Flag listings with `PricePerSqFt` < 0.4 * median_circle_rate or > 3.5 * upper_quantile; treat as outliers (remove or cap).
5. Deduplicate multi-posted listings (match by ProjectName + BHK + Floor + area ranges + fuzzy price).
6. Extract structured amenities from free-text using keyword & embedding similarity.
7. Temporal normalization: compute `AdjustedPrice = Price / InflationIndex` to compare across years.
8. Geocoding validation: Reverse geocode lat/lon; discard points falling outside city polygon (shapely / geopandas containment).
9. Developer brand consolidation (merge spelling variants), compute `DeveloperReputationScore` from historical delivery delays & RERA complaints.

---
## 4. Feature Engineering (India Context)
- Price normalization: `RelPrice = PricePerSqFt / CircleRate` (captures premium/discount vs guideline).
- Spatial features: H3 geohash index (resolution ~8) for locality encoding, distance gradients (metro, IT park, CBD, airport, waterfront).
- Accessibility indices: Weighted score combining distances (inverse-squared) to key POIs.
- Infrastructure growth: Binary flags for future metro line within X meters (from published alignment data).
- Regulatory: `IsCompliant = RERARegistered & OccupancyCertificate`; penalty feature if not compliant.
- Vaastu orientation: Map `Facing` to azimuth, derive boolean `NorthEastPreferred` etc.
- Density proxy: `Floor/TotalFloors` + building height category.
- Area efficiency: `CarpetEfficiency = CarpetArea / BuiltUpArea`.
- Amenity richness: Count + TF-IDF embedding cluster → `AmenityCluster`.
- Interaction examples: `DeveloperReputationScore * RelPrice`, `CarpetEfficiency * BHK`, `H3Cluster * CircleRate`.
- Temporal: Year, Quarter, Season (Monsoon vs Non-monsoon), pre/post major infrastructure announcements.
- Outlier robust transforms: log1p on skewed (`Price`, `BuiltUpArea`), Yeo-Johnson where zero values present.

Feature selection: Boruta for tree models; SHAP importance pruning; stability selection across temporal splits.

---
## 5. Modeling Approach
Baseline & advanced models:
1. Hedonic Regression (linear with regularization) for interpretability.
2. CatBoost (handles categorical efficiently, robust to high cardinality like `ProjectName`).
3. LightGBM / XGBoost (gradient boosting for accuracy).
4. Deep Tabular: TabNet / FT-Transformer (optional experiment).
5. Geo-ensemble: Train separate city models then stack meta-learner.
6. Quantile Regression (LightGBM with quantile objective or conformal prediction) for intervals.

Hyperparameter tuning: Optuna with early stopping; city-aware folds (GroupKFold by `City`) to avoid leakage; time-based validation for drift handling.

Bias & fairness checks: Compare residuals across property types & cities; ensure no systematic undervaluation for certain localities.

---
## 6. Evaluation Metrics
- RMSE & MAE on `log(Price)` and raw INR.
- Median Absolute Percentage Error (MdAPE) (robust for skew).  
- MAPE (reported but watch high variance for low-priced units).  
- R² and adjusted R² for hedonic model.  
- Coverage of prediction intervals (target 90% interval containing ~90% actual).
- City-specific error dashboards (Mumbai vs Bengaluru vs Tier-2).  
- Premium detection accuracy: classify `RelPrice > 1.25` (F1 score).

Diagnostics: SHAP summary, PDP on `RelPrice`, error heatmap across H3 geohash, temporal drift plot.

---
## 7. Sample (Pseudo) Results (Replace With Actual)
| Model | RMSE (₹) | MdAPE (%) | Interval Coverage | Notes |
|-------|----------|-----------|------------------|-------|
| Hedonic Ridge | 12.5L | 15.2 | 86% | High interpretability |
| CatBoost | 9.8L | 11.0 | 89% | Best single model |
| Ensemble (CatBoost + LightGBM + Hedonic) | 9.4L | 10.6 | 91% | Slight gain & better calibration |

SHAP Top Drivers (example): CarpetArea, City_Bangalore, DistanceMetro, RelPrice, DeveloperReputationScore, AmenitiesCount, VaastuScore.

---
## 8. Key Insights (India)
- Circle rate premiums vary strongly with infrastructure proximity (metro, tech hub); models capture uplift gradient.
- Carpet efficiency & Vaastu orientation have modest but consistent impact in certain cities (Bengaluru, Chennai). 
- Developer reputation reduces uncertainty (narrower prediction intervals) — established builders have more stable pricing.
- Regulatory compliance (RERA + OccupancyCertificate) correlates with ~8–15% premium after controlling for locality.
- RelPrice (vs circle rate) is a powerful standardized feature enabling cross-city comparison.

---
## 9. Reproducible Pipeline
Suggested folder layout:
```
data/
  raw/               # Scraped / sourced datasets
  interim/           # After cleaning & merging
  processed/         # Final model-ready parquet files
features/            # Cached feature matrices
models/              # Serialized model artifacts (.pkl / .cbm / .json)
notebooks/           # EDA & experiments
configs/             # YAML configs for training
reports/             # HTML/markdown model cards & SHAP plots
scripts/             # ETL, feature engineering, training, evaluation
geo/                 # Shapefiles, circle rate zones
```

Example command flow:
```bash
python scripts/ingest.py --sources configs/sources.yaml
python scripts/clean.py --input data/raw --output data/interim
python scripts/features.py --input data/interim --output data/processed --geo geo/
python scripts/train.py --config configs/catboost.yaml
python scripts/evaluate.py --model models/catboost.cbm --data data/processed/train.parquet
python scripts/explain.py --model models/catboost.cbm --sample 5000 --out reports/shap_summary.html
python scripts/intervals.py --model models/catboost.cbm --data data/processed/holdout.parquet
```

---
## 10. Sample Code Snippet (Simplified Hedonic + CatBoost)
```python
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from catboost import CatBoostRegressor

df = pd.read_csv("data/processed/india_housing.csv")

# Derived features
df['CarpetEfficiency'] = df['CarpetArea'] / df['BuiltUpArea'].replace(0, np.nan)
df['PricePerSqFt'] = df['Price'] / df['BuiltUpArea']
df['RelPrice'] = df['PricePerSqFt'] / df['CircleRate']
df['AgeOfProperty'] = (pd.to_datetime(df['TransactionDate']).dt.year - df['CompletionYear']).clip(lower=0)
df['FloorRatio'] = df['Floor'] / df['TotalFloors'].replace(0, np.nan)

target = np.log1p(df['Price'])  # log transform target

# Select columns
numeric = ['BuiltUpArea', 'CarpetArea', 'CarpetEfficiency', 'BHK', 'Bathrooms', 'AgeOfProperty', 'FloorRatio', 'DistanceMetro', 'DistanceSchool', 'CircleRate', 'RelPrice']
categorical = ['City', 'Locality', 'ProjectName', 'Developer', 'Facing', 'Furnishing', 'RERARegistered']

X = df[numeric + categorical]

# Train/validation split (group by city to prevent leakage)
gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(gkf.split(X, target, groups=df['City']))
X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]

cat_features = [X.columns.get_loc(c) for c in categorical]

model = CatBoostRegressor(
	depth=8,
	learning_rate=0.05,
	iterations=2000,
	loss_function='RMSE',
	eval_metric='RMSE',
	early_stopping_rounds=100,
	verbose=200
)
model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features)

pred_val = model.predict(X_val)
rmse = mean_squared_error(y_val, pred_val, squared=False)
mae = mean_absolute_error(y_val, pred_val)
mdape = np.median(np.abs((np.expm1(y_val) - np.expm1(pred_val)) / np.expm1(y_val))) * 100
print("Validation RMSE (log space):", rmse)
print("MAE (log space):", mae)
print("Median APE (%):", mdape)
```

---
## 11. Uncertainty & Intervals
- Use conformal prediction: calibrate residuals on validation fold to produce upper/lower bounds.  
- Quantile CatBoost / LightGBM: train for 0.1 and 0.9 quantiles to derive interval.  
- Report coverage per city to monitor calibration drift.

---
## 12. Next Steps
- Integrate time-series price trend modeling (ARIMA / Prophet layered with ML residuals).
- Add rental yield features (rent listings) to derive investment attractiveness.
- Incorporate environmental risk (flood plains, air quality index) for long-term valuation.
- Deploy FastAPI microservice with caching (Redis) and geospatial nearest-neighbor fallback when confidence low.
- Continuous retraining pipeline (Airflow / Prefect) with data drift detectors (Evidently).

---
## 13. Contact / Attribution
Add your LinkedIn, email, and project repository link. Cite any third-party data sources & respect scraping terms. Replace placeholders above before publishing.

<small>NOTE: All example values & metrics are illustrative; customize with your real data and results.</small>


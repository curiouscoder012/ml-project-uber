# Uber India Ride Bookings — ML Project

End-to-end ML pipeline on the NCR/Uber ride bookings dataset covering data cleaning,
EDA, two supervised models, and two unsupervised models — all on the same underlying
data, viewed from four different angles.

## Dataset

- **Source:** NCR/Uber Ride Bookings dataset (`ncr_ride_bookings.csv`)
- **Size:** 150,000 rows × 21 columns
- **Fields:** Booking ID, Customer ID, Date/Time, Booking Status, Vehicle Type, Pickup/Drop
  Location, Avg VTAT, Avg CTAT, Booking Value, Ride Distance, Driver Ratings, Customer
  Rating, Payment Method, plus cancellation/incompletion reason fields

**Booking outcome distribution:**

| Status | Count | Share |
|---|---|---|
| Completed | 93,000 | 62% |
| Cancelled by Driver | 27,000 | 18% |
| No Driver Found | 10,500 | 7% |
| Cancelled by Customer | 10,500 | 7% |
| Incomplete | 9,000 | 6% |

**Key structural fact used throughout:** `Booking Value`, `Ride Distance`, `Driver
Ratings`, and `Customer Rating` are only populated when `Booking Status == "Completed"`.
Using them to predict cancellation would be data leakage, so the classifier below is
restricted to features known *before* the ride happens.

## Pipeline

1. **Load & clean** — strip quoted ID strings, merge Date + Time into a proper
   datetime, derive `Hour`, `DayOfWeek`, `Month`, `IsWeekend`.
2. **EDA** — outcome distribution, vehicle type / payment method mix, hourly ride
   volume, cancellation rate by vehicle type, cancellation reasons, fare/distance/rating
   distributions for completed rides.
3. **Feature engineering** — two separate feature sets: pre-ride features (time,
   vehicle type, VTAT, pickup/drop area) for the classifier, and post-ride features
   (+ distance, CTAT) for the regressor. Pickup/Drop Location (high-cardinality) is
   collapsed to the top 30 areas + `"Other"`.
4. **Classification** — predict whether a booking completes vs. cancels/fails.
5. **Regression** — predict fare (`Booking Value`) for completed rides.
6. **KMeans + PCA** — unsupervised segmentation of completed rides.
7. **DBSCAN** — density-based anomaly detection on fare vs. distance.

## Results

### 1. Cancellation classifier (Random Forest)

Predicts `Target_Completed` from pre-ride-only features (vehicle type, hour, day,
month, weekend flag, pickup/drop area, Avg VTAT, payment method). Trained on 102,000
rows (rows missing VTAT/payment method dropped), 80/20 stratified split, `class_weight="balanced"`.

| Model | ROC-AUC |
|---|---|
| Random Forest (300 trees, max_depth=12) | 0.6853 |
| Logistic Regression baseline | 0.6919 |

Test-set classification report (Random Forest, threshold 0.5):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Not Completed | 0.14 | 0.97 | 0.24 | 1,800 |
| Completed | 0.99 | 0.40 | 0.57 | 18,600 |
| **Accuracy** | | | **0.45** | 20,400 |

With `class_weight="balanced"`, the model trades raw accuracy for recall on the
minority class (catching likely-to-fail bookings rather than just being right on
average on a ~91%-Completed test set). ROC-AUC (~0.69, threshold-independent) is the
number to trust here — it shows real but moderate signal: pre-ride features
(vehicle type, time of day, VTAT) partially predict completion, but most of what
actually drives a cancellation (driver mood, a customer changing their mind) isn't
captured in this data. Notably, plain Logistic Regression matched the Random Forest
here — the relationship isn't complex enough to reward a heavier model.

### 2. Fare regression (XGBoost)

Predicts `Booking Value` for completed rides from `Ride Distance`, Avg VTAT/CTAT,
hour, driver/customer rating, vehicle type, payment method, weekend flag.
`XGBRegressor(n_estimators=500, max_depth=2, learning_rate=0.1)`.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost | 277.56 | 388.79 | 0.0524 |
| Linear Regression baseline | — | — | 0.0547 |

**Honest read:** R² is close to 0 for both models, and raw correlation between fare
and distance/VTAT/CTAT is ~0.005–0.006 — essentially zero. This isn't a modeling
failure (300+ trees aren't missing a signal that's actually there); it means
`Booking Value` in this dataset is largely decoupled from trip characteristics,
consistent with the fares being synthetically generated rather than computed as
`base_fare + rate × distance + surge` the way real fares are.

Re-running the same regression on the DBSCAN-filtered "normal" subset (anomalous
fares removed, see below) tightens the errors slightly but doesn't change the
underlying conclusion:

| Model (anomalies removed) | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost | 245.41 | 307.40 | 0.0606 |
| Linear Regression baseline | — | — | 0.0632 |

### 3. KMeans segmentation + PCA

Clustered completed rides (Ride Distance, Booking Value, Avg VTAT, Avg CTAT, Driver
Ratings, Customer Rating; standardized) after an elbow/silhouette scan over k = 2–8.
Fit with **k = 7**:

| Cluster | Distance | Value | VTAT | CTAT | Driver Rating | Customer Rating | Count |
|---|---|---|---|---|---|---|---|
| 0 | 25.69 | 460.95 | 8.55 | 29.96 | 4.27 | 3.51 | 9,740 |
| 1 | 26.33 | **1,600.31** | 8.47 | 29.88 | 4.26 | 4.44 | 5,273 |
| 2 | 34.78 | 441.64 | 11.42 | 37.24 | 4.36 | 4.52 | 16,510 |
| 3 | 14.05 | 440.31 | 5.69 | 35.51 | 4.36 | 4.51 | 16,578 |
| 4 | 18.03 | 438.52 | 11.32 | 22.53 | 4.36 | 4.52 | 16,897 |
| 5 | 25.97 | 452.59 | 8.62 | 30.07 | 3.43 | 4.48 | 11,556 |
| 6 | 37.52 | 431.83 | 5.47 | 25.07 | 4.35 | 4.53 | 16,446 |

Cluster 1 stands out sharply — same distance/pickup profile as clusters 0/5 but ~3.5×
the fare, i.e. a premium/high-surge fare group rather than a longer-trip group
(consistent with fare being decoupled from distance). Clusters 0 and 5 have visibly
lower ratings than the rest (customer rating 3.51 and driver rating 3.43
respectively) — likely a "rough ride experience" segment.

2D PCA is used only for visualization, not clustering: the first two components
explain **33.6%** of variance (16.8% + 16.7%), so the scatter plot is a partial view.
Vehicle type mix is nearly uniform across all seven clusters (~25% Auto, 20% Go Mini,
18% Go Sedan, 15% Bike, 12% Premier Sedan, 7% eBike, 3% Uber XL) — clusters are driven
by fare/rating/timing patterns, not vehicle choice.

### 4. DBSCAN anomaly detection

Density-based outlier detection on the 2D fare-vs-distance relationship
(`eps=0.15`, `min_samples=50`, standardized features):

- **1,689 rides flagged as anomalies (1.82%)**
- 1 dense "normal" cluster identified

Flagged rides are dominated by very high fares (₹4,000+) that don't scale with
distance — e.g. ₹4,277 for an 8.66 km Go Mini ride, ₹4,220 for a 10.11 km Auto ride —
plausible surge-pricing spikes, data entry errors, or cases worth a manual/fraud
review.

## Tech stack

`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (RandomForest,
LogisticRegression, LinearRegression, KMeans, DBSCAN, PCA, `ColumnTransformer` /
`Pipeline`), `xgboost`. `RANDOM_STATE = 42` fixed throughout for reproducibility.

## How to run

1. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn xgboost`
2. Update the CSV path in the data-loading cell to point to your local copy of
   `ncr_ride_bookings.csv`.
3. Run all cells top to bottom — later cells (regression, clustering) depend on
   dataframes built earlier in the notebook.

## Possible extensions

- Tune DBSCAN's `eps`/`min_samples` via a k-distance plot instead of fixed values.
- Try `AgglomerativeClustering` and compare cluster structure to KMeans via a dendrogram.
- Add `GridSearchCV`/`RandomizedSearchCV` for the classifier and regressor.
- Engineer pickup–drop **location pairs** binned by NCR sub-region (route-level demand)
  instead of raw locality names, to avoid extreme cardinality.
- Replace plain feature importances with SHAP values for per-prediction explanations
  on the cancellation classifier.
- Cross-check the fare-regression conclusion against a real fare-generating dataset
  (e.g. `yasserh/uber-fares-dataset`) to confirm the near-zero R² here is a property
  of this dataset's (likely synthetic) fares, not the modeling approach.

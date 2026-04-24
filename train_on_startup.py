"""
train_on_startup.py
Chạy khi Render build để tạo model .pkl đúng version môi trường.
Không cần upload .pkl lên GitHub nữa.
"""
import os, json, numpy as np, pandas as pd, joblib, warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Nếu model đã tồn tại thì bỏ qua
if os.path.exists(os.path.join(MODEL_DIR, "model_rain_clf.pkl")):
    print("Models already exist – skipping training")
    exit(0)

print("Training models from scratch...")

# ── Tạo dữ liệu tổng hợp ─────────────────────────────────────────────────────
np.random.seed(42)
N = 8760
from datetime import datetime, timedelta
start = datetime(2024, 1, 1)
timestamps = [start + timedelta(hours=i) for i in range(N)]

rows = []
for ts in timestamps:
    hour = ts.hour; month = ts.month
    season_temp = 20 + 10 * np.sin((month - 3) / 12 * 2 * np.pi)
    daily_var   = 5  * np.sin((hour - 6) / 24 * 2 * np.pi)
    temp = round(float(season_temp + daily_var + np.random.normal(0, 1.5)), 1)
    hum  = round(float(np.clip(85 - (temp - 25) * 0.8 + np.random.normal(0, 5), 30, 99)), 1)
    is_rainy = 1 if 5 <= month <= 9 else 0
    base_p   = 0.1 + 0.35 * is_rainy + 0.003 * max(0, hum - 60)
    noise    = np.random.normal(0, 0.08)
    p1  = float(np.clip(base_p + noise, 0, 1))
    p3  = float(np.clip(base_p + noise * 0.9 + np.random.normal(0, 0.05), 0, 1))
    p6  = float(np.clip(base_p + noise * 0.7 + np.random.normal(0, 0.07), 0, 1))
    p12 = float(np.clip(base_p + noise * 0.5 + np.random.normal(0, 0.10), 0, 1))
    def rmm(p): return round(float(np.random.exponential(3.5)), 2) if np.random.rand() < p else 0.0
    rf1, rf3, rf6, rf12 = rmm(p1), rmm(p3), rmm(p6), rmm(p12)
    actual = round(max(0, 0.5*rf1 + 0.25*rf3 + 0.15*rf6 + 0.1*rf12 + np.random.normal(0, 0.5)), 2)
    rows.append(dict(timestamp=ts, temperature_c=temp, humidity_rh=hum,
                     rain_prob_1h=round(p1,3), rain_prob_3h=round(p3,3),
                     rain_prob_6h=round(p6,3), rain_prob_12h=round(p12,3),
                     rain_forecast_1h_mm=rf1, rain_forecast_3h_mm=rf3,
                     rain_forecast_6h_mm=rf6, rain_forecast_12h_mm=rf12,
                     rain_actual_mm=actual))

df = pd.DataFrame(rows)
df["hour"]  = df["timestamp"].dt.hour
df["month"] = df["timestamp"].dt.month

for lag in [1, 3, 6]:
    df[f"temp_lag{lag}"]  = df["temperature_c"].shift(lag)
    df[f"hum_lag{lag}"]   = df["humidity_rh"].shift(lag)
    df[f"rain_lag{lag}"]  = df["rain_actual_mm"].shift(lag)

df["temp_rolling3"] = df["temperature_c"].rolling(3).mean()
df["hum_rolling3"]  = df["humidity_rh"].rolling(3).mean()
df["rain_rolling6"] = df["rain_actual_mm"].rolling(6).sum()
df["hour_sin"]  = np.sin(2*np.pi * df["hour"]  / 24)
df["hour_cos"]  = np.cos(2*np.pi * df["hour"]  / 24)
df["month_sin"] = np.sin(2*np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2*np.pi * df["month"] / 12)
df["heat_index"] = df["temperature_c"] + 0.33 * (
    df["humidity_rh"]/100 * 6.105 * np.exp(17.27*df["temperature_c"]/(237.7+df["temperature_c"]))) - 4
df = df.dropna().reset_index(drop=True)

FEATURES = [
    "temperature_c","humidity_rh",
    "rain_prob_1h","rain_prob_3h","rain_prob_6h","rain_prob_12h",
    "rain_forecast_1h_mm","rain_forecast_3h_mm","rain_forecast_6h_mm","rain_forecast_12h_mm",
    "temp_lag1","temp_lag3","hum_lag1","hum_lag3","rain_lag1","rain_lag3","rain_lag6",
    "temp_rolling3","hum_rolling3","rain_rolling6",
    "hour_sin","hour_cos","month_sin","month_cos","heat_index"
]
WATER_FEAT = ["temperature_c","humidity_rh","rain_prob_1h","rain_prob_3h",
              "hour_sin","hour_cos","month_sin","month_cos","heat_index"]
RAIN_THRESH = 0.5

df["rain_label"] = (df["rain_actual_mm"] > RAIN_THRESH).astype(int)
X = df[FEATURES]

# Scaler
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# Model 1: Classifier
clf = RandomForestClassifier(n_estimators=100, max_depth=10,
                              min_samples_leaf=5, class_weight="balanced",
                              random_state=42, n_jobs=-1)
clf.fit(X_s, df["rain_label"])
print(f"Classifier trained. Rain ratio: {df['rain_label'].mean():.2%}")

# Model 2: Regressor
reg = GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                 learning_rate=0.05, subsample=0.8, random_state=42)
reg.fit(X_s, df["rain_actual_mm"])
print("Regressor trained.")

# Model 3: Water
def water_target(row):
    base = 83
    temp_bonus = max(0, (row["temperature_c"] - 28) * 15)
    hum_bonus  = 10 if row["humidity_rh"] < 40 else 0
    uvi_proxy  = max(0, (row["temperature_c"] - 25) * 0.3)
    uvi_bonus  = min(uvi_proxy * 5, 50)
    rain_disc  = -10 if row["rain_prob_1h"] > 0.5 else 0
    return round((base + temp_bonus + hum_bonus + uvi_bonus + rain_disc) / 50) * 50

df["water_target_ml"] = df.apply(water_target, axis=1)
Xw = df[WATER_FEAT]
scaler_w = StandardScaler()
Xw_s = scaler_w.fit_transform(Xw)
water_model = Ridge(alpha=1.0)
water_model.fit(Xw_s, df["water_target_ml"])
print("Water model trained.")

# Export
joblib.dump(clf,         os.path.join(MODEL_DIR, "model_rain_clf.pkl"))
joblib.dump(reg,         os.path.join(MODEL_DIR, "model_rain_reg.pkl"))
joblib.dump(water_model, os.path.join(MODEL_DIR, "model_water.pkl"))
joblib.dump(scaler,      os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(scaler_w,    os.path.join(MODEL_DIR, "scaler_water.pkl"))

meta = {
    "version": "1.0.0",
    "trained_on": str(pd.Timestamp.now()),
    "n_samples": len(df),
    "features": FEATURES,
    "water_features": WATER_FEAT,
    "rain_threshold_mm": RAIN_THRESH,
    "rain_warning_prob": 0.40,
    "water_clip_min": 50,
    "water_clip_max": 500,
    "models": {
        "rain_clf":     "model_rain_clf.pkl",
        "rain_reg":     "model_rain_reg.pkl",
        "water":        "model_water.pkl",
        "scaler":       "scaler.pkl",
        "scaler_water": "scaler_water.pkl",
    }
}
with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("✅ All models saved to models/")

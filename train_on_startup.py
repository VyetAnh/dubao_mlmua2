"""
train_on_startup.py
Chạy khi Render build – không dùng pandas, chỉ dùng numpy + sklearn
"""
import os, json, numpy as np, joblib, warnings
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

if os.path.exists(os.path.join(MODEL_DIR, "model_rain_clf.pkl")):
    print("Models already exist – skipping training")
    exit(0)

print("Training models from scratch (no pandas)...")

np.random.seed(42)
N = 8760  # 1 năm, mỗi giờ

# ── Tạo dữ liệu tổng hợp bằng numpy thuần ────────────────────────────────────
hours  = np.arange(N)
months = ((hours // 720) % 12) + 1   # ~30 ngày/tháng
hour_of_day = hours % 24

season_temp = 20 + 10 * np.sin((months - 3) / 12 * 2 * np.pi)
daily_var   =  5 * np.sin((hour_of_day - 6) / 24 * 2 * np.pi)
temp = np.round(season_temp + daily_var + np.random.normal(0, 1.5, N), 1)
hum  = np.clip(np.round(85 - (temp - 25) * 0.8 + np.random.normal(0, 5, N), 1), 30, 99)

is_rainy = ((months >= 5) & (months <= 9)).astype(float)
base_p   = 0.1 + 0.35 * is_rainy + 0.003 * np.clip(hum - 60, 0, None)
noise    = np.random.normal(0, 0.08, N)

p1  = np.clip(base_p + noise,                              0, 1)
p3  = np.clip(base_p + noise*0.9 + np.random.normal(0, 0.05, N), 0, 1)
p6  = np.clip(base_p + noise*0.7 + np.random.normal(0, 0.07, N), 0, 1)
p12 = np.clip(base_p + noise*0.5 + np.random.normal(0, 0.10, N), 0, 1)

def rain_mm(p):
    mask = np.random.rand(N) < p
    vals = np.random.exponential(3.5, N)
    return np.where(mask, np.round(vals, 2), 0.0)

rf1  = rain_mm(p1)
rf3  = rain_mm(p3)
rf6  = rain_mm(p6)
rf12 = rain_mm(p12)
actual = np.round(np.clip(0.5*rf1 + 0.25*rf3 + 0.15*rf6 + 0.1*rf12
                          + np.random.normal(0, 0.5, N), 0, None), 2)

# Lag features
def lag(arr, k):
    out = np.empty_like(arr)
    out[:k] = arr[0]
    out[k:] = arr[:-k]
    return out

def rolling_mean(arr, k):
    out = np.empty_like(arr)
    for i in range(len(arr)):
        start = max(0, i - k + 1)
        out[i] = arr[start:i+1].mean()
    return out

def rolling_sum(arr, k):
    out = np.empty_like(arr)
    for i in range(len(arr)):
        start = max(0, i - k + 1)
        out[i] = arr[start:i+1].sum()
    return out

temp_lag1 = lag(temp, 1); temp_lag3 = lag(temp, 3)
hum_lag1  = lag(hum,  1); hum_lag3  = lag(hum,  3)
rain_lag1 = lag(actual, 1); rain_lag3 = lag(actual, 3); rain_lag6 = lag(actual, 6)
temp_r3   = rolling_mean(temp, 3)
hum_r3    = rolling_mean(hum,  3)
rain_r6   = rolling_sum(actual, 6)

hour_sin  = np.sin(2*np.pi * hour_of_day / 24)
hour_cos  = np.cos(2*np.pi * hour_of_day / 24)
month_sin = np.sin(2*np.pi * months / 12)
month_cos = np.cos(2*np.pi * months / 12)
heat_idx  = temp + 0.33*(hum/100*6.105*np.exp(17.27*temp/(237.7+temp))) - 4

# ── Build feature matrix ───────────────────────────────────────────────────────
X = np.column_stack([
    temp, hum,
    p1, p3, p6, p12,
    rf1, rf3, rf6, rf12,
    temp_lag1, temp_lag3,
    hum_lag1,  hum_lag3,
    rain_lag1, rain_lag3, rain_lag6,
    temp_r3, hum_r3, rain_r6,
    hour_sin, hour_cos, month_sin, month_cos,
    heat_idx
])

FEATURES = [
    "temperature_c","humidity_rh",
    "rain_prob_1h","rain_prob_3h","rain_prob_6h","rain_prob_12h",
    "rain_forecast_1h_mm","rain_forecast_3h_mm","rain_forecast_6h_mm","rain_forecast_12h_mm",
    "temp_lag1","temp_lag3","hum_lag1","hum_lag3",
    "rain_lag1","rain_lag3","rain_lag6",
    "temp_rolling3","hum_rolling3","rain_rolling6",
    "hour_sin","hour_cos","month_sin","month_cos","heat_index"
]

WATER_FEAT = [
    "temperature_c","humidity_rh",
    "rain_prob_1h","rain_prob_3h",
    "hour_sin","hour_cos","month_sin","month_cos","heat_index"
]
WATER_IDX = [FEATURES.index(f) for f in WATER_FEAT]

RAIN_THRESH = 0.5
y_clf = (actual > RAIN_THRESH).astype(int)
y_reg = actual

# Water target
base      = 83
tb        = np.clip((temp - 28) * 15, 0, None)
hb        = np.where(hum < 40, 10, 0)
uv_proxy  = np.clip((temp - 25) * 0.3, 0, None)
uv_b      = np.clip(uv_proxy * 5, 0, 50)
rain_disc = np.where(p1 > 0.5, -10, 0)
y_water   = np.round((base + tb + hb + uv_b + rain_disc) / 50) * 50

# Scale
scaler   = StandardScaler()
X_s      = scaler.fit_transform(X)

scaler_w = StandardScaler()
Xw_s     = scaler_w.fit_transform(X[:, WATER_IDX])

# Train
print("  → Classifier...")
clf = RandomForestClassifier(n_estimators=100, max_depth=10,
                              min_samples_leaf=5, class_weight="balanced",
                              random_state=42, n_jobs=-1)
clf.fit(X_s, y_clf)

print("  → Regressor...")
reg = GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                 learning_rate=0.05, subsample=0.8,
                                 random_state=42)
reg.fit(X_s, y_reg)

print("  → Water model...")
water_model = Ridge(alpha=1.0)
water_model.fit(Xw_s, y_water)

# Save
joblib.dump(clf,         os.path.join(MODEL_DIR, "model_rain_clf.pkl"))
joblib.dump(reg,         os.path.join(MODEL_DIR, "model_rain_reg.pkl"))
joblib.dump(water_model, os.path.join(MODEL_DIR, "model_water.pkl"))
joblib.dump(scaler,      os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(scaler_w,    os.path.join(MODEL_DIR, "scaler_water.pkl"))

meta = {
    "version":           "1.0.0",
    "trained_on":        datetime.utcnow().isoformat(),
    "n_samples":         N,
    "features":          FEATURES,
    "water_features":    WATER_FEAT,
    "rain_threshold_mm": RAIN_THRESH,
    "rain_warning_prob": 0.40,
    "water_clip_min":    50,
    "water_clip_max":    500,
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

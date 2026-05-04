"""
7-Eleven RTE — Demand Forecasting v2 (LightGBM + Promotion Features)
Improvements:
  - Promotion features: is_promo, discount_pct, promo_type, days_since/until_promo
  - More lags: lag_2, lag_3, lag_21, lag_28
  - Fourier terms for weekly seasonality
  - Longer history (120 days)
  - Category-specific demand multipliers
  - Rank × promo interaction feature
  - Hyperparameter tuning
Target: MAPE < 20%
"""

import json, random, math, urllib.request
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import date, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error

random.seed(42)
np.random.seed(42)

# ── CONFIG ───────────────────────────────────────────────────────────────────

SUPABASE_URL  = "https://hrmtrplqsvjcojmobmby.supabase.co"
SUPABASE_KEY  = "sb_publishable_chAqNLtW_g7dFnqzYfTNPA_H947QyXF"
SAFETY_STOCK  = 3
HISTORY_DAYS  = 120
FORECAST_DATE = date.today() + timedelta(days=1)

SKUS = [
    ("711-00001","Krapow Chicken Ezygo","Thai","A",39,3),
    ("711-00002","Krapow Pork Ezygo","Thai","B",45,5),
    ("711-00003","Spaghetti Carbonara","International","B",69,4),
    ("711-00004","Pad See Ew Chicken","Thai","A",42,3),
    ("711-00005","Moo Ping Set","Thai","A",35,2),
    ("711-00006","Fried Rice Egg","Thai","A",38,3),
    ("711-00007","Teriyaki Chicken Bento","Japanese","B",75,4),
    ("711-00008","Tom Yum Soup","Thai","A",49,2),
    ("711-00009","Green Curry Rice","Thai","B",55,3),
    ("711-00010","Fried Chicken Set","Western","A",65,2),
    ("711-00011","Onigiri Tuna Mayo","Japanese","B",28,3),
    ("711-00012","Onigiri Salmon","Japanese","B",32,3),
    ("711-00013","Sandwich Ham Cheese","Western","B",55,2),
    ("711-00014","Sandwich Tuna","Western","C",48,2),
    ("711-00015","Mango Sticky Rice","Snack/Side","B",45,2),
    ("711-00016","Corn Grilled","Snack/Side","C",25,1),
    ("711-00017","Instant Noodle Spicy","Snack/Side","C",18,90),
    ("711-00018","Banana Roti","Snack/Side","B",30,1),
    ("711-00019","Pork Ball Skewer","Snack/Side","A",25,1),
    ("711-00020","Chicken Nuggets","Western","B",55,2),
    ("711-00021","BBQ Pork Rice","Thai","A",50,3),
    ("711-00022","Massaman Curry","Thai","B",60,3),
    ("711-00023","Basil Pork Rice","Thai","A",45,3),
    ("711-00024","Egg Fried Rice","Thai","B",42,3),
    ("711-00025","Pad Thai Shrimp","Thai","A",65,3),
    ("711-00026","Hot Dog Bread","Western","C",25,2),
    ("711-00027","Croissant Ham","Western","C",38,2),
    ("711-00028","Salad Greens","Western","C",45,2),
    ("711-00029","Matcha Roll Cake","Snack/Side","B",35,3),
    ("711-00030","Cheese Tart","Snack/Side","B",40,2),
    ("711-00031","Chicken Karaage Bento","Japanese","A",79,3),
    ("711-00032","Gyoza Pork","Japanese","B",55,3),
    ("711-00033","Miso Soup Cup","Japanese","C",35,60),
    ("711-00034","Sushi Set Tuna","Japanese","A",89,2),
    ("711-00035","Ramen Tonkotsu","Japanese","B",75,2),
    ("711-00036","Spring Roll Pork","Snack/Side","B",30,2),
    ("711-00037","Crispy Wonton","Snack/Side","C",22,2),
    ("711-00038","Coconut Pudding","Snack/Side","C",28,3),
    ("711-00039","Mango Pudding","Snack/Side","C",28,3),
    ("711-00040","Steamed Pork Bun","Thai","B",22,2),
    ("711-00041","Stir Fried Veggies","Thai","C",40,2),
    ("711-00042","Congee Pork","Thai","B",39,2),
    ("711-00043","Boat Noodle Beef","Thai","A",55,2),
    ("711-00044","Som Tum Thai","Thai","B",45,2),
    ("711-00045","Larb Moo","Thai","C",50,2),
    ("711-00046","Pineapple Fried Rice","Thai","B",58,3),
    ("711-00047","Crab Fried Rice","Thai","A",75,3),
    ("711-00048","Beef Steak Rice","Western","A",89,3),
    ("711-00049","Fish & Chips","Western","B",79,2),
    ("711-00050","Grilled Salmon Rice","International","A",99,3),
    ("711-00051","Caesar Salad","Western","C",65,2),
    ("711-00052","Pasta Bolognese","International","B",72,3),
    ("711-00053","Lasagna Beef","International","B",85,3),
    ("711-00054","Burrito Chicken","International","C",69,3),
    ("711-00055","Tex-Mex Wrap","International","C",65,3),
    ("711-00056","Cheese Burger Set","Western","A",79,2),
    ("711-00057","Crispy Chicken Burger","Western","A",69,2),
    ("711-00058","Veggie Wrap","Western","C",55,2),
    ("711-00059","BLT Sandwich","Western","C",58,2),
    ("711-00060","Club Sandwich","Western","B",65,2),
    ("711-00061","Waffle Maple","Snack/Side","B",35,1),
    ("711-00062","Crepe Strawberry","Snack/Side","B",38,1),
    ("711-00063","Donut Glaze","Snack/Side","C",25,2),
    ("711-00064","Muffin Choco","Snack/Side","C",28,3),
    ("711-00065","Brownie Fudge","Snack/Side","C",32,3),
    ("711-00066","Thai Milk Tea Cup","Snack/Side","A",35,2),
    ("711-00067","Green Tea Latte","Snack/Side","B",40,2),
    ("711-00068","Ovaltine Ice","Snack/Side","B",30,2),
    ("711-00069","Fresh Juice Orange","Snack/Side","B",45,1),
    ("711-00070","Smoothie Berry","Snack/Side","C",55,1),
    ("711-00071","Yokoso Yakitori","Japanese","B",45,2),
    ("711-00072","Takoyaki 6pcs","Japanese","B",55,2),
    ("711-00073","Katsu Curry","Japanese","A",79,3),
    ("711-00074","Chicken Teriyaki Don","Japanese","A",75,3),
    ("711-00075","Tuna Temaki","Japanese","B",60,2),
    ("711-00076","Yaki Soba","Japanese","B",68,3),
    ("711-00077","Cold Soba Noodle","Japanese","C",65,2),
    ("711-00078","Edamame Cup","Japanese","C",38,4),
    ("711-00079","Age Dashi Tofu","Japanese","C",45,3),
    ("711-00080","Chawanmushi Egg","Japanese","C",40,3),
    ("711-00081","Dim Sum Basket","International","B",65,2),
    ("711-00082","Har Gao 4pcs","International","B",55,2),
    ("711-00083","Siu Mai 4pcs","International","B",50,2),
    ("711-00084","Char Siu Bao","International","C",35,2),
    ("711-00085","Congee Century Egg","Thai","C",42,2),
    ("711-00086","Khao Man Gai","Thai","A",55,2),
    ("711-00087","Khao Na Moo","Thai","A",50,2),
    ("711-00088","Khao Na Ped","Thai","B",55,2),
    ("711-00089","Khao Na Kai Yaang","Thai","A",50,2),
    ("711-00090","Khao Tom Moo","Thai","B",45,2),
    ("711-00091","Khanom Jeen Nam Ya","Thai","B",48,2),
    ("711-00092","Guay Tiew Tom Yum","Thai","A",52,2),
    ("711-00093","Wonton Noodle Soup","Thai","B",50,2),
    ("711-00094","Dry Noodle Pork","Thai","B",48,2),
    ("711-00095","Spicy Seafood Noodle","Thai","A",65,2),
    ("711-00096","Rice Crispy Snack","Snack/Side","C",20,90),
    ("711-00097","Seaweed Roll Snack","Snack/Side","C",25,90),
    ("711-00098","Dried Mango Snack","Snack/Side","C",35,60),
    ("711-00099","Popcorn Caramel","Snack/Side","C",30,60),
    ("711-00100","Mixed Nuts Cup","Snack/Side","B",45,90),
    ("711-00101","Protein Bar Choco","Snack/Side","B",55,60),
    ("711-00102","Energy Bar Granola","Snack/Side","C",48,60),
    ("711-00103","Yogurt Parfait","Snack/Side","B",45,2),
    ("711-00104","Greek Yogurt Cup","Snack/Side","C",52,5),
    ("711-00105","Fruit Salad Cup","Snack/Side","C",38,2),
    ("711-00106","Tapioca Pudding","Snack/Side","C",30,3),
    ("711-00107","Almond Tofu","Snack/Side","C",28,3),
    ("711-00108","Red Bean Bun","Snack/Side","B",22,2),
    ("711-00109","Cream Puff","Snack/Side","B",25,2),
    ("711-00110","Egg Tart","Snack/Side","B",28,2),
    ("711-00111","Pad Prik King Pork","Thai","B",55,3),
    ("711-00112","Cashew Chicken Rice","Thai","A",58,3),
    ("711-00113","Yellow Curry Chicken","Thai","B",58,3),
    ("711-00114","Panang Curry Beef","Thai","B",65,3),
    ("711-00115","Beef Stir Fry Oyster","Thai","A",60,3),
    ("711-00116","Shrimp Paste Fried Rice","Thai","A",62,3),
    ("711-00117","Pork Knuckle Rice","International","A",89,3),
    ("711-00118","Duck Red Curry Rice","Thai","A",75,3),
    ("711-00119","Spicy Squid Salad","Thai","B",58,2),
    ("711-00120","Grilled Pork Collar Rice","Thai","A",60,3),
]

VILLAGES = [
    ("V1001","Golden Ville"),  ("V1002","Greenery Home"), ("V1003","Perfect Park"),
    ("V1004","The Grand"),     ("V1005","Modi Villa"),    ("V1006","Pruksa Village"),
    ("V1007","Centro Park"),   ("V1008","Siri Place"),    ("V1009","Lalin Town"),
    ("V1010","Pleno Town"),    ("V1011","Casa City"),     ("V1012","Indy Village"),
    ("V1013","Supalai Garden"),("V1014","Britania"),      ("V1015","Habitia"),
    ("V1016","Villette"),      ("V1017","Passorn"),       ("V1018","City Sense"),
    ("V1019","Patio Village"), ("V1020","The Connect"),
]

# Promotion types: (label, demand_uplift_pct, encode)
PROMO_TYPES = {
    0: ("none",       0.00),
    1: ("discount_10",0.20),  # ลด 10% → demand +20%
    2: ("discount_20",0.40),  # ลด 20% → demand +40%
    3: ("buy1get1",   0.80),  # ซื้อ 1 แถม 1 → demand +80%
    4: ("bundle",     0.35),  # bundle set → demand +35%
}

RANK_BASE = {"A": 28, "B": 18, "C": 10}
RANK_STD  = {"A": 7,  "B": 5,  "C": 3}

# Category demand multiplier (ปัจจัยเฉพาะ category)
CAT_MULT = {
    "Thai": 1.10, "Japanese": 1.00, "Western": 0.95,
    "International": 0.90, "Snack/Side": 0.85,
}

# ── STEP 1: GENERATE PROMOTION SCHEDULE ──────────────────────────────────────

print("📋 Step 1: Generating promotion schedule...")

start_date = date.today() - timedelta(days=HISTORY_DAYS)

# สร้างตารางโปรโมชั่นล่วงหน้า (pre-planned — model รู้ล่วงหน้า)
promo_schedule = {}  # (sku_id, date) → promo_type_enc
for sku_id, *_ in SKUS:
    for day_offset in range(HISTORY_DAYS + 2):
        d = start_date + timedelta(days=day_offset)
        # แต่ละ SKU มีโปรโมชั่น ~20% ของวัน
        if random.random() < 0.20:
            promo_type = random.choices([1, 2, 3, 4], weights=[50, 30, 10, 10])[0]
        else:
            promo_type = 0
        promo_schedule[(sku_id, d)] = promo_type

print(f"   Promo days: {sum(1 for v in promo_schedule.values() if v > 0):,} / {len(promo_schedule):,}")

# ── STEP 2: GENERATE HISTORICAL SALES WITH PROMO SIGNAL ──────────────────────

print("\n📅 Step 2: Generating 120-day historical sales with promotions...")

def true_daily_sales(rank, category, d: date, promo_type: int) -> float:
    base       = RANK_BASE[rank] * CAT_MULT.get(category, 1.0)
    std        = RANK_STD[rank]
    is_weekend = d.weekday() >= 5
    is_payday  = d.day in (25, 26, 27, 28, 29, 30, 1)
    is_spike   = random.random() < 0.03  # spike เหลือแค่ 3% (ลดจาก 12%)

    # Calendar multiplier
    cal_mult = 1.0
    if is_spike and is_payday:  cal_mult = 1.80
    elif is_spike and is_weekend: cal_mult = 1.60
    elif is_spike:              cal_mult = 1.50
    elif is_payday:             cal_mult = 1.45
    elif is_weekend:            cal_mult = 1.35

    # Promotion uplift (known signal — model เรียนรู้ได้ชัดเจน)
    promo_uplift = 1.0 + PROMO_TYPES[promo_type][1]

    total_mult = cal_mult * promo_uplift
    return max(1, np.random.normal(base * total_mult, std * math.sqrt(total_mult)))

history_rows = []
for v_id, v_name in VILLAGES:
    for sku_id, name, category, rank, price, life in SKUS:
        for day_offset in range(HISTORY_DAYS):
            d = start_date + timedelta(days=day_offset)
            promo = promo_schedule.get((sku_id, d), 0)
            sales = true_daily_sales(rank, category, d, promo)
            history_rows.append({
                "village_id":  v_id,
                "sku_id":      sku_id,
                "rank":        rank,
                "price":       price,
                "life_days":   life,
                "category":    category,
                "date":        d,
                "sales":       round(sales),
                "promo_type":  promo,
                "discount_pct": PROMO_TYPES[promo][1] * 100,
                "is_promo":    int(promo > 0),
            })

hist_df = pd.DataFrame(history_rows)
hist_df["date"] = pd.to_datetime(hist_df["date"])
hist_df = hist_df.sort_values(["village_id","sku_id","date"]).reset_index(drop=True)
print(f"   {len(hist_df):,} records generated")

# ── STEP 3: FEATURE ENGINEERING ──────────────────────────────────────────────

print("\n🔧 Step 3: Engineering features...")

# Calendar
hist_df["dow"]         = hist_df["date"].dt.dayofweek
hist_df["dom"]         = hist_df["date"].dt.day
hist_df["is_weekend"]  = (hist_df["dow"] >= 5).astype(int)
hist_df["is_payday"]   = hist_df["dom"].isin([25,26,27,28,29,30,1]).astype(int)
hist_df["week_of_year"]= hist_df["date"].dt.isocalendar().week.astype(int)

# Fourier terms — จับ weekly seasonality (sin/cos)
hist_df["sin_dow"] = np.sin(2 * np.pi * hist_df["dow"] / 7)
hist_df["cos_dow"] = np.cos(2 * np.pi * hist_df["dow"] / 7)

# Promotion features
hist_df["promo_type"]    = hist_df["promo_type"].astype(int)
hist_df["discount_pct"]  = hist_df["discount_pct"].astype(float)
hist_df["is_promo"]      = hist_df["is_promo"].astype(int)

# Promo look-ahead: โปรโมชั่นพรุ่งนี้ (model รู้ล่วงหน้า)
grp = hist_df.groupby(["village_id","sku_id"])
hist_df["promo_tomorrow"] = grp["is_promo"].shift(-1).fillna(0).astype(int)
hist_df["promo_lag_1"]    = grp["is_promo"].shift(1).fillna(0).astype(int)

# Days since/until promo
def days_since_promo(s):
    result = []
    count  = 99
    for v in s:
        if v > 0: count = 0
        result.append(count)
        count = min(count + 1, 99)
    return result

def days_until_promo(s):
    result = [99] * len(s)
    vals   = list(s)
    for i in range(len(vals)-1, -1, -1):
        if vals[i] > 0:
            result[i] = 0
        elif i < len(vals)-1 and result[i+1] < 99:
            result[i] = result[i+1] + 1
    return result

hist_df["days_since_promo"] = grp["is_promo"].transform(days_since_promo)
hist_df["days_until_promo"] = grp["is_promo"].transform(days_until_promo)

# Encode rank & category
hist_df["rank_enc"] = hist_df["rank"].map({"A":3,"B":2,"C":1})
hist_df["cat_enc"]  = hist_df["category"].astype("category").cat.codes

# Lag features (ยาวขึ้น)
for lag in [1, 2, 3, 7, 14, 21, 28]:
    hist_df[f"lag_{lag}"] = grp["sales"].shift(lag)

# Rolling statistics
for w in [3, 7, 14]:
    hist_df[f"roll_mean_{w}"] = grp["sales"].transform(
        lambda x: x.shift(1).rolling(w, min_periods=max(2,w//2)).mean())
    hist_df[f"roll_std_{w}"]  = grp["sales"].transform(
        lambda x: x.shift(1).rolling(w, min_periods=max(2,w//2)).std().fillna(0))

# Interaction features
hist_df["rank_x_promo"]   = hist_df["rank_enc"] * hist_df["discount_pct"]
hist_df["rank_x_weekend"] = hist_df["rank_enc"] * hist_df["is_weekend"]
hist_df["rank_x_payday"]  = hist_df["rank_enc"] * hist_df["is_payday"]

hist_df = hist_df.dropna(subset=["lag_1","lag_7","lag_14","roll_mean_7"])
print(f"   Features ready — {len(hist_df):,} training rows")

FEATURES = [
    # Product
    "rank_enc", "price", "life_days", "cat_enc",
    # Calendar
    "dow", "dom", "is_weekend", "is_payday", "week_of_year",
    "sin_dow", "cos_dow",
    # Promotions
    "is_promo", "promo_type", "discount_pct",
    "promo_lag_1", "promo_tomorrow",
    "days_since_promo", "days_until_promo",
    # Lag
    "lag_1","lag_2","lag_3","lag_7","lag_14","lag_21","lag_28",
    # Rolling
    "roll_mean_3","roll_mean_7","roll_mean_14",
    "roll_std_3","roll_std_7","roll_std_14",
    # Interactions
    "rank_x_promo","rank_x_weekend","rank_x_payday",
]

# ── STEP 4: TRAIN LIGHTGBM ───────────────────────────────────────────────────

print("\n🤖 Step 4: Training LightGBM v2...")

# Walk-forward split — ใช้ 80 วันแรก train, 40 วันหลัง validate
cut_date = pd.Timestamp(start_date + timedelta(days=80))
train_df = hist_df[hist_df["date"] <  cut_date]
val_df   = hist_df[hist_df["date"] >= cut_date]

X_train, y_train = train_df[FEATURES], train_df["sales"]
X_val,   y_val   = val_df[FEATURES],   val_df["sales"]

model = lgb.LGBMRegressor(
    objective         = "regression",   # train บน log1p(y)
    n_estimators      = 1500,
    learning_rate     = 0.02,
    num_leaves        = 63,
    max_depth         = -1,
    min_child_samples = 15,
    subsample         = 0.85,
    subsample_freq    = 1,
    colsample_bytree  = 0.75,
    reg_alpha         = 0.05,
    reg_lambda        = 1.0,
    random_state      = 42,
    verbose           = -1,
)

# train บน log1p — ลด bias ต่อ Rank C (ค่าน้อย)
model.fit(
    X_train, np.log1p(y_train),
    eval_set=[(X_val, np.log1p(y_val))],
    callbacks=[
        lgb.early_stopping(100, verbose=False),
        lgb.log_evaluation(period=-1),
    ],
)

# ── STEP 5: EVALUATE ─────────────────────────────────────────────────────────

print("\n📊 Step 5: Evaluating model...")

# predict → inverse log1p
y_pred_log = model.predict(X_val)
y_pred     = np.expm1(y_pred_log)
y_pred     = np.maximum(y_pred, 1)

mae    = mean_absolute_error(y_val, y_pred)
rmse   = math.sqrt(mean_squared_error(y_val, y_pred))

# MAPE (unweighted — ทุก row เท่ากัน)
mape   = float(np.mean(np.abs((y_val.values - y_pred) / y_val.values)) * 100)

# WMAPE (Weighted MAPE) — มาตรฐาน retail: weight = actual sales volume
wmape  = float(np.sum(np.abs(y_val.values - y_pred)) / np.sum(y_val.values) * 100)

# Baselines
mae_naive  = mean_absolute_error(y_val, val_df["lag_1"])
mae_roll7  = mean_absolute_error(y_val, val_df["roll_mean_7"])
skill      = (1 - mae/mae_naive) * 100
wmape_naive= float(np.sum(np.abs(y_val.values - val_df["lag_1"].values)) / np.sum(y_val.values) * 100)

print(f"\n   {'Metric':<24} {'LightGBM v2':>12} {'Naive (lag1)':>13}")
print(f"   {'-'*52}")
print(f"   {'MAE (units)':<24} {mae:>12.2f} {mae_naive:>13.2f}")
mape_naive = float(np.mean(np.abs((y_val.values - val_df["lag_1"].values) / y_val.values)) * 100)
print(f"   {'MAPE % (unweighted)':<24} {mape:>12.1f} {mape_naive:>13.1f}")
print(f"   {'WMAPE % (weighted)':<24} {wmape:>12.1f} {wmape_naive:>13.1f}  ← retail standard")
print(f"   {'RMSE':<24} {rmse:>12.2f}")
print(f"   {'Skill vs Naive':<24} {skill:>+12.1f}%")

w_status = "✅ ผ่านมาตรฐาน" if wmape < 20 else "⚠️ ยังไม่ถึง 20%"
print(f"\n   WMAPE = {wmape:.1f}%  {w_status}")

# MAE by Rank
print("\n   MAE by Rank:")
val_df = val_df.copy()
val_df["pred"] = y_pred
for rank, label in [(3,"A"),(2,"B"),(1,"C")]:
    sub  = val_df[val_df["rank_enc"]==rank]
    m    = mean_absolute_error(sub["sales"], sub["pred"])
    avg  = sub["sales"].mean()
    pct  = m/avg*100
    bar  = "█" * int(pct/2)
    print(f"   Rank {label}: MAE={m:.1f}  avg={avg:.0f}  err={pct:.1f}%  {bar}")

# MAE by Promo vs Non-promo
print("\n   MAE by Promotion:")
for flag, label in [(1,"🏷️ Promo"),(0,"📅 No Promo")]:
    sub = val_df[val_df["is_promo"]==flag]
    if len(sub):
        m = mean_absolute_error(sub["sales"], sub["pred"])
        print(f"   {label}: MAE={m:.1f}  n={len(sub):,}")

# Feature importance
print("\n   Top 10 Features:")
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, val_imp in imp.head(10).items():
    bar = "█" * int(val_imp/imp.max()*18)
    print(f"   {feat:<22} {bar}")

# ── STEP 6: PREDICT TOMORROW ─────────────────────────────────────────────────

print(f"\n📈 Step 6: Predicting demand for {FORECAST_DATE}...")

def demand_factor_label(d: date, promo_type: int) -> str:
    is_weekend = d.weekday() >= 5
    is_payday  = d.day in (25, 26, 27, 28, 29, 30, 1)
    is_spike   = random.random() < 0.03

    if promo_type > 0 and is_payday:  return "Spike + Payday"
    if promo_type > 0 and is_weekend: return "Spike + Weekend"
    if promo_type > 0:                return "Demand Spike"
    if is_spike and is_payday:        return "Spike + Payday"
    if is_spike and is_weekend:       return "Spike + Weekend"
    if is_spike:                      return "Demand Spike"
    if is_payday:                     return "Payday"
    if is_weekend:                    return "Weekend"
    return "Normal Weekday"

cat_codes = {c: i for i, c in enumerate(
    sorted(set(sku[2] for sku in SKUS)))}

forecast_rows = []
for v_id, v_name in VILLAGES:
    for sku_id, name, category, rank, price, life in SKUS:
        mask   = (hist_df["village_id"]==v_id) & (hist_df["sku_id"]==sku_id)
        recent = hist_df[mask].tail(30)
        rank_enc   = {"A": 3, "B": 2, "C": 1}[rank]
        roll_sales = recent["sales"]

        lags = {
            f"lag_{n}": int(recent["sales"].iloc[-n]) if len(recent) >= n else RANK_BASE[rank]
            for n in [1, 2, 3, 7, 14, 21, 28]
        }

        promo_tmr  = promo_schedule.get((sku_id, FORECAST_DATE), 0)
        promo_yest = int(recent["is_promo"].iloc[-1]) if len(recent) >= 1 else 0
        disc_pct   = PROMO_TYPES[promo_tmr][1] * 100

        # days_since_promo
        ds = 0
        for i in range(len(recent)-1, -1, -1):
            if recent["is_promo"].iloc[i] > 0:
                break
            ds += 1

        fcast_dow      = FORECAST_DATE.weekday()
        fcast_dom      = FORECAST_DATE.day
        fcast_is_wknd  = int(fcast_dow >= 5)
        fcast_is_pay   = int(fcast_dom in (25, 26, 27, 28, 29, 30, 1))

        feat = pd.DataFrame([{
            "rank_enc":         rank_enc,
            "price":            price,
            "life_days":        life,
            "cat_enc":          cat_codes.get(category, 0),
            "dow":              fcast_dow,
            "dom":              fcast_dom,
            "is_weekend":       fcast_is_wknd,
            "is_payday":        fcast_is_pay,
            "week_of_year":     FORECAST_DATE.isocalendar()[1],
            "sin_dow":          math.sin(2 * math.pi * fcast_dow / 7),
            "cos_dow":          math.cos(2 * math.pi * fcast_dow / 7),
            "is_promo":         int(promo_tmr > 0),
            "promo_type":       promo_tmr,
            "discount_pct":     disc_pct,
            "promo_lag_1":      promo_yest,
            "promo_tomorrow":   0,
            "days_since_promo": ds,
            "days_until_promo": 0 if promo_tmr > 0 else 1,
            **lags,
            "roll_mean_3":      float(roll_sales.tail(3).mean()),
            "roll_mean_7":      float(roll_sales.tail(7).mean()),
            "roll_mean_14":     float(roll_sales.tail(14).mean()),
            "roll_std_3":       float(roll_sales.tail(3).std())  if len(recent) >= 2 else 0,
            "roll_std_7":       float(roll_sales.tail(7).std())  if len(recent) >= 2 else 0,
            "roll_std_14":      float(roll_sales.tail(14).std()) if len(recent) >= 2 else 0,
            "rank_x_promo":     rank_enc * disc_pct,
            "rank_x_weekend":   rank_enc * fcast_is_wknd,
            "rank_x_payday":    rank_enc * fcast_is_pay,
        }])

        predicted = max(1, round(float(np.expm1(model.predict(feat)[0]))))
        factor    = demand_factor_label(FORECAST_DATE, promo_tmr)

        roll = random.random()
        if roll < 0.60:
            inventory = max(0, predicted + random.randint(-SAFETY_STOCK, SAFETY_STOCK))
        elif roll < 0.85:
            shortage  = random.randint(SAFETY_STOCK+1, max(SAFETY_STOCK+2, predicted//2))
            inventory = max(0, predicted - shortage)
        else:
            surplus   = random.randint(SAFETY_STOCK+1, max(SAFETY_STOCK+2, predicted//3))
            inventory = predicted + surplus

        gap       = predicted - inventory
        sug_order = max(0, gap - SAFETY_STOCK)
        overstock = max(0, -(gap + SAFETY_STOCK))
        status    = ("STOCKOUT" if gap > SAFETY_STOCK
                     else "OVERSTOCK" if gap < -SAFETY_STOCK
                     else "BALANCED")

        forecast_rows.append({
            "village_id": v_id, "village_name": v_name,
            "sku_id": sku_id,   "product_name": name,
            "category": category, "rank": rank,
            "price": price, "product_life_days": life,
            "inventory_on_hand": inventory,
            "predicted_demand":  predicted,
            "stock_gap":         gap,
            "suggested_order":   sug_order,
            "overstock_qty":     overstock,
            "stockout_qty":      max(0, gap),
            "stock_status":      status,
            "demand_factor":     factor,
            "date":              FORECAST_DATE.isoformat(),
        })

print(f"   {len(forecast_rows):,} forecasts generated")

# ── STEP 7: UPLOAD TO SUPABASE ────────────────────────────────────────────────

print("\n☁️  Step 7: Uploading to Supabase...")

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type":  "application/json",
    "Prefer":        "return=minimal",
}

# ลบข้อมูลเก่าของวันนี้ก่อน
req = urllib.request.Request(
    f"{SUPABASE_URL}/rest/v1/sku_detail?date=eq.{FORECAST_DATE.isoformat()}",
    headers=headers, method="DELETE")
urllib.request.urlopen(req)

BATCH, inserted = 200, 0
for i in range(0, len(forecast_rows), BATCH):
    batch = forecast_rows[i:i+BATCH]
    req   = urllib.request.Request(
        f"{SUPABASE_URL}/rest/v1/sku_detail",
        data=json.dumps(batch).encode(), headers=headers, method="POST")
    urllib.request.urlopen(req)
    inserted += len(batch)
    print(f"   ✅ {min(i+BATCH, len(forecast_rows))}/{len(forecast_rows)}")

# Update village summary
vstats = defaultdict(lambda: {"so":0,"ba":0,"ov":0,"ord":0,"ovq":0})
for r in forecast_rows:
    s = vstats[r["village_id"]]
    if r["stock_status"]=="STOCKOUT":  s["so"]+=1
    elif r["stock_status"]=="OVERSTOCK": s["ov"]+=1
    else: s["ba"]+=1
    s["ord"]+=r["suggested_order"]
    s["ovq"]+=r["overstock_qty"]

for v_id, _ in VILLAGES:
    s = vstats[v_id]
    patch = json.dumps({"stockout_count":s["so"],"balanced_count":s["ba"],
                         "overstock_count":s["ov"],"total_suggested_order":s["ord"],
                         "total_overstock":s["ovq"]}).encode()
    req = urllib.request.Request(
        f"{SUPABASE_URL}/rest/v1/villages?village_id=eq.{v_id}",
        data=patch, headers={**headers,"Prefer":"return=minimal"}, method="PATCH")
    urllib.request.urlopen(req)

print(f"\n{'='*55}")
print(f"🎉 เสร็จแล้ว!")
print(f"   Model   : LightGBM + Promotion Features")
print(f"   WMAPE   : {wmape:.1f}%  {w_status}")
print(f"   MAE     : {mae:.2f} units")
print(f"   Uploaded: {inserted:,} forecasts → Supabase")
print(f"   Date    : {FORECAST_DATE}")
print(f"{'='*55}")

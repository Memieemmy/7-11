"""
7-Eleven RTE Demand Forecasting — Synthetic Data Generator
Fixes from audit:
  1. Predicted_Demand uses Normal distribution (not Uniform)
  2. Rank A > B > C demand (meaningful differentiation)
  3. Demand_Factor multipliers are realistic (Payday/Weekend > Weekday)
  4. Inventory generated independently from demand (no artificial 0.83 correlation)
  5. OVERSTOCK cases included (~15% of SKUs)
  6. Stock_Status threshold based on safety stock logic
"""

import json, random
from datetime import date, timedelta

random.seed(42)

# ── CONFIG ──────────────────────────────────────────────────────────────────

VILLAGES = [
    ("V1001", "Golden Ville"),   ("V1002", "Greenery Home"),
    ("V1003", "Perfect Park"),   ("V1004", "The Grand"),
    ("V1005", "Modi Villa"),     ("V1006", "Pruksa Village"),
    ("V1007", "Centro Park"),    ("V1008", "Siri Place"),
    ("V1009", "Lalin Town"),     ("V1010", "Pleno Town"),
    ("V1011", "Casa City"),      ("V1012", "Indy Village"),
    ("V1013", "Supalai Garden"), ("V1014", "Britania"),
    ("V1015", "Habitia"),        ("V1016", "Villette"),
    ("V1017", "Passorn"),        ("V1018", "City Sense"),
    ("V1019", "Patio Village"),  ("V1020", "The Connect"),
]

SKUS = [
    # (SKU_ID, Product_Name, Category, Rank, Price, Product_Life_Days)
    ("711-00001", "Krapow Chicken Ezygo",       "Thai",              "A", 39, 3),
    ("711-00002", "Krapow Pork Ezygo",          "Thai",              "B", 45, 5),
    ("711-00003", "Spaghetti Carbonara",        "International",     "B", 69, 4),
    ("711-00004", "Pad See Ew Chicken",         "Thai",              "A", 42, 3),
    ("711-00005", "Moo Ping Set",               "Thai",              "A", 35, 2),
    ("711-00006", "Fried Rice Egg",             "Thai",              "A", 38, 3),
    ("711-00007", "Teriyaki Chicken Bento",     "Japanese",          "B", 75, 4),
    ("711-00008", "Tom Yum Soup",               "Thai",              "A", 49, 2),
    ("711-00009", "Green Curry Rice",           "Thai",              "B", 55, 3),
    ("711-00010", "Fried Chicken Set",          "Western",           "A", 65, 2),
    ("711-00011", "Onigiri Tuna Mayo",          "Japanese",          "B", 28, 3),
    ("711-00012", "Onigiri Salmon",             "Japanese",          "B", 32, 3),
    ("711-00013", "Sandwich Ham Cheese",        "Western",           "B", 55, 2),
    ("711-00014", "Sandwich Tuna",              "Western",           "C", 48, 2),
    ("711-00015", "Mango Sticky Rice",          "Snack/Side Dish",   "B", 45, 2),
    ("711-00016", "Corn Grilled",               "Snack/Side Dish",   "C", 25, 1),
    ("711-00017", "Instant Noodle Spicy",       "Snack/Side Dish",   "C", 18, 90),
    ("711-00018", "Banana Roti",                "Snack/Side Dish",   "B", 30, 1),
    ("711-00019", "Pork Ball Skewer",           "Snack/Side Dish",   "A", 25, 1),
    ("711-00020", "Chicken Nuggets",            "Western",           "B", 55, 2),
    ("711-00021", "BBQ Pork Rice",              "Thai",              "A", 50, 3),
    ("711-00022", "Massaman Curry",             "Thai",              "B", 60, 3),
    ("711-00023", "Basil Pork Rice",            "Thai",              "A", 45, 3),
    ("711-00024", "Egg Fried Rice",             "Thai",              "B", 42, 3),
    ("711-00025", "Pad Thai Shrimp",            "Thai",              "A", 65, 3),
    ("711-00026", "Hot Dog Bread",              "Western",           "C", 25, 2),
    ("711-00027", "Croissant Ham",              "Western",           "C", 38, 2),
    ("711-00028", "Salad Greens",               "Western",           "C", 45, 2),
    ("711-00029", "Matcha Roll Cake",           "Snack/Side Dish",   "B", 35, 3),
    ("711-00030", "Cheese Tart",                "Snack/Side Dish",   "B", 40, 2),
    ("711-00031", "Chicken Karaage Bento",      "Japanese",          "A", 79, 3),
    ("711-00032", "Gyoza Pork",                 "Japanese",          "B", 55, 3),
    ("711-00033", "Miso Soup Cup",              "Japanese",          "C", 35, 60),
    ("711-00034", "Sushi Set Tuna",             "Japanese",          "A", 89, 2),
    ("711-00035", "Ramen Tonkotsu",             "Japanese",          "B", 75, 2),
    ("711-00036", "Spring Roll Pork",           "Snack/Side Dish",   "B", 30, 2),
    ("711-00037", "Crispy Wonton",              "Snack/Side Dish",   "C", 22, 2),
    ("711-00038", "Coconut Pudding",            "Snack/Side Dish",   "C", 28, 3),
    ("711-00039", "Mango Pudding",              "Snack/Side Dish",   "C", 28, 3),
    ("711-00040", "Steamed Pork Bun",           "Thai",              "B", 22, 2),
    ("711-00041", "Stir Fried Veggies",         "Thai",              "C", 40, 2),
    ("711-00042", "Congee Pork",                "Thai",              "B", 39, 2),
    ("711-00043", "Boat Noodle Beef",           "Thai",              "A", 55, 2),
    ("711-00044", "Som Tum Thai",               "Thai",              "B", 45, 2),
    ("711-00045", "Larb Moo",                   "Thai",              "C", 50, 2),
    ("711-00046", "Pineapple Fried Rice",       "Thai",              "B", 58, 3),
    ("711-00047", "Crab Fried Rice",            "Thai",              "A", 75, 3),
    ("711-00048", "Beef Steak Rice",            "Western",           "A", 89, 3),
    ("711-00049", "Fish & Chips",               "Western",           "B", 79, 2),
    ("711-00050", "Grilled Salmon Rice",        "International",     "A", 99, 3),
    ("711-00051", "Caesar Salad",               "Western",           "C", 65, 2),
    ("711-00052", "Pasta Bolognese",            "International",     "B", 72, 3),
    ("711-00053", "Lasagna Beef",               "International",     "B", 85, 3),
    ("711-00054", "Burrito Chicken",            "International",     "C", 69, 3),
    ("711-00055", "Tex-Mex Wrap",               "International",     "C", 65, 3),
    ("711-00056", "Cheese Burger Set",          "Western",           "A", 79, 2),
    ("711-00057", "Crispy Chicken Burger",      "Western",           "A", 69, 2),
    ("711-00058", "Veggie Wrap",                "Western",           "C", 55, 2),
    ("711-00059", "BLT Sandwich",               "Western",           "C", 58, 2),
    ("711-00060", "Club Sandwich",              "Western",           "B", 65, 2),
    ("711-00061", "Waffle Maple",               "Snack/Side Dish",   "B", 35, 1),
    ("711-00062", "Crepe Strawberry",           "Snack/Side Dish",   "B", 38, 1),
    ("711-00063", "Donut Glaze",                "Snack/Side Dish",   "C", 25, 2),
    ("711-00064", "Muffin Choco",               "Snack/Side Dish",   "C", 28, 3),
    ("711-00065", "Brownie Fudge",              "Snack/Side Dish",   "C", 32, 3),
    ("711-00066", "Thai Milk Tea Cup",          "Snack/Side Dish",   "A", 35, 2),
    ("711-00067", "Green Tea Latte",            "Snack/Side Dish",   "B", 40, 2),
    ("711-00068", "Ovaltine Ice",               "Snack/Side Dish",   "B", 30, 2),
    ("711-00069", "Fresh Juice Orange",         "Snack/Side Dish",   "B", 45, 1),
    ("711-00070", "Smoothie Berry",             "Snack/Side Dish",   "C", 55, 1),
    ("711-00071", "Yokoso Yakitori",            "Japanese",          "B", 45, 2),
    ("711-00072", "Takoyaki 6pcs",              "Japanese",          "B", 55, 2),
    ("711-00073", "Katsu Curry",                "Japanese",          "A", 79, 3),
    ("711-00074", "Chicken Teriyaki Don",       "Japanese",          "A", 75, 3),
    ("711-00075", "Tuna Temaki",                "Japanese",          "B", 60, 2),
    ("711-00076", "Yaki Soba",                  "Japanese",          "B", 68, 3),
    ("711-00077", "Cold Soba Noodle",           "Japanese",          "C", 65, 2),
    ("711-00078", "Edamame Cup",                "Japanese",          "C", 38, 4),
    ("711-00079", "Age Dashi Tofu",             "Japanese",          "C", 45, 3),
    ("711-00080", "Chawanmushi Egg",            "Japanese",          "C", 40, 3),
    ("711-00081", "Dim Sum Basket",             "International",     "B", 65, 2),
    ("711-00082", "Har Gao 4pcs",               "International",     "B", 55, 2),
    ("711-00083", "Siu Mai 4pcs",               "International",     "B", 50, 2),
    ("711-00084", "Char Siu Bao",               "International",     "C", 35, 2),
    ("711-00085", "Congee Century Egg",         "Thai",              "C", 42, 2),
    ("711-00086", "Khao Man Gai",               "Thai",              "A", 55, 2),
    ("711-00087", "Khao Na Moo",                "Thai",              "A", 50, 2),
    ("711-00088", "Khao Na Ped",                "Thai",              "B", 55, 2),
    ("711-00089", "Khao Na Kai Yaang",          "Thai",              "A", 50, 2),
    ("711-00090", "Khao Tom Moo",               "Thai",              "B", 45, 2),
    ("711-00091", "Khanom Jeen Nam Ya",         "Thai",              "B", 48, 2),
    ("711-00092", "Guay Tiew Tom Yum",          "Thai",              "A", 52, 2),
    ("711-00093", "Wonton Noodle Soup",         "Thai",              "B", 50, 2),
    ("711-00094", "Dry Noodle Pork",            "Thai",              "B", 48, 2),
    ("711-00095", "Spicy Seafood Noodle",       "Thai",              "A", 65, 2),
    ("711-00096", "Rice Crispy Snack",          "Snack/Side Dish",   "C", 20, 90),
    ("711-00097", "Seaweed Roll Snack",         "Snack/Side Dish",   "C", 25, 90),
    ("711-00098", "Dried Mango Snack",          "Snack/Side Dish",   "C", 35, 60),
    ("711-00099", "Popcorn Caramel",            "Snack/Side Dish",   "C", 30, 60),
    ("711-00100", "Mixed Nuts Cup",             "Snack/Side Dish",   "B", 45, 90),
    ("711-00101", "Protein Bar Choco",          "Snack/Side Dish",   "B", 55, 60),
    ("711-00102", "Energy Bar Granola",         "Snack/Side Dish",   "C", 48, 60),
    ("711-00103", "Yogurt Parfait",             "Snack/Side Dish",   "B", 45, 2),
    ("711-00104", "Greek Yogurt Cup",           "Snack/Side Dish",   "C", 52, 5),
    ("711-00105", "Fruit Salad Cup",            "Snack/Side Dish",   "C", 38, 2),
    ("711-00106", "Tapioca Pudding",            "Snack/Side Dish",   "C", 30, 3),
    ("711-00107", "Almond Tofu",                "Snack/Side Dish",   "C", 28, 3),
    ("711-00108", "Red Bean Bun",               "Snack/Side Dish",   "B", 22, 2),
    ("711-00109", "Cream Puff",                 "Snack/Side Dish",   "B", 25, 2),
    ("711-00110", "Egg Tart",                   "Snack/Side Dish",   "B", 28, 2),
    ("711-00111", "Pad Prik King Pork",         "Thai",              "B", 55, 3),
    ("711-00112", "Cashew Chicken Rice",        "Thai",              "A", 58, 3),
    ("711-00113", "Yellow Curry Chicken",       "Thai",              "B", 58, 3),
    ("711-00114", "Panang Curry Beef",          "Thai",              "B", 65, 3),
    ("711-00115", "Beef Stir Fry Oyster",       "Thai",              "A", 60, 3),
    ("711-00116", "Shrimp Paste Fried Rice",    "Thai",              "A", 62, 3),
    ("711-00117", "Pork Knuckle Rice",          "International",     "A", 89, 3),
    ("711-00118", "Duck Red Curry Rice",        "Thai",              "A", 75, 3),
    ("711-00119", "Spicy Squid Salad",          "Thai",              "B", 58, 2),
    ("711-00120", "Grilled Pork Collar Rice",   "Thai",              "A", 60, 3),
]

# Demand base (mean) by Rank — fixes Rank having no effect
RANK_DEMAND_BASE = {"A": 28, "B": 18, "C": 10}
RANK_DEMAND_STD  = {"A": 7,  "B": 5,  "C": 3}

# Demand multipliers by factor
FACTOR_MULTIPLIER = {
    "Normal Weekday": 1.00,
    "Weekend":        1.35,   # +35%
    "Payday":         1.45,   # +45%
    "Demand Spike":   2.00,   # +100%
    "Spike + Weekend": 2.30,  # +130%
    "Spike + Payday":  2.50,  # +150%
}

# Safety stock buffer (units) — determines BALANCED vs STOCKOUT
SAFETY_STOCK = 3

# Target stock coverage (days) — how many days of demand to hold as inventory
COVERAGE_DAYS = 1.5


def clamp_int(val, lo, hi):
    return max(lo, min(hi, round(val)))


def demand_factor_for_date(d: date) -> str:
    is_weekend = d.weekday() >= 5
    is_payday  = d.day in (25, 26, 27, 28, 29, 30, 1)
    is_spike   = random.random() < 0.12  # 12% chance of demand spike

    if is_spike and is_payday:  return "Spike + Payday"
    if is_spike and is_weekend: return "Spike + Weekend"
    if is_spike:                return "Demand Spike"
    if is_payday:               return "Payday"
    if is_weekend:              return "Weekend"
    return "Normal Weekday"


def generate_record(village_id, village_name, sku, obs_date: date) -> dict:
    sku_id, name, category, rank, price, life = sku
    factor = demand_factor_for_date(obs_date)
    mult   = FACTOR_MULTIPLIER[factor]

    # Predicted demand — Normal distribution, Rank-differentiated, factor-adjusted
    base_demand = random.gauss(RANK_DEMAND_BASE[rank], RANK_DEMAND_STD[rank])
    predicted   = clamp_int(base_demand * mult, 1, 80)

    # Inventory — generated independently to produce realistic status mix:
    #   ~45% BALANCED, ~25% STOCKOUT, ~30% OVERSTOCK
    roll = random.random()
    if roll < 0.45:     # BALANCED: inventory close to demand
        inventory = clamp_int(predicted + random.randint(-SAFETY_STOCK, SAFETY_STOCK), 0, 120)
    elif roll < 0.70:   # STOCKOUT: inventory meaningfully below demand
        shortage  = random.randint(SAFETY_STOCK + 1, max(SAFETY_STOCK + 2, predicted // 2))
        inventory = clamp_int(predicted - shortage, 0, 120)
    else:               # OVERSTOCK: inventory meaningfully above demand
        surplus   = random.randint(SAFETY_STOCK + 1, max(SAFETY_STOCK + 2, predicted // 3))
        inventory = clamp_int(predicted + surplus, 0, 120)

    gap         = predicted - inventory
    sug_order   = max(0, gap - SAFETY_STOCK)
    overstock   = max(0, -(gap + SAFETY_STOCK))

    if gap > SAFETY_STOCK:
        status = "STOCKOUT"
    elif gap < -SAFETY_STOCK:
        status = "OVERSTOCK"
    else:
        status = "BALANCED"

    return {
        "Village_ID":       village_id,
        "Village_Name":     village_name,
        "SKU_ID":           sku_id,
        "Product_Name":     name,
        "Category":         category,
        "Rank":             rank,
        "Price":            price,
        "Product_Life_Days": life,
        "Inventory_On_Hand": inventory,
        "Predicted_Demand": predicted,
        "Stock_Gap":        gap,
        "Suggested_Order":  sug_order,
        "Overstock_Qty":    overstock,
        "Stockout_Qty":     max(0, gap),
        "Stock_Status":     status,
        "Demand_Factor":    factor,
        "Date":             obs_date.isoformat(),
    }


def pick_date_for_village(village_id: str) -> date:
    return date.today()


def build_village_summary(village_id, village_name, records) -> dict:
    stockout  = sum(1 for r in records if r["Stock_Status"] == "STOCKOUT")
    overstock = sum(1 for r in records if r["Stock_Status"] == "OVERSTOCK")
    balanced  = sum(1 for r in records if r["Stock_Status"] == "BALANCED")
    return {
        "Village_ID":           village_id,
        "Village_Name":         village_name,
        "total_skus":           len(records),
        "stockout_count":       stockout,
        "balanced_count":       balanced,
        "overstock_count":      overstock,
        "total_suggested_order": sum(r["Suggested_Order"] for r in records),
        "total_overstock":      sum(r["Overstock_Qty"]    for r in records),
    }


# ── GENERATE ────────────────────────────────────────────────────────────────

detail   = []
villages = []

for v_id, v_name in VILLAGES:
    obs_date = pick_date_for_village(v_id)
    records  = [generate_record(v_id, v_name, sku, obs_date) for sku in SKUS]
    detail.extend(records)
    villages.append(build_village_summary(v_id, v_name, records))

output = {"villages": villages, "detail": detail}

with open("dashboard_data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

# ── VALIDATION ──────────────────────────────────────────────────────────────

import statistics, collections

print(f"Generated {len(detail)} records across {len(villages)} villages")
print()

rank_demand = collections.defaultdict(list)
for r in detail:
    rank_demand[r["Rank"]].append(r["Predicted_Demand"])
print("Predicted_Demand by Rank (should be A > B > C):")
for rank in ["A", "B", "C"]:
    d = rank_demand[rank]
    print(f"  Rank {rank}: mean={statistics.mean(d):.1f}, stdev={statistics.stdev(d):.1f}")

print()
factor_demand = collections.defaultdict(list)
for r in detail:
    factor_demand[r["Demand_Factor"]].append(r["Predicted_Demand"])
print("Predicted_Demand by Demand_Factor:")
for f, d in sorted(factor_demand.items(), key=lambda x: -statistics.mean(x[1])):
    print(f"  {f}: mean={statistics.mean(d):.1f}")

print()
status_counts = collections.Counter(r["Stock_Status"] for r in detail)
print("Stock_Status distribution:")
for s, n in status_counts.most_common():
    print(f"  {s}: {n} ({n/len(detail)*100:.1f}%)")

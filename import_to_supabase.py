"""
Import dashboard_data.json → Supabase
รัน: python3 import_to_supabase.py
"""

import json, urllib.request, urllib.error

# ── CONFIG ───────────────────────────────────────────────────────────────────
SUPABASE_URL = "https://hrmtrplqsvjcojmobmby.supabase.co"
SUPABASE_KEY = "sb_publishable_chAqNLtW_g7dFnqzYfTNPA_H947QyXF"
HEADERS = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type":  "application/json",
    "Prefer":        "return=minimal",
}
BATCH_SIZE = 100  # insert ครั้งละ 100 rows

# ── HELPERS ──────────────────────────────────────────────────────────────────

def post(endpoint, rows, upsert=False):
    url  = f"{SUPABASE_URL}/rest/v1/{endpoint}"
    body = json.dumps(rows).encode()
    hdrs = {**HEADERS, "Prefer": "resolution=merge-duplicates,return=minimal"} if upsert else HEADERS
    req  = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req) as res:
            return res.status
    except urllib.error.HTTPError as e:
        print(f"  ❌ HTTP {e.code}: {e.read().decode()}")
        return e.code

def delete_today(table, date_str):
    url = f"{SUPABASE_URL}/rest/v1/{table}?date=eq.{date_str}"
    req = urllib.request.Request(url, headers=HEADERS, method="DELETE")
    try:
        with urllib.request.urlopen(req) as res:
            return res.status
    except urllib.error.HTTPError as e:
        print(f"  ❌ DELETE {e.code}: {e.read().decode()}")
        return e.code

def insert_batches(table, rows, upsert=False):
    total   = len(rows)
    success = 0
    for i in range(0, total, BATCH_SIZE):
        batch  = rows[i:i + BATCH_SIZE]
        status = post(table, batch, upsert=upsert)
        if status in (200, 201):
            success += len(batch)
            print(f"  ✅ {table}: {min(i+BATCH_SIZE, total)}/{total}")
        else:
            print(f"  ❌ batch {i//BATCH_SIZE + 1} failed (status {status})")
    return success

# ── LOAD ─────────────────────────────────────────────────────────────────────

print("📂 โหลดข้อมูลจาก dashboard_data.json...")
with open("dashboard_data.json", encoding="utf-8") as f:
    data = json.load(f)

villages = data["villages"]
detail   = data["detail"]
print(f"   villages: {len(villages)} rows")
print(f"   sku_detail: {len(detail)} rows\n")

# ── INSERT villages ───────────────────────────────────────────────────────────

print("🏪 Import villages (upsert)...")
village_rows = [
    {
        "village_id":            v["Village_ID"],
        "village_name":          v["Village_Name"],
        "total_skus":            v["total_skus"],
        "stockout_count":        v["stockout_count"],
        "balanced_count":        v["balanced_count"],
        "overstock_count":       v["overstock_count"],
        "total_suggested_order": v["total_suggested_order"],
        "total_overstock":       v["total_overstock"],
    }
    for v in villages
]
insert_batches("villages", village_rows, upsert=True)

# ── INSERT sku_detail ─────────────────────────────────────────────────────────

today_str = detail[0]["Date"] if detail else ""
if today_str:
    print(f"\n🗑️  ลบข้อมูลวัน {today_str} เก่าออกก่อน...")
    delete_today("sku_detail", today_str)

print("\n📦 Import sku_detail...")
detail_rows = [
    {
        "village_id":        d["Village_ID"],
        "village_name":      d["Village_Name"],
        "sku_id":            d["SKU_ID"],
        "product_name":      d["Product_Name"],
        "category":          d["Category"],
        "rank":              d["Rank"],
        "price":             d["Price"],
        "product_life_days": d["Product_Life_Days"],
        "inventory_on_hand": d["Inventory_On_Hand"],
        "predicted_demand":  d["Predicted_Demand"],
        "stock_gap":         d["Stock_Gap"],
        "suggested_order":   d["Suggested_Order"],
        "overstock_qty":     d["Overstock_Qty"],
        "stockout_qty":      d["Stockout_Qty"],
        "stock_status":      d["Stock_Status"],
        "demand_factor":     d["Demand_Factor"],
        "date":              d["Date"],
    }
    for d in detail
]
insert_batches("sku_detail", detail_rows)

print("\n🎉 เสร็จแล้ว! เช็คข้อมูลใน Supabase → Table Editor ได้เลยค่ะ")

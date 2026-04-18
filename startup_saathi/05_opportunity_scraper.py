# Databricks notebook source
# Databricks notebook source
# ============================================================
# NOTEBOOK 05: Startup Opportunity Scraper  (v4)
# StartupSaathi — Legal Navigator | Bharat Bricks Hacks 2026
#
# Changes in v4:
#   - Stores state + city from the API response
#   - ALTER TABLE adds columns if table already exists from v2/v3
#   - No RDD usage (serverless safe)
# ============================================================

# COMMAND ----------

# DBTITLE 1,Parameters & Imports
import json
import time
import logging
import requests
from datetime import datetime, timezone

from pyspark.sql.types import (
    StructType, StructField, StringType, TimestampType
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("opportunity_scraper")

dbutils.widgets.text("MAX_RECORDS", "50", "Max records to keep")
MAX_RECORDS = int(dbutils.widgets.get("MAX_RECORDS"))
logger.info(f"MAX_RECORDS = {MAX_RECORDS}")

# COMMAND ----------

# DBTITLE 1,Config
CATALOG = "startup_hackathon"
SCHEMA  = "legal_data"
TABLE   = "startup_opportunities"
FQN     = f"{CATALOG}.{SCHEMA}.{TABLE}"

API_URL = "https://seedfundapi.startupindia.gov.in:3535/api/portfoliofilter"

HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0",
    "Origin": "https://seedfund.startupindia.gov.in",
    "Referer": "https://seedfund.startupindia.gov.in/portfolio",
}

PAYLOAD = {
    "total_balance": 0,
    "grant_balance": 0,
    "sectors": [],
    "states": [],
    "cities": [],
}

# COMMAND ----------

# DBTITLE 1,Ensure catalog & schema
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")
print(f"✅ Catalog={CATALOG}  Schema={SCHEMA}")

# COMMAND ----------

# DBTITLE 1,Create or migrate table (adds state & city if missing)
# Create fresh if it doesn't exist
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {FQN} (
        name          STRING    COMMENT 'Incubator name',
        description   STRING    COMMENT 'Short summary',
        sector        STRING    COMMENT 'Sectors, comma-separated',
        link          STRING    COMMENT 'Profile URL — MERGE key',
        discovered_at TIMESTAMP COMMENT 'Last written timestamp',
        state         STRING    COMMENT 'Indian state of the incubator',
        city          STRING    COMMENT 'City of the incubator'
    )
    TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""")

# If table was created by an older notebook version (v2/v3),
# state & city columns won't exist yet — add them safely.
existing_cols = {
    row["col_name"]
    for row in spark.sql(f"DESCRIBE TABLE {FQN}").collect()
}
for col, dtype, comment in [
    ("state", "STRING", "Indian state of the incubator"),
    ("city",  "STRING", "City of the incubator"),
]:
    if col not in existing_cols:
        spark.sql(f"ALTER TABLE {FQN} ADD COLUMN {col} {dtype} COMMENT '{comment}'")
        print(f"  ➕ Added column: {col}")

print(f"✅ Table {FQN} ready with all columns.")

# COMMAND ----------

# DBTITLE 1,Fetch raw API response
def fetch_raw() -> list[dict]:
    logger.info(f"POST {API_URL}")
    resp = requests.post(API_URL, headers=HEADERS, json=PAYLOAD, timeout=10)
    resp.raise_for_status()
    body = resp.json()
    if isinstance(body, dict):
        records = body.get("data", [])
        if not records:
            for v in body.values():
                if isinstance(v, list) and len(v) > 0:
                    records = v
                    break
    elif isinstance(body, list):
        records = body
    else:
        records = []
    logger.info(f"API returned {len(records)} raw records.")
    return records

raw_records = fetch_raw()

if raw_records:
    print("=== SAMPLE RECORD (first item) ===")
    print(json.dumps(raw_records[0], indent=2, ensure_ascii=False)[:2000])
    print("\n=== ALL KEYS ===", list(raw_records[0].keys()))
else:
    print("WARNING: API returned 0 records!")

# COMMAND ----------

# DBTITLE 1,Transform  — extract all 7 fields
def _extract_name(raw: dict) -> str:
    for key in ("incubator_name", "name", "startupName", "title"):
        val = raw.get(key)
        if val:
            return str(val).strip()
    return "Unknown"


def _extract_sector(raw: dict) -> str:
    sector = raw.get("sectors") or raw.get("sector") or raw.get("sectorName")
    if not sector:
        return "all"
    if isinstance(sector, str):
        return sector.strip() or "all"
    if isinstance(sector, list):
        parts = [
            str(item.get("name") or item).strip()
            for item in sector
        ]
        return ", ".join(p for p in parts if p) or "all"
    return str(sector).strip() or "all"


def fetch_site_url(record_id: int) -> str:
    fallback = f"https://seedfund.startupindia.gov.in/incubators/{record_id}"
    try:
        resp = requests.get(
            f"https://seedfundapi.startupindia.gov.in:3535/api/getportfoliolistDesc/{record_id}",
            headers=HEADERS,
            timeout=8,
        )
        resp.raise_for_status()
        body = resp.json()
        data = body.get("data") or []
        if data:
            site_url = data[0].get("site_url") or ""
            if site_url and site_url.startswith("http"):
                return site_url.strip()
    except Exception as e:
        logger.warning(f"fetch_site_url({record_id}) failed: {e} — using fallback")
    return fallback


def _extract_link(raw: dict) -> str:
    record_id = raw.get("Id") or raw.get("id") or raw.get("incubator_user_id")
    if record_id:
        return fetch_site_url(int(record_id))
    return ""


def _extract_state(raw: dict) -> str:
    for key in ("state", "State", "stateName"):
        val = raw.get(key)
        if val:
            return str(val).strip()
    return ""


def _extract_city(raw: dict) -> str:
    for key in ("city", "City", "cityName"):
        val = raw.get(key)
        if val:
            return str(val).strip()
    return ""


def transform_records(raw_records: list[dict], max_records: int) -> list[dict]:
    now = datetime.now(timezone.utc)
    seen: set[str] = set()
    cleaned = []
    skipped_no_link = 0

    for i, raw in enumerate(raw_records):
        if i > 0 and i % 10 == 0:
            time.sleep(0.5)
        link = _extract_link(raw)
        if not link:
            skipped_no_link += 1
            continue
        if link in seen:
            continue
        seen.add(link)

        cleaned.append({
            "name":          _extract_name(raw),
            "description":   str(raw.get("description") or "").strip(),
            "sector":        _extract_sector(raw),
            "link":          link,
            "discovered_at": now,
            "state":         _extract_state(raw),
            "city":          _extract_city(raw),
        })

        if len(cleaned) >= max_records:
            break

    if skipped_no_link:
        logger.warning(f"Skipped {skipped_no_link} records with no resolvable link.")
    logger.info(f"Transformed {len(cleaned)} records.")
    return cleaned

# COMMAND ----------

# DBTITLE 1,Run pipeline
try:
    total_fetched = len(raw_records)
    records       = transform_records(raw_records, MAX_RECORDS)

    if not records:
        raise ValueError(
            f"Transform produced 0 records from {total_fetched} raw. "
            "Check the SAMPLE RECORD printed above to verify field names."
        )

    schema = StructType([
        StructField("name",          StringType(),    True),
        StructField("description",   StringType(),    True),
        StructField("sector",        StringType(),    True),
        StructField("link",          StringType(),    False),
        StructField("discovered_at", TimestampType(), True),
        StructField("state",         StringType(),    True),
        StructField("city",          StringType(),    True),
    ])
    df_new = spark.createDataFrame(records, schema=schema)
    df_new.createOrReplaceTempView("opportunities_incoming")

    print(f"\n── Preview ({len(records)} records) ──")
    df_new.select("name", "state", "city", "sector").show(5, truncate=50)

    # Existing links for diff — DataFrame.collect(), no RDD
    existing_links = {
        row["link"]
        for row in spark.sql(f"SELECT link FROM {FQN}").collect()
    }

    spark.sql(f"""
        MERGE INTO {FQN} AS target
        USING opportunities_incoming AS source
        ON target.link = source.link
        WHEN MATCHED THEN UPDATE SET
            target.name          = source.name,
            target.description   = source.description,
            target.sector        = source.sector,
            target.discovered_at = source.discovered_at,
            target.state         = source.state,
            target.city          = source.city
        WHEN NOT MATCHED THEN
            INSERT (name, description, sector, link, discovered_at, state, city)
            VALUES (source.name, source.description, source.sector,
                    source.link, source.discovered_at, source.state, source.city)
    """)

    incoming = {r["link"] for r in records}
    inserted = len(incoming - existing_links)
    updated  = len(incoming & existing_links)
    skipped  = total_fetched - len(records)

    print("\n" + "=" * 60)
    print(f"  ✅  {datetime.now(timezone.utc).isoformat()}")
    print(f"  Fetched   : {total_fetched}")
    print(f"  Written   : {len(records)}")
    print(f"  Inserted  : {inserted}")
    print(f"  Updated   : {updated}")
    print(f"  Skipped   : {skipped}")
    print("=" * 60)

except requests.exceptions.Timeout:
    logger.error("API timed out after 10 s.")
    raise
except requests.exceptions.HTTPError as e:
    logger.error(f"HTTP {e.response.status_code}: {e.response.text[:400]}")
    raise
except Exception as e:
    logger.error(f"Pipeline error: {e}", exc_info=True)
    raise

# COMMAND ----------

# DBTITLE 1,Verify
spark.sql(f"""
    SELECT name, state, city, sector, discovered_at
    FROM {FQN}
    ORDER BY discovered_at DESC
    LIMIT 5
""").show(truncate=55)
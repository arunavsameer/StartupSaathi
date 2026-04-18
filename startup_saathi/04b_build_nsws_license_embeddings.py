# Databricks notebook source
# ============================================================
# NOTEBOOK 04b: Build NSWS License FAISS Indices
# StartupSaathi — Legal Navigator | Bharat Bricks Hacks 2026
#
# PURPOSE:
#   Replaces the previous ad-hoc embedding approach for the
#   task graph.  This notebook:
#     1. Loads nsws_dpiit_data_refined_sectors.json from the
#        Unity Catalog Volume — this file already has a 'sector'
#        field per license (12 sectors, including "General").
#     2. Loads 5 state JSON files:
#          nsws_andhra_pradesh_data.json
#          nsws_delhi_data.json
#          nsws_gujarat_data.json
#          nsws_madhyapradesh_data.json
#          nsws_maharashtra_data.json
#     3. Stores ALL raw license data as Delta tables so the app
#        can join back for price, paymentType, authorityName, etc.
#     4. Embeds ONLY (licenseName + details) — nothing else.
#        DPIIT licenses are embedded PER SECTOR (one FAISS index
#        per sector).  State licenses are embedded PER STATE
#        (one FAISS index per state).
#     5. Saves every index + metadata pickle to the UC Volume so
#        the Streamlit app can load them at runtime.
#
# SEARCH BEHAVIOUR (see CELL 12):
#   - User picks a sector + optionally a state.
#   - The query is encoded and searched against:
#       (a) The matching DPIIT sector index
#       (b) Optionally the "General" DPIIT index  ← boolean flag
#       (c) The matching state index
#   - Hits are joined back to Delta tables to return full details
#     (price, paymentType, authorityName, referenceId, etc.)
#
# FREE-TIER CONSTRAINTS RESPECTED:
#   - No GPU.  CPU-only sentence-transformers.
#   - No Databricks Vector Search (paid feature).
#   - No external vector DB.
#   - Only faiss-cpu + sentence-transformers + numpy + pyspark
#     (all pip-installable on Databricks Free Edition / DBR 14+).
#   - All FAISS work happens on the driver in memory — easily
#     fits: largest index < 250 items × 384 × 4 bytes ≈ 384 KB.
#
# RUN ORDER:
#   Run once after uploading the JSON files to the Volume.
#   Re-run whenever the JSON files are updated.
#   Independent of Notebooks 03 & 04 (PDF / FAISS for PDFs).
#
# INPUT FILES  (upload to the Volume path below before running):
#   nsws_dpiit_data_refined_sectors.json
#   nsws_andhra_pradesh_data.json
#   nsws_delhi_data.json
#   nsws_gujarat_data.json
#   nsws_madhyapradesh_data.json
#   nsws_maharashtra_data.json
#
# OUTPUT (all written to ARTIFACT_VOLUME):
#   Delta tables:
#     startup_hackathon.legal_data.nsws_dpiit_licenses
#     startup_hackathon.legal_data.nsws_state_licenses
#   FAISS indices + metadata pickles:
#     dpiit_faiss_<sector>.bin  /  dpiit_meta_<sector>.pkl
#     state_faiss_<state>.bin   /  state_meta_<state>.pkl
# ============================================================

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 1 — Install dependencies
#
# faiss-cpu   : FAISS vector search, CPU-only build (free-tier safe)
# sentence-transformers : MiniLM embedding model (same as rest of app)
#
# NOTE: %pip restarts the Python kernel automatically; all subsequent
# cells run in the restarted kernel so imports resolve correctly.
# ─────────────────────────────────────────────────────────────

%pip install -q faiss-cpu sentence-transformers

# COMMAND ----------

# MAGIC %pip install faiss-cpu==1.7.4 pdfplumber==0.10.3 "protobuf<5"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 2 — Imports and constants
# ─────────────────────────────────────────────────────────────

import json
import os
import re
import pickle
import shutil
import logging
from pathlib import Path
from html.parser import HTMLParser

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from pyspark.sql import Row, SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, FloatType, BooleanType, ArrayType
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

spark = SparkSession.builder.getOrCreate()

# ── Unity Catalog paths ────────────────────────────────────────
CATALOG         = "startup_hackathon"
SCHEMA          = "legal_data"
ARTIFACT_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/model_artifacts"

# Delta table names
DPIIT_TABLE     = f"{CATALOG}.{SCHEMA}.nsws_dpiit_licenses"
STATE_TABLE     = f"{CATALOG}.{SCHEMA}.nsws_state_licenses"

# ── JSON source files (must be uploaded to the Volume before running) ──
DPIIT_JSON_PATH = f"{ARTIFACT_VOLUME}/nsws_dpiit_data_refined_sectors.json"

# Map internal key → JSON filename on Volume
STATE_JSON_FILES = {
    "andhra_pradesh" : f"{ARTIFACT_VOLUME}/nsws_andhra_pradesh_data.json",
    "delhi"          : f"{ARTIFACT_VOLUME}/nsws_delhi_data.json",
    "gujarat"        : f"{ARTIFACT_VOLUME}/nsws_gujarat_data.json",
    "madhya_pradesh" : f"{ARTIFACT_VOLUME}/nsws_madhyapradesh_data.json",
    "maharashtra"    : f"{ARTIFACT_VOLUME}/nsws_maharashtra_data.json",
}

# Human-readable display names (used by app UI)
STATE_DISPLAY_NAMES = {
    "andhra_pradesh" : "Andhra Pradesh",
    "delhi"          : "Delhi",
    "gujarat"        : "Gujarat",
    "madhya_pradesh" : "Madhya Pradesh",
    "maharashtra"    : "Maharashtra",
}

# ── Embedding model ────────────────────────────────────────────
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM        = 384   # output dimension of MiniLM-L12-v2; do not change

# ── Search behaviour toggle ────────────────────────────────────
# Set to True  → when a user picks a specific sector the search
#                also includes "General" licenses in the results.
# Set to False → only licenses tagged with the chosen sector are
#                searched (stricter, less noise).
INCLUDE_GENERAL_IN_SECTOR_SEARCH: bool = True

# ── Sectors present in the DPIIT refined JSON ─────────────────
# These are detected dynamically in CELL 4, but listed here for
# documentation and for the app's sector-picker UI.
KNOWN_DPIIT_SECTORS = [
    "AgriTech",
    "Climate/Energy",
    "DeepTech",
    "E-commerce",
    "EdTech",
    "FinTech",
    "FoodTech",
    "General",
    "HealthTech",
    "IT/SaaS",
    "Logistics",
    "Media/Gaming",
]

print("✅ Imports and constants loaded.")
print(f"   DPIIT source   : {DPIIT_JSON_PATH}")
print(f"   State sources  : {list(STATE_JSON_FILES.keys())}")
print(f"   Artifact Volume: {ARTIFACT_VOLUME}")
print(f"   INCLUDE_GENERAL: {INCLUDE_GENERAL_IN_SECTOR_SEARCH}")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 3 — Helper utilities
# ─────────────────────────────────────────────────────────────

class _HTMLStripper(HTMLParser):
    """Minimal HTML tag stripper — no external deps needed."""
    def __init__(self):
        super().__init__()
        self._parts = []

    def handle_data(self, data: str):
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts).strip()


def strip_html(raw: str) -> str:
    """Remove HTML tags and normalise whitespace."""
    if not raw:
        return ""
    s = _HTMLStripper()
    s.feed(raw)
    text = s.get_text()
    # collapse runs of whitespace / newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_embed_text(license_name: str, details: str) -> str:
    """
    Constructs the text string that will be embedded for each license.
    We embed ONLY licenseName + stripped details — no other metadata.

    Format:  "<licenseName>. <details>"

    If details is empty the period is still added so the sentence
    transformer sees a natural sentence boundary.
    """
    name    = (license_name or "").strip()
    detail  = strip_html(details or "")
    if detail:
        return f"{name}. {detail}"
    return f"{name}."


def load_and_flatten_json(path: str) -> list[dict]:
    """
    Load a multi-batch NSWS JSON (list of response objects) and return
    a flat list of individual license dicts from all batches.
    """
    with open(path, "r", encoding="utf-8") as fh:
        batches = json.load(fh)

    flat = []
    for batch in batches:
        flat.extend(batch.get("data", []))

    return flat


def safe_float(value) -> float:
    """Convert price to float; return 0.0 if None/missing."""
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


print("✅ Helper utilities defined.")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 4 — Load and flatten DPIIT JSON
# ─────────────────────────────────────────────────────────────

print(f"Loading DPIIT data from {DPIIT_JSON_PATH} …")
dpiit_raw = load_and_flatten_json(DPIIT_JSON_PATH)
print(f"✅ Loaded {len(dpiit_raw)} DPIIT licenses.")

# Detect sectors
sector_counts = {}
for lic in dpiit_raw:
    sec = lic.get("sector", "General") or "General"
    sector_counts[sec] = sector_counts.get(sec, 0) + 1

print("\n📊 DPIIT license counts by sector:")
for sec, cnt in sorted(sector_counts.items()):
    print(f"   {sec:<20} {cnt:>4} licenses")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 5 — Store DPIIT raw data as Delta table
#
# This table is the source of truth for price, paymentType,
# authorityName, referenceId, etc.  The FAISS metadata only
# stores licenseId + licenseName; all other fields are fetched
# from Delta after the vector search.
# ─────────────────────────────────────────────────────────────

DPIIT_SCHEMA = StructType([
    StructField("licenseId",          IntegerType(),  True),
    StructField("licenseName",        StringType(),   True),
    StructField("paymentType",        StringType(),   True),
    StructField("price",              FloatType(),    True),
    StructField("details",            StringType(),   True),
    StructField("sector",             StringType(),   True),
    StructField("ministryId",         IntegerType(),  True),
    StructField("departmentId",       IntegerType(),  True),
    StructField("ministryName",       StringType(),   True),
    StructField("departmentName",     StringType(),   True),
    StructField("referenceId",        StringType(),   True),
    StructField("urlTitle",           StringType(),   True),
    StructField("approvalType",       StringType(),   True),
    StructField("availableOnNsws",    BooleanType(),  True),
    StructField("scope",              BooleanType(),  True),
    StructField("publish",            BooleanType(),  True),
])

dpiit_rows = []
for lic in dpiit_raw:
    dpiit_rows.append(Row(
        licenseId       = int(lic.get("licenseId") or 0),
        licenseName     = str(lic.get("licenseName") or ""),
        paymentType     = str(lic.get("paymentType") or "FREE"),
        price           = safe_float(lic.get("price")),
        details         = strip_html(lic.get("details") or ""),
        sector          = str(lic.get("sector") or "General"),
        ministryId      = int(lic.get("ministryId") or 0),
        departmentId    = int(lic.get("departmentId") or 0),
        ministryName    = str(lic.get("ministryName") or ""),
        departmentName  = str(lic.get("departmentName") or ""),
        referenceId     = str(lic.get("referenceId") or ""),
        urlTitle        = str(lic.get("urlTitle") or ""),
        approvalType    = str(lic.get("approvalType") or ""),
        availableOnNsws = bool(lic.get("availableOnNsws", False)),
        scope           = bool(lic.get("scope", False)),
        publish         = bool(lic.get("publish", True)),
    ))

dpiit_df = spark.createDataFrame(dpiit_rows, schema=DPIIT_SCHEMA)

(
    dpiit_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(DPIIT_TABLE)
)

print(f"✅ DPIIT Delta table written: {DPIIT_TABLE}")
print(f"   Rows: {dpiit_df.count()}")
spark.sql(f"""
    SELECT sector, COUNT(*) AS cnt, SUM(CASE WHEN price > 0 THEN 1 ELSE 0 END) AS paid
    FROM {DPIIT_TABLE}
    GROUP BY sector
    ORDER BY cnt DESC
""").show()

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 6 — Load and flatten state JSON files
# ─────────────────────────────────────────────────────────────

state_licenses: dict[str, list[dict]] = {}

for state_key, json_path in STATE_JSON_FILES.items():
    print(f"Loading {state_key} from {json_path} …")
    licenses = load_and_flatten_json(json_path)
    state_licenses[state_key] = licenses
    print(f"   ✅ {len(licenses)} licenses loaded.")

print(f"\n📊 State license counts:")
for state_key, lics in state_licenses.items():
    display = STATE_DISPLAY_NAMES[state_key]
    print(f"   {display:<20} {len(lics):>4} licenses")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 7 — Store state raw data as a single Delta table
#
# All 5 states share one table; a 'state_key' column lets the
# app filter per state.  This makes join lookups uniform.
# ─────────────────────────────────────────────────────────────

STATE_SCHEMA = StructType([
    StructField("state_key",          StringType(),   False),  # e.g. "gujarat"
    StructField("stateName",          StringType(),   True),   # e.g. "Gujarat"
    StructField("licenseId",          IntegerType(),  True),
    StructField("licenseName",        StringType(),   True),
    StructField("paymentType",        StringType(),   True),
    StructField("price",              FloatType(),    True),
    StructField("details",            StringType(),   True),
    StructField("authorityId",        IntegerType(),  True),
    StructField("authorityName",      StringType(),   True),
    StructField("referenceId",        StringType(),   True),
    StructField("urlTitle",           StringType(),   True),
    StructField("approvalType",       StringType(),   True),
    StructField("availableOnNsws",    BooleanType(),  True),
    StructField("scope",              BooleanType(),  True),
    StructField("publish",            BooleanType(),  True),
])

state_rows = []
for state_key, lics in state_licenses.items():
    display_name = STATE_DISPLAY_NAMES[state_key]
    for lic in lics:
        state_rows.append(Row(
            state_key       = state_key,
            stateName       = str(lic.get("stateName") or display_name),
            licenseId       = int(lic.get("licenseId") or 0),
            licenseName     = str(lic.get("licenseName") or ""),
            paymentType     = str(lic.get("paymentType") or "FREE"),
            price           = safe_float(lic.get("price")),
            details         = strip_html(lic.get("details") or ""),
            authorityId     = int(lic.get("authorityId") or 0),
            authorityName   = str(lic.get("authorityName") or ""),
            referenceId     = str(lic.get("referenceId") or ""),
            urlTitle        = str(lic.get("urlTitle") or ""),
            approvalType    = str(lic.get("approvalType") or ""),
            availableOnNsws = bool(lic.get("availableOnNsws", False)),
            scope           = bool(lic.get("scope", False)),
            publish         = bool(lic.get("publish", True)),
        ))

state_df = spark.createDataFrame(state_rows, schema=STATE_SCHEMA)

(
    state_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(STATE_TABLE)
)

print(f"✅ State Delta table written: {STATE_TABLE}")
print(f"   Total rows: {state_df.count()}")
spark.sql(f"""
    SELECT state_key, stateName, COUNT(*) AS cnt
    FROM {STATE_TABLE}
    GROUP BY state_key, stateName
    ORDER BY cnt DESC
""").show()

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 8 — Load embedding model
#
# We use the same multilingual MiniLM the rest of the app uses
# so all embeddings live in the same semantic space.
#
# Loading once here; both DPIIT and state embedding loops share
# this single model instance (no redundant downloads).
# ─────────────────────────────────────────────────────────────

print(f"Loading embedding model: {EMBED_MODEL_NAME} …")
print("(This downloads ~120 MB on first run; cached on re-runs.)")

embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

print(f"✅ Model loaded.  Output dimension: {embed_model.get_sentence_embedding_dimension()}")
assert embed_model.get_sentence_embedding_dimension() == EMBED_DIM, (
    f"Expected {EMBED_DIM}-d embeddings; got "
    f"{embed_model.get_sentence_embedding_dimension()}. "
    f"Check EMBED_MODEL_NAME constant."
)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 9 — Build per-sector FAISS indices for DPIIT licenses
#
# For each sector we:
#   1. Collect all licenses belonging to that sector.
#   2. Build the embed_text string (licenseName + details ONLY).
#   3. Encode in batch (CPU, normalize_embeddings=True).
#   4. Build a FAISS IndexFlatIP (inner-product = cosine after L2
#      normalization — fast, exact, free-tier safe).
#   5. Save index → {ARTIFACT_VOLUME}/dpiit_faiss_{sector_key}.bin
#   6. Save metadata pickle → {ARTIFACT_VOLUME}/dpiit_meta_{sector_key}.pkl
#      Metadata stored per vector: {licenseId, licenseName, sector}
#      — all other fields (price, authority, etc.) are fetched from
#      the Delta table at query time.
#
# sector_key = sector string with spaces replaced by underscores and
#              slashes replaced by hyphens, lowercased.
#              e.g. "Climate/Energy" → "climate-energy"
#                   "IT/SaaS"        → "it-saas"
# ─────────────────────────────────────────────────────────────

def sector_to_key(sector: str) -> str:
    """Convert sector display name to a safe filesystem key."""
    return sector.lower().replace("/", "-").replace(" ", "_")


# Group licenses by sector
sector_groups: dict[str, list[dict]] = {}
for lic in dpiit_raw:
    sec = lic.get("sector") or "General"
    sector_groups.setdefault(sec, []).append(lic)

print(f"Building FAISS indices for {len(sector_groups)} DPIIT sectors …\n")

dpiit_faiss_registry: dict[str, dict] = {}  # sector → {index, metadata, sector_key}

for sector, licenses in sorted(sector_groups.items()):
    sec_key  = sector_to_key(sector)
    n        = len(licenses)

    # ── Build embed texts ────────────────────────────────────
    texts = [
        build_embed_text(lic.get("licenseName", ""), lic.get("details", ""))
        for lic in licenses
    ]

    # ── Encode — batch_size=64 is safe on CPU free-tier ─────
    print(f"  [{sector}] Encoding {n} licenses …")
    vectors = embed_model.encode(
        texts,
        batch_size        = 64,
        convert_to_numpy  = True,
        normalize_embeddings = True,   # L2-normalise → cosine via inner product
        show_progress_bar = False,
    ).astype(np.float32)

    assert vectors.shape == (n, EMBED_DIM), (
        f"Shape mismatch for sector '{sector}': {vectors.shape}"
    )

    # ── Build FAISS IndexFlatIP ──────────────────────────────
    index = faiss.IndexFlatIP(EMBED_DIM)   # inner-product (≡ cosine after normalisation)
    index.add(vectors)

    # ── Metadata list (aligned with index) ──────────────────
    metadata = [
        {
            "licenseId"  : lic.get("licenseId"),
            "licenseName": lic.get("licenseName", ""),
            "sector"     : sector,
        }
        for lic in licenses
    ]

    # ── Save to /tmp first, then copy to Volume ──────────────
    tmp_index = f"/tmp/dpiit_faiss_{sec_key}.bin"
    tmp_meta  = f"/tmp/dpiit_meta_{sec_key}.pkl"
    vol_index = f"{ARTIFACT_VOLUME}/dpiit_faiss_{sec_key}.bin"
    vol_meta  = f"{ARTIFACT_VOLUME}/dpiit_meta_{sec_key}.pkl"

    faiss.write_index(index, tmp_index)
    with open(tmp_meta, "wb") as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    shutil.copy(tmp_index, vol_index)
    shutil.copy(tmp_meta,  vol_meta)

    size_kb = os.path.getsize(tmp_index) / 1024
    print(f"     ✅ Index saved ({size_kb:.1f} KB)  → {vol_index}")

    # Store in registry for the search smoke test later
    dpiit_faiss_registry[sector] = {
        "index"     : index,
        "metadata"  : metadata,
        "sector_key": sec_key,
        "vol_index" : vol_index,
        "vol_meta"  : vol_meta,
    }

print(f"\n✅ All {len(dpiit_faiss_registry)} DPIIT sector indices built and saved.")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 10 — Build per-state FAISS indices
#
# Same approach as CELL 9 but grouped by state rather than sector.
# Each state gets:
#   {ARTIFACT_VOLUME}/state_faiss_{state_key}.bin
#   {ARTIFACT_VOLUME}/state_meta_{state_key}.pkl
#
# Metadata per vector: {licenseId, licenseName, state_key, stateName}
# Full details fetched from nsws_state_licenses Delta table at query time.
# ─────────────────────────────────────────────────────────────

print(f"Building FAISS indices for {len(state_licenses)} states …\n")

state_faiss_registry: dict[str, dict] = {}  # state_key → {index, metadata, ...}

for state_key, licenses in sorted(state_licenses.items()):
    display_name = STATE_DISPLAY_NAMES[state_key]
    n            = len(licenses)

    # ── Build embed texts ────────────────────────────────────
    texts = [
        build_embed_text(lic.get("licenseName", ""), lic.get("details", ""))
        for lic in licenses
    ]

    # ── Encode ───────────────────────────────────────────────
    print(f"  [{display_name}] Encoding {n} licenses …")
    vectors = embed_model.encode(
        texts,
        batch_size           = 64,
        convert_to_numpy     = True,
        normalize_embeddings = True,
        show_progress_bar    = False,
    ).astype(np.float32)

    assert vectors.shape == (n, EMBED_DIM), (
        f"Shape mismatch for state '{state_key}': {vectors.shape}"
    )

    # ── FAISS index ───────────────────────────────────────────
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)

    # ── Metadata list ─────────────────────────────────────────
    metadata = [
        {
            "licenseId"  : lic.get("licenseId"),
            "licenseName": lic.get("licenseName", ""),
            "state_key"  : state_key,
            "stateName"  : STATE_DISPLAY_NAMES[state_key],
        }
        for lic in licenses
    ]

    # ── Persist ───────────────────────────────────────────────
    tmp_index = f"/tmp/state_faiss_{state_key}.bin"
    tmp_meta  = f"/tmp/state_meta_{state_key}.pkl"
    vol_index = f"{ARTIFACT_VOLUME}/state_faiss_{state_key}.bin"
    vol_meta  = f"{ARTIFACT_VOLUME}/state_meta_{state_key}.pkl"

    faiss.write_index(index, tmp_index)
    with open(tmp_meta, "wb") as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    shutil.copy(tmp_index, vol_index)
    shutil.copy(tmp_meta,  vol_meta)

    size_kb = os.path.getsize(tmp_index) / 1024
    print(f"     ✅ Index saved ({size_kb:.1f} KB)  → {vol_index}")

    state_faiss_registry[state_key] = {
        "index"     : index,
        "metadata"  : metadata,
        "state_key" : state_key,
        "vol_index" : vol_index,
        "vol_meta"  : vol_meta,
    }

print(f"\n✅ All {len(state_faiss_registry)} state indices built and saved.")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 11 — Search helpers
#
# These functions replicate what the Streamlit app (rag.py /
# models.py) should call at runtime.  They are also used below
# for smoke-testing.
#
# search_dpiit(query, sector, top_k, include_general)
#   → searches the chosen sector's FAISS index
#   → optionally also searches the "General" index
#   → deduplicates by licenseId
#   → enriches results from Delta (price, authority, etc.)
#   → returns list[dict]
#
# search_state(query, state_key, top_k)
#   → searches the chosen state's FAISS index
#   → enriches results from Delta
#   → returns list[dict]
#
# search_all(query, sector, state_key, top_k, include_general)
#   → combines both searches, returns {dpiit: [...], state: [...]}
#
# RUNTIME NOTE:
#   In the actual app (rag.py) you load index/metadata from the
#   Volume files rather than from the in-memory registry built
#   here.  See CELL 13 for the Volume-load pattern.
# ─────────────────────────────────────────────────────────────

def _encode_query(query: str) -> np.ndarray:
    """Encode a single query string into a normalised L2 float32 vector."""
    vec = embed_model.encode(
        [query],
        convert_to_numpy     = True,
        normalize_embeddings = True,
    ).astype(np.float32)
    return vec  # shape (1, EMBED_DIM)


def _enrich_dpiit(license_ids: list[int]) -> dict[int, dict]:
    """
    Fetch full license details from the DPIIT Delta table for a
    list of licenseIds.  Returns a dict keyed by licenseId.
    """
    if not license_ids:
        return {}
    id_list = ", ".join(str(i) for i in license_ids)
    rows = spark.sql(f"""
        SELECT
            licenseId,
            licenseName,
            sector,
            paymentType,
            price,
            details,
            ministryName,
            departmentName,
            referenceId,
            urlTitle,
            approvalType,
            availableOnNsws
        FROM {DPIIT_TABLE}
        WHERE licenseId IN ({id_list})
    """).collect()
    return {row["licenseId"]: dict(row.asDict()) for row in rows}


def _enrich_state(license_ids: list[int], state_key: str) -> dict[int, dict]:
    """
    Fetch full license details from the State Delta table for a
    list of licenseIds within a specific state.
    """
    if not license_ids:
        return {}
    id_list = ", ".join(str(i) for i in license_ids)
    rows = spark.sql(f"""
        SELECT
            licenseId,
            licenseName,
            state_key,
            stateName,
            paymentType,
            price,
            details,
            authorityName,
            referenceId,
            urlTitle,
            approvalType,
            availableOnNsws
        FROM {STATE_TABLE}
        WHERE state_key = '{state_key}'
          AND licenseId IN ({id_list})
    """).collect()
    return {row["licenseId"]: dict(row.asDict()) for row in rows}


def search_dpiit(
    query: str,
    sector: str,
    top_k: int = 10,
    include_general: bool = INCLUDE_GENERAL_IN_SECTOR_SEARCH,
) -> list[dict]:
    """
    Search DPIIT licenses for a given sector.

    Parameters
    ----------
    query           : Free-text description of the startup / activity.
    sector          : One of KNOWN_DPIIT_SECTORS (exact match).
    top_k           : Number of results to return (per-index, before dedup).
    include_general : If True, also search the "General" sector index and
                      merge results (controlled by INCLUDE_GENERAL_IN_SECTOR_SEARCH).

    Returns
    -------
    List of result dicts sorted by cosine similarity (descending), each with:
      licenseId, licenseName, sector, paymentType, price, details,
      ministryName, departmentName, referenceId, urlTitle, approvalType,
      availableOnNsws, score (float, higher = more relevant)
    """
    q_vec = _encode_query(query)

    # Collect (score, metadata) pairs from each index we'll search
    hits: dict[int, tuple[float, dict]] = {}  # licenseId → (best_score, meta)

    def _search_one_sector(sec_name: str):
        entry = dpiit_faiss_registry.get(sec_name)
        if entry is None:
            logger.warning(f"No FAISS index found for sector '{sec_name}'. Skipping.")
            return
        idx   = entry["index"]
        metas = entry["metadata"]
        k     = min(top_k, idx.ntotal)
        if k == 0:
            return
        scores, positions = idx.search(q_vec, k)
        for score, pos in zip(scores[0], positions[0]):
            if pos < 0:
                continue
            meta = metas[pos]
            lid  = meta["licenseId"]
            # Keep only the highest-scoring hit per licenseId
            if lid not in hits or score > hits[lid][0]:
                hits[lid] = (float(score), meta)

    # Always search the requested sector
    _search_one_sector(sector)

    # Optionally add General results
    if include_general and sector != "General":
        _search_one_sector("General")

    if not hits:
        return []

    # Enrich from Delta
    enriched = _enrich_dpiit(list(hits.keys()))

    results = []
    for lid, (score, meta) in hits.items():
        row = enriched.get(lid, {})
        results.append({
            "licenseId"     : lid,
            "licenseName"   : row.get("licenseName", meta["licenseName"]),
            "sector"        : row.get("sector", meta["sector"]),
            "paymentType"   : row.get("paymentType", ""),
            "price"         : row.get("price", 0.0),
            "details"       : row.get("details", ""),
            "ministryName"  : row.get("ministryName", ""),
            "departmentName": row.get("departmentName", ""),
            "referenceId"   : row.get("referenceId", ""),
            "urlTitle"      : row.get("urlTitle", ""),
            "approvalType"  : row.get("approvalType", ""),
            "availableOnNsws": row.get("availableOnNsws", False),
            "score"         : round(score, 4),
        })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def search_state(
    query: str,
    state_key: str,
    top_k: int = 10,
) -> list[dict]:
    """
    Search state licenses for a given state.

    Parameters
    ----------
    query     : Free-text description of the startup / activity.
    state_key : One of the keys in STATE_JSON_FILES
                (e.g. "gujarat", "maharashtra").
    top_k     : Number of results to return.

    Returns
    -------
    List of result dicts sorted by cosine similarity (descending), each with:
      licenseId, licenseName, state_key, stateName, paymentType, price,
      details, authorityName, referenceId, urlTitle, approvalType,
      availableOnNsws, score (float)
    """
    entry = state_faiss_registry.get(state_key)
    if entry is None:
        raise ValueError(
            f"Unknown state_key '{state_key}'. "
            f"Valid options: {list(STATE_JSON_FILES.keys())}"
        )

    q_vec = _encode_query(query)
    idx   = entry["index"]
    metas = entry["metadata"]
    k     = min(top_k, idx.ntotal)
    if k == 0:
        return []

    scores, positions = idx.search(q_vec, k)

    # Collect licenseIds for Delta enrichment
    hits: list[tuple[float, dict]] = []
    for score, pos in zip(scores[0], positions[0]):
        if pos < 0:
            continue
        hits.append((float(score), metas[pos]))

    if not hits:
        return []

    hit_ids  = [h[1]["licenseId"] for h in hits]
    enriched = _enrich_state(hit_ids, state_key)

    results = []
    for score, meta in hits:
        lid = meta["licenseId"]
        row = enriched.get(lid, {})
        results.append({
            "licenseId"     : lid,
            "licenseName"   : row.get("licenseName", meta["licenseName"]),
            "state_key"     : state_key,
            "stateName"     : row.get("stateName", STATE_DISPLAY_NAMES[state_key]),
            "paymentType"   : row.get("paymentType", ""),
            "price"         : row.get("price", 0.0),
            "details"       : row.get("details", ""),
            "authorityName" : row.get("authorityName", ""),
            "referenceId"   : row.get("referenceId", ""),
            "urlTitle"      : row.get("urlTitle", ""),
            "approvalType"  : row.get("approvalType", ""),
            "availableOnNsws": row.get("availableOnNsws", False),
            "score"         : round(score, 4),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def search_all(
    query: str,
    sector: str,
    state_key: str,
    top_k: int = 10,
    include_general: bool = INCLUDE_GENERAL_IN_SECTOR_SEARCH,
) -> dict:
    """
    Combined entry point: search both DPIIT (sector-based) and a state index.

    Returns
    -------
    {
      "dpiit" : list[dict],   # DPIIT central licenses
      "state" : list[dict],   # State-specific licenses
    }

    The app calls this once and renders both result groups in the
    checklist (DPIIT = central requirements, state = local requirements).
    """
    dpiit_results = search_dpiit(
        query,
        sector          = sector,
        top_k           = top_k,
        include_general = include_general,
    )
    state_results = search_state(
        query,
        state_key = state_key,
        top_k     = top_k,
    )
    return {
        "dpiit": dpiit_results,
        "state": state_results,
    }


print("✅ Search helper functions defined.")
print(f"   search_dpiit(query, sector, top_k, include_general)")
print(f"   search_state(query, state_key, top_k)")
print(f"   search_all(query, sector, state_key, top_k, include_general)")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 12 — Smoke test: in-memory search
#
# Uses the in-memory registries built in CELL 9 & 10.
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("SMOKE TEST — In-memory FAISS search")
print("=" * 60)

TEST_CASES = [
    {
        "query"    : "I am building a food delivery app with cloud kitchen",
        "sector"   : "FoodTech",
        "state_key": "maharashtra",
        "top_k"    : 5,
    },
    {
        "query"    : "AgriTech startup connecting farmers to buyers with mobile app",
        "sector"   : "AgriTech",
        "state_key": "gujarat",
        "top_k"    : 5,
    },
    {
        "query"    : "FinTech wallet and payments platform for rural users",
        "sector"   : "FinTech",
        "state_key": "madhya_pradesh",
        "top_k"    : 5,
    },
]

for tc in TEST_CASES:
    print(f"\n{'─'*55}")
    print(f"Query    : {tc['query']}")
    print(f"Sector   : {tc['sector']}  |  State: {STATE_DISPLAY_NAMES[tc['state_key']]}")
    print(f"include_general={INCLUDE_GENERAL_IN_SECTOR_SEARCH}")
    print(f"{'─'*55}")

    results = search_all(
        query           = tc["query"],
        sector          = tc["sector"],
        state_key       = tc["state_key"],
        top_k           = tc["top_k"],
        include_general = INCLUDE_GENERAL_IN_SECTOR_SEARCH,
    )

    print(f"\n  DPIIT Central ({len(results['dpiit'])} hits):")
    for r in results["dpiit"]:
        paid = f"₹{r['price']:.0f}" if r["price"] and r["price"] > 0 else r["paymentType"]
        print(f"    [{r['score']:.3f}] {r['licenseName'][:65]}")
        print(f"           Sector={r['sector']}  Fee={paid}  Dept={r['departmentName'][:40]}")

    print(f"\n  State: {STATE_DISPLAY_NAMES[tc['state_key']]} ({len(results['state'])} hits):")
    for r in results["state"]:
        paid = f"₹{r['price']:.0f}" if r["price"] and r["price"] > 0 else r["paymentType"]
        print(f"    [{r['score']:.3f}] {r['licenseName'][:65]}")
        print(f"           Authority={r['authorityName'][:40]}  Fee={paid}")

print(f"\n✅ Smoke tests complete.")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 13 — Volume-load pattern for the app (rag.py reference)
#
# The Streamlit app (rag.py / models.py) must load FAISS
# indices from the Volume at startup, not from the in-memory
# registries built during this notebook run.
#
# Paste this pattern into rag.py / models.py.
# ─────────────────────────────────────────────────────────────

RUNTIME_LOAD_EXAMPLE = '''
# ── In rag.py / models.py — runtime load pattern ─────────────
import faiss, pickle, numpy as np
from sentence_transformers import SentenceTransformer

ARTIFACT_VOLUME  = "/Volumes/startup_hackathon/legal_data/model_artifacts"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM        = 384

# Toggle: include General sector in sector-specific searches?
INCLUDE_GENERAL_IN_SECTOR_SEARCH = True   # ← flip to False to tighten results

KNOWN_DPIIT_SECTORS = [
    "AgriTech", "Climate/Energy", "DeepTech", "E-commerce",
    "EdTech", "FinTech", "FoodTech", "General",
    "HealthTech", "IT/SaaS", "Logistics", "Media/Gaming",
]
SUPPORTED_STATES = [
    "andhra_pradesh", "delhi", "gujarat", "madhya_pradesh", "maharashtra"
]

def sector_to_key(s): return s.lower().replace("/", "-").replace(" ", "_")

@st.cache_resource   # Streamlit cache — load once per app session
def load_all_indices():
    model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

    dpiit_idx = {}
    for sec in KNOWN_DPIIT_SECTORS:
        key = sector_to_key(sec)
        idx  = faiss.read_index(f"{ARTIFACT_VOLUME}/dpiit_faiss_{key}.bin")
        with open(f"{ARTIFACT_VOLUME}/dpiit_meta_{key}.pkl", "rb") as f:
            meta = pickle.load(f)
        dpiit_idx[sec] = {"index": idx, "metadata": meta}

    state_idx = {}
    for sk in SUPPORTED_STATES:
        idx  = faiss.read_index(f"{ARTIFACT_VOLUME}/state_faiss_{sk}.bin")
        with open(f"{ARTIFACT_VOLUME}/state_meta_{sk}.pkl", "rb") as f:
            meta = pickle.load(f)
        state_idx[sk] = {"index": idx, "metadata": meta}

    return model, dpiit_idx, state_idx
'''

print("Reference app-load pattern (copy into rag.py):")
print(RUNTIME_LOAD_EXAMPLE)

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 14 — Verify Volume artifacts
# ─────────────────────────────────────────────────────────────

print("Verifying all artifacts on Volume …\n")

all_ok = True

# DPIIT
for sector in sorted(sector_groups.keys()):
    sec_key   = sector_to_key(sector)
    vol_index = f"{ARTIFACT_VOLUME}/dpiit_faiss_{sec_key}.bin"
    vol_meta  = f"{ARTIFACT_VOLUME}/dpiit_meta_{sec_key}.pkl"
    ok = os.path.exists(vol_index) and os.path.exists(vol_meta)
    status = "✅" if ok else "❌"
    if not ok:
        all_ok = False
    size_kb = os.path.getsize(vol_index) / 1024 if ok else 0
    print(f"  {status} DPIIT [{sector:<16}]  index={size_kb:.1f} KB")

print()

# States
for state_key in sorted(state_licenses.keys()):
    vol_index = f"{ARTIFACT_VOLUME}/state_faiss_{state_key}.bin"
    vol_meta  = f"{ARTIFACT_VOLUME}/state_meta_{state_key}.pkl"
    ok = os.path.exists(vol_index) and os.path.exists(vol_meta)
    status = "✅" if ok else "❌"
    if not ok:
        all_ok = False
    size_kb = os.path.getsize(vol_index) / 1024 if ok else 0
    display = STATE_DISPLAY_NAMES[state_key]
    print(f"  {status} State [{display:<20}]  index={size_kb:.1f} KB")

# Delta tables
for tbl in [DPIIT_TABLE, STATE_TABLE]:
    try:
        cnt = spark.sql(f"SELECT COUNT(*) AS n FROM {tbl}").collect()[0]["n"]
        print(f"\n  ✅ Delta table {tbl}  ({cnt} rows)")
    except Exception as e:
        print(f"\n  ❌ Delta table {tbl} — {e}")
        all_ok = False

print()
if all_ok:
    print("🎉  All artifacts verified successfully!")
    print(f"   DPIIT sectors : {len(sector_groups)}")
    print(f"   States        : {len(state_licenses)}")
    print(f"   Total licenses: {len(dpiit_raw) + sum(len(v) for v in state_licenses.values())}")
else:
    print("⚠️  Some artifacts are missing — check errors above.")

print("""
─────────────────────────────────────────────────
NEXT STEPS
─────────────────────────────────────────────────
1. Update rag.py / models.py using the pattern
   printed in CELL 13.
2. In the app, replace the old task-graph search
   with calls to search_all(query, sector, state).
3. Restart the Databricks App to pick up changes.
4. Re-run this notebook whenever the JSON source
   files are updated in the Volume.
─────────────────────────────────────────────────
""")
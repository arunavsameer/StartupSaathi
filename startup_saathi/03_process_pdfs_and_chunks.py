# Databricks notebook source
# ============================================================
# NOTEBOOK 03: Process PDFs, CSVs, JSON → legal_chunks
# StartupSaathi — Legal Navigator | Bharat Bricks Hacks 2026
#
# PURPOSE: Reads all documents from Unity Catalog Volumes,
#          extracts text, chunks it, embeds it, and writes
#          results to the legal_chunks Delta table.
#
# RUN: Manually whenever you add new PDFs to the Volume.
#      Already-ingested files are automatically skipped.
#      After this notebook, ALWAYS run Notebook 04 to rebuild FAISS.
# ============================================================

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 1: Install dependencies
# This MUST be the first cell. Restart Python after this runs.
# ─────────────────────────────────────────────────────────────

%pip install pdfplumber sentence-transformers

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 2: Imports and constants
# ─────────────────────────────────────────────────────────────

import os
import uuid
import json
import pickle
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from pyspark.sql import Row
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    ArrayType, FloatType, TimestampType
)

# ── Volume paths ──────────────────────────────────────────────
PDF_VOLUME      = "/Volumes/startup_hackathon/legal_data/raw_pdfs"
STRUCT_VOLUME   = "/Volumes/startup_hackathon/legal_data/raw_structured"
ARTIFACT_VOLUME = "/Volumes/startup_hackathon/legal_data/model_artifacts"
DELTA_TABLE     = "startup_hackathon.legal_data.legal_chunks"

# ── Chunking parameters ───────────────────────────────────────
CHUNK_WORD_SIZE = 300    # target words per chunk
CHUNK_OVERLAP   = 50     # overlap in words between consecutive chunks
BATCH_SIZE      = 32     # embedding batch size — safe for 15 GB RAM

# ── Embedding model ───────────────────────────────────────────
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM        = 384

print("✅ Imports complete.")
print(f"   PDF Volume:      {PDF_VOLUME}")
print(f"   Struct Volume:   {STRUCT_VOLUME}")
print(f"   Artifact Volume: {ARTIFACT_VOLUME}")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 3: Metadata configuration
#
# ADD NEW PDFs HERE by adding an entry to PDF_METADATA keyed
# by the exact filename you upload to the raw_pdfs Volume.
# sector_tag / phase_tag values must be comma-separated strings.
# ─────────────────────────────────────────────────────────────

PDF_METADATA: Dict[str, Dict[str, str]] = {
    # ── Exact filenames from your volume ───────────────────────────────────────
    
    "spice_plus.pdf": {
        "sector_tag": "all",
        "phase_tag": "incorporation"
    },

    "welcome_kit_for_tax_payers.pdf": {
        "sector_tag": "all",
        "phase_tag": "post-incorporation"
    },

    "startup_action_plan.pdf": {
        "sector_tag": "all",
        "phase_tag": "post-incorporation"
    },

    "fssai_licensing_regulations.pdf": {
        "sector_tag": "food_tech",
        "phase_tag": "post-incorporation"
    },

    "the_factories_act.pdf": {
        "sector_tag": "manufacturing",
        "phase_tag": "operations"
    },

    "contract_labour_act.pdf": {
        "sector_tag": "manufacturing,services",
        "phase_tag": "operations"
    },

    "payment_of_gratuity_act.pdf": {
        "sector_tag": "all",
        "phase_tag": "operations"
    },

    "india startup law & policy guidebook.pdf": {
        "sector_tag": "all",
        "phase_tag": "all"
    },

    "india-business-guide-start-up-to-set-up.pdf": {
        "sector_tag": "all",
        "phase_tag": "all"
    },

    "startup_india_kit.pdf": {
        "sector_tag": "all",
        "phase_tag": "all"
    },

    "startup_report_5year.pdf": {
        "sector_tag": "all",
        "phase_tag": "all"
    },

    "unified_manual_mca.pdf": {
        "sector_tag": "all",
        "phase_tag": "incorporation"
    },

    # ── Fallback ──────────────────────────────────────────────────────────────
    "__default__": {
        "sector_tag": "all",
        "phase_tag": "all"
    },
}

# Alternate names the user might upload — normalize to keys above
FILENAME_ALIASES = {
    # normalize long names
    "india startup law & policy guidebook.pdf": "india startup law & policy guidebook.pdf",
    "india-startup-law-policy-guidebook.pdf": "india startup law & policy guidebook.pdf",

    "india-business-guide-start-up-to-set-up.pdf": "india-business-guide-start-up-to-set-up.pdf",
    "india business guide start up to set up.pdf": "india-business-guide-start-up-to-set-up.pdf",

    # spice variants
    "mca_spice_guide.pdf": "spice_plus.pdf",

    # fssai variants
    "fssai_registration.pdf": "fssai_licensing_regulations.pdf",
}

def get_pdf_metadata(filename: str) -> Dict[str, str]:
    """Return sector_tag and phase_tag for a given PDF filename."""
    key = filename.lower().strip()
    key = FILENAME_ALIASES.get(key, key)
    return PDF_METADATA.get(key, PDF_METADATA["__default__"])

print("✅ Metadata config loaded.")
print(f"   Known PDFs: {[k for k in PDF_METADATA if k != '__default__']}")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 4: Helper — text chunking with overlap
# ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove excessive whitespace and non-printable characters."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\u0900-\u097F\u0A00-\u0A7F]', ' ', text)  # keep ASCII + Devanagari + Gurmukhi
    return text.strip()

def chunk_text(text: str, chunk_words: int = CHUNK_WORD_SIZE, overlap_words: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks of approximately chunk_words words.
    Returns a list of chunk strings.
    """
    words = text.split()
    if len(words) == 0:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 50:   # ignore tiny fragments
            chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_words - overlap_words  # slide forward with overlap

    return chunks

# Smoke test
_test = chunk_text("word " * 700, chunk_words=300, overlap_words=50)
print(f"✅ Chunker test: 700 words → {len(_test)} chunks (expected ~3)")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 5: Load embedding model (CPU only)
# This model is ~450 MB and loads once for the whole notebook.
# ─────────────────────────────────────────────────────────────

print(f"Loading embedding model: {EMBED_MODEL_NAME}")
print("This may take 1–2 minutes on first run (downloads ~450 MB)...")

embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

print(f"✅ Embedding model loaded. Output dimension: {embed_model.get_sentence_embedding_dimension()}")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 6: Embedding helper with batching
# ─────────────────────────────────────────────────────────────

def embed_texts(texts: List[str], batch_size: int = BATCH_SIZE) -> List[List[float]]:
    """
    Embed a list of strings in batches.
    Returns a list of Python float lists (Delta-compatible ARRAY<FLOAT>).
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = embed_model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        all_embeddings.extend(embeddings.tolist())  # convert numpy → Python list
        print(f"   Embedded batch {i // batch_size + 1} ({len(all_embeddings)}/{len(texts)} chunks)")
    return all_embeddings

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 7: Check already-ingested source files (deduplication)
# Re-running this notebook is always safe — skips existing files.
# ─────────────────────────────────────────────────────────────

try:
    existing_df = spark.sql(f"SELECT DISTINCT source_file FROM {DELTA_TABLE}")
    already_ingested = {row["source_file"] for row in existing_df.collect()}
    print(f"✅ Already ingested {len(already_ingested)} source file(s):")
    for f in sorted(already_ingested):
        print(f"   ↳ {f}")
except Exception as e:
    already_ingested = set()
    print(f"ℹ️  legal_chunks table is empty or missing. Starting fresh. ({e})")

# COMMAND ----------

# DBTITLE 1,Process PDF files from Volume
# ─────────────────────────────────────────────────────────────
# CELL 8: Process PDF files from the Volume (FINAL FIX)
# ─────────────────────────────────────────────────────────────

import tempfile

def process_pdf_from_dbfs(pdf_path: str, filename: str) -> List[Dict]:
    meta = get_pdf_metadata(filename)
    rows = []

    try:
        # ✅ Read file as binary using Spark
        binary_df = spark.read.format("binaryFile").load(pdf_path)
        file_bytes = binary_df.select("content").collect()[0][0]

        # Write to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        # Process with pdfplumber
        with pdfplumber.open(tmp_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text()
                if not raw_text:
                    continue

                cleaned = clean_text(raw_text)
                if len(cleaned.split()) < 20:
                    continue

                chunks = chunk_text(cleaned)
                for chunk in chunks:
                    rows.append({
                        "chunk_id":    str(uuid.uuid4()),
                        "source_file": filename,
                        "page_number": page_num,
                        "chunk_text":  chunk,
                        "sector_tag":  meta["sector_tag"],
                        "phase_tag":   meta["phase_tag"],
                    })

    except Exception as e:
        print(f"   ⚠️  Error reading {filename}: {e}")

    return rows


# List all PDFs
pdf_files = [
    f.path for f in dbutils.fs.ls(PDF_VOLUME)
    if f.name.lower().endswith(".pdf")
]

print(f"✅ Found {len(pdf_files)} PDF(s) in Volume:")
for p in pdf_files:
    print(f"   ↳ {p}")


# Process PDFs
new_pdf_rows = []

for pdf_path in pdf_files:
    filename = os.path.basename(pdf_path)

    if filename in already_ingested:
        print(f"\n⏭️  Skipping (already ingested): {filename}")
        continue

    print(f"\n📄 Processing: {filename}")
    rows = process_pdf_from_dbfs(pdf_path, filename)
    print(f"   → {len(rows)} chunks extracted")

    new_pdf_rows.extend(rows)

print(f"\n✅ Total new PDF chunks (before embedding): {len(new_pdf_rows)}")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 9: Process BNS CSV (FIXED VERSION)
# ─────────────────────────────────────────────────────────────

CSV_FILENAME = "bns_sections.csv"   # ✅ match your uploaded file

new_csv_rows = []

if CSV_FILENAME in already_ingested:
    print(f"⏭️  Skipping {CSV_FILENAME} (already ingested)")

else:
    print(f"📊 Processing {CSV_FILENAME}...")

    try:
        # ✅ Read directly using Spark (no /dbfs)
        bns_df = spark.read.option("header", "true").option("inferSchema", "true").csv(
            f"{STRUCT_VOLUME}/{CSV_FILENAME}"
        )

        print("Schema:")
        bns_df.printSchema()

        col_names = [c.lower() for c in bns_df.columns]

        # 🔍 Detect text column
        text_col = None
        for candidate in ["description", "section_text", "text", "content", "punishment", "offence"]:
            if candidate in col_names:
                text_col = bns_df.columns[col_names.index(candidate)]
                break

        if text_col is None:
            text_col = bns_df.columns[-1]

        # 🔍 Detect section column
        section_col = None
        for candidate in ["section", "section_number", "id", "section_id", "chapter"]:
            if candidate in col_names:
                section_col = bns_df.columns[col_names.index(candidate)]
                break

        print(f"   Using text column: '{text_col}', section column: '{section_col}'")

        # ⚠️ LIMIT rows to avoid memory crash (important for free tier)
        rows_data = bns_df.limit(1000).collect()

        for i, row in enumerate(rows_data):
            raw_text = str(row[text_col]) if row[text_col] else ""

            if not raw_text or raw_text.lower() in ("null", "none", "nan"):
                continue

            prefix = f"BNS Section {row[section_col]}: " if section_col and row[section_col] else ""
            full_text = clean_text(prefix + raw_text)

            new_csv_rows.append({
                "chunk_id":    str(uuid.uuid4()),
                "source_file": CSV_FILENAME,
                "page_number": i + 1,
                "chunk_text":  full_text,
                "sector_tag":  "all",
                "phase_tag":   "all",
            })

        print(f"   ✅ {len(new_csv_rows)} chunks from {CSV_FILENAME}")

    except Exception as e:
        print(f"   ❌ Error processing {CSV_FILENAME}: {e}")
        import traceback
        traceback.print_exc()

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 10: Process myscheme_startup.json (FIXED VERSION)
# ─────────────────────────────────────────────────────────────

JSON_FILENAME = "myscheme_startup.json"
JSON_PATH = f"{STRUCT_VOLUME}/{JSON_FILENAME}"

new_json_rows = []

if JSON_FILENAME in already_ingested:
    print(f"⏭️  Skipping {JSON_FILENAME} (already ingested)")

else:
    print(f"📋 Processing {JSON_FILENAME}...")

    try:
        # ✅ Read JSON safely from DBFS (as string → valid for JSON)
        json_str = dbutils.fs.head(JSON_PATH, 1024 * 1024 * 10)  # up to 10MB
        raw_data = json.loads(json_str)

        # Handle both list-of-dicts and dict structures
        if isinstance(raw_data, list):
            schemes = raw_data
        elif isinstance(raw_data, dict):
            for key in ["schemes", "items", "data", "results"]:
                if key in raw_data:
                    schemes = raw_data[key]
                    break
            else:
                schemes = list(raw_data.values()) if raw_data else []
        else:
            schemes = []

        print(f"   Found {len(schemes)} scheme entries")

        # ⚠️ Limit to avoid memory issues (important)
        # Ensure schemes is a list
        if isinstance(schemes, dict):
            schemes = list(schemes.values())

        # Limit size safely
        schemes = list(schemes)[:1000]

        for i, scheme in enumerate(schemes):
            if not isinstance(scheme, dict):
                continue

            name = (
                scheme.get("schemeName")
                or scheme.get("name")
                or scheme.get("title")
                or f"Scheme {i+1}"
            )

            description = (
                scheme.get("schemeDescription")
                or scheme.get("description")
                or scheme.get("benefit")
                or scheme.get("details")
                or ""
            )

            eligibility = scheme.get("eligibility") or scheme.get("criteria") or ""
            benefits = scheme.get("benefits") or scheme.get("benefit_details") or ""

            text_parts = [f"Scheme: {name}"]

            if description:
                text_parts.append(f"Description: {description}")
            if eligibility:
                text_parts.append(f"Eligibility: {eligibility}")
            if benefits:
                text_parts.append(f"Benefits: {benefits}")

            full_text = clean_text(" | ".join(text_parts))

            # Debug print (only first few)
            if i < 3:
                print(f"   DEBUG [{i}]: {full_text[:100]}")

            # Relax filter (VERY IMPORTANT)
            if len(full_text.strip()) == 0:
                continue

            new_json_rows.append({
                "chunk_id":    str(uuid.uuid4()),
                "source_file": JSON_FILENAME,
                "page_number": i + 1,
                "chunk_text":  full_text,
                "sector_tag":  "all",
                "phase_tag":   "all",
            })

        print(f"   ✅ {len(new_json_rows)} chunks from {JSON_FILENAME}")

    except Exception as e:
        print(f"   ❌ Error processing {JSON_FILENAME}: {e}")
        import traceback
        traceback.print_exc()

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 11: Combine all new rows and embed them
# ─────────────────────────────────────────────────────────────

all_new_rows = new_pdf_rows + new_csv_rows + new_json_rows
print(f"\n📦 Total new chunks to embed: {len(all_new_rows)}")
print(f"   From PDFs:  {len(new_pdf_rows)}")
print(f"   From CSV:   {len(new_csv_rows)}")
print(f"   From JSON:  {len(new_json_rows)}")

if len(all_new_rows) == 0:
    print("\n✅ Nothing new to ingest. All files already in legal_chunks.")
    dbutils.notebook.exit("Nothing new to ingest.")

# ── Embed all new chunks ──────────────────────────────────────
print("\n🔢 Starting embedding... (this takes a few minutes)")
texts_to_embed = [row["chunk_text"] for row in all_new_rows]
embeddings = embed_texts(texts_to_embed)

# Attach embeddings back to rows
for row, emb in zip(all_new_rows, embeddings):
    row["embedding"] = emb  # Python list of floats

print(f"\n✅ Embedding complete. {len(all_new_rows)} chunks embedded.")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 12: Convert to Spark DataFrame and write to Delta
# ─────────────────────────────────────────────────────────────

now = datetime.utcnow()
for row in all_new_rows:
    row["ingested_at"] = now

# Define explicit Delta schema
delta_schema = StructType([
    StructField("chunk_id",    StringType(),              False),
    StructField("source_file", StringType(),              False),
    StructField("page_number", IntegerType(),             True),
    StructField("chunk_text",  StringType(),              True),
    StructField("embedding",   ArrayType(FloatType()),    True),
    StructField("sector_tag",  StringType(),              True),
    StructField("phase_tag",   StringType(),              True),
    StructField("ingested_at", TimestampType(),           True),
])

# Build PySpark Row objects
spark_rows = [
    Row(
        chunk_id    = r["chunk_id"],
        source_file = r["source_file"],
        page_number = r["page_number"],
        chunk_text  = r["chunk_text"],
        embedding   = [float(x) for x in r["embedding"]],  # ensure Python floats
        sector_tag  = r["sector_tag"],
        phase_tag   = r["phase_tag"],
        ingested_at = r["ingested_at"],
    )
    for r in all_new_rows
]

chunks_df = spark.createDataFrame(spark_rows, schema=delta_schema)
print(f"✅ DataFrame created: {chunks_df.count()} rows")

# COMMAND ----------

# Append to Delta table (not overwrite — preserves existing chunks)
(
    chunks_df.write
    .format("delta")
    .mode("append")
    .saveAsTable(DELTA_TABLE)
)

print(f"✅ Written to {DELTA_TABLE}")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 13: Verify the final state of legal_chunks
# ─────────────────────────────────────────────────────────────

spark.sql(f"""
    SELECT source_file, COUNT(*) AS num_chunks, MIN(ingested_at) AS first_ingested
    FROM {DELTA_TABLE}
    GROUP BY source_file
    ORDER BY source_file
""").show(truncate=50)

total = spark.sql(f"SELECT COUNT(*) AS total FROM {DELTA_TABLE}").collect()[0]["total"]
print(f"\n✅ Total chunks in legal_chunks table: {total}")
print("\n🚀 Next step: Run Notebook 04 to rebuild the FAISS index.")
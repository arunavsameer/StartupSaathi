# Databricks notebook source
# ============================================================
# NOTEBOOK 02b: Generate Task Graph from Data (Embedding-Driven)
# StartupSaathi — Legal Navigator | Bharat Bricks Hacks 2026
#
# PURPOSE:
#   Replaces the hardcoded TASKS list in 02_populate_task_graph.py.
#   Instead of hand-writing every task, this notebook:
#     1. Loads indian_startup_requirements.json (510 compliance items)
#        and the state-specific nsws_*_data.json files from the
#        uploaded JSON datasets.
#     2. Embeds every requirement using the same MiniLM model the
#        rest of the app already uses.
#     3. Clusters similar requirements into coherent task groups
#        using cosine-similarity bucketing (no external ML libs
#        beyond sentence-transformers — free-tier safe).
#     4. Infers sector_filter, size_filter, phase, prereq_ids,
#        and est_days from the data rather than hard-coding them.
#     5. Writes the resulting tasks into the task_graph Delta table
#        (same schema as before — the app never needs to change).
#
# FREE-TIER CONSTRAINTS RESPECTED:
#   - No GPU / CUDA. Embedding runs on CPU.
#   - No vector store cluster needed (in-memory numpy only).
#   - No extra paid packages — only sentence-transformers +
#     scikit-learn (both pip-installable on free DBR).
#   - Driver-memory only (no Spark UDFs over large frames).
#   - ~512 tasks × 384 floats × 4 bytes ≈ 786 KB — trivially safe.
#
# RUN: Once. Safe to re-run — it REPLACES the task_graph table.
#      Notebook 03 (PDFs) and Notebook 04 (FAISS) are independent.
#
# INPUT FILES (upload to Databricks Volumes before running):
#   /Volumes/startup_hackathon/legal_data/model_artifacts/
#       indian_startup_requirements.json
#       nsws_dpiit_data.json
#       States/  (directory with all nsws_*_data.json files)
#
# OUTPUT:  startup_hackathon.legal_data.task_graph  (Delta table)
# ============================================================

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 1: Install dependencies
# Only sentence-transformers + scikit-learn needed.
# Both are safe on Databricks Free Edition (DBR 14+).
# ─────────────────────────────────────────────────────────────

%pip install -q sentence-transformers scikit-learn

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 2: Imports & constants
# ─────────────────────────────────────────────────────────────

import json
import os
import re
import uuid
import zipfile
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────
# Adjust ARTIFACT_VOLUME if your Volume path differs.
ARTIFACT_VOLUME = "/Volumes/startup_hackathon/legal_data/model_artifacts"

REQUIREMENTS_JSON  = f"{ARTIFACT_VOLUME}/indian_startup_requirements.json"
DPIIT_JSON         = f"{ARTIFACT_VOLUME}/nsws_dpiit_data.json"
STATES_DIR         = f"{ARTIFACT_VOLUME}/States"      # folder with nsws_*_data.json

DELTA_TABLE = "startup_hackathon.legal_data.task_graph"

# ── Embedding model — identical to the one used in models.py ──
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM        = 384

# ── Clustering threshold ───────────────────────────────────
# Two requirements are merged into one task if their embedding
# cosine similarity exceeds this value.  Tune between 0.80–0.92:
#   Higher → more granular tasks (closer to 1-per-requirement)
#   Lower  → more aggressive merging (fewer, broader tasks)
MERGE_THRESHOLD = 0.85

# ── Sector & size mapping ──────────────────────────────────
# Maps the startup_types strings from the JSON → the sector_filter
# keys used by app.py (SECTOR_OPTIONS). Unmapped types fall back
# to "all".
STARTUP_TYPE_TO_SECTOR = {
    "Tech":             "tech",
    "SaaS":             "tech",
    "AI/ML":            "tech",
    "Mobile App":       "tech",
    "IoT":              "tech",
    "Deep Tech":        "tech",
    "Smart Home":       "tech",
    "Cybersecurity":    "tech",
    "Data Centers":     "tech",
    "Semiconductor":    "tech",
    "Robotics":         "tech",
    "SpaceTech":        "tech",
    "Drone Tech":       "tech",
    "EdTech":           "edtech",
    "Coaching":         "edtech",
    "Skill Development":"edtech",
    "Schools":          "edtech",
    "FinTech":          "fintech",
    "WealthTech":       "fintech",
    "InsurTech":        "fintech",
    "Lending":          "fintech",
    "Banking Tech":     "fintech",
    "Crypto":           "fintech",
    "Web3":             "fintech",
    "Fund Managers":    "fintech",
    "Venture Capital":  "fintech",
    "HealthTech":       "healthcare",
    "MedTech":          "healthcare",
    "Pharma":           "healthcare",
    "Diagnostics":      "healthcare",
    "Clinics":          "healthcare",
    "Telemedicine":     "healthcare",
    "Ayush":            "healthcare",
    "Animal Health":    "healthcare",
    "Senior Care":      "healthcare",
    "Food":             "food_tech",
    "Food & Beverage":  "food_tech",
    "Food Processing":  "food_tech",
    "D2C":              "ecommerce",
    "E-commerce":       "ecommerce",
    "Retail":           "ecommerce",
    "FMCG":             "ecommerce",
    "Manufacturing":    "manufacturing",
    "Electronics":      "manufacturing",
    "Automotive":       "manufacturing",
    "Chemical":         "manufacturing",
    "Textile":          "manufacturing",
    "Packaging":        "manufacturing",
    "Printing":         "manufacturing",
    "Agri-tech":        "agri",
    "Agriculture":      "agri",
    "Logistics":        "logistics",
    "Delivery":         "logistics",
    "Air Cargo":        "logistics",
    "All":              "all",
}

# Phase inference: keywords in category/name → phase label
PHASE_KEYWORDS = {
    "incorporation": [
        "incorporat", "register company", "name reserv", "spice",
        "certificate of incorporat", "mca", "roc", "pan", "tan",
        "bank account", "memorandum", "articles of association",
        "company formation", "din", "dsc",
    ],
    "post-incorporation": [
        "gst", "dpiit", "startup india", "shops and establishment",
        "professional tax", "fssai", "trademark", "copyright", "patent",
        "import export code", "iec", "msme udyam", "udyog aadhaar",
        "drug license", "pollution noc", "fire noc", "sebi",
        "rbi", "nbfc", "irdai", "trai", "dot",
    ],
    "operations": [
        "epf", "esic", "provident fund", "gratuity", "maternity",
        "labour", "labor", "minimum wages", "hr", "payroll",
        "annual return", "compliance calendar", "income tax filing",
        "tds", "advance tax", "audit", "annual filing",
        "director kyc", "annual general meeting", "agm",
    ],
}

# Estimated days heuristic by phase
PHASE_EST_DAYS = {
    "incorporation":      5,
    "post-incorporation": 7,
    "operations":        10,
}

print("✅ Imports and constants ready.")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 3: Load source data
#
# Three sources:
#   A. indian_startup_requirements.json   — 510 national requirements
#   B. nsws_dpiit_data.json              — central ministry licenses
#   C. States/nsws_*_data.json           — state-level licenses
#
# We normalise all three into a flat list of dicts with keys:
#   id, name, description, category, startup_types,
#   states_required, official_portal, source
# ─────────────────────────────────────────────────────────────

def _strip_html(text: str) -> str:
    """Remove HTML tags from NSWS 'details' fields."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def load_national_requirements(path: str) -> list[dict]:
    """Load indian_startup_requirements.json → normalised list."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for r in data.get("requirements", []):
        items.append({
            "id":              f"NR_{r['id']}",
            "name":            r.get("name", ""),
            "description":     r.get("description", ""),
            "category":        r.get("category", ""),
            "startup_types":   r.get("startup_types", ["All"]),
            "states_required": r.get("states_required", ["All"]),
            "official_portal": r.get("official_portal", ""),
            "source":          "national",
        })
    logger.info(f"Loaded {len(items)} national requirements.")
    return items


def load_nsws_json(path: str, source_tag: str) -> list[dict]:
    """
    Load an NSWS JSON file (dpiit or state-specific).
    Each file is a list of response dicts; each has a 'data' list
    of license objects.  We flatten them all.
    """
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    items = []
    seen_ids = set()
    for entry in entries:
        for lic in entry.get("data", []):
            lid = lic.get("licenseId")
            if lid in seen_ids:
                continue
            seen_ids.add(lid)
            name   = lic.get("licenseName", "").strip()
            detail = _strip_html(lic.get("details", ""))
            dept   = lic.get("departmentName", "")
            scope  = lic.get("scope", False)  # True = state-level
            if not name:
                continue
            items.append({
                "id":              f"NSWS_{source_tag}_{lid}",
                "name":            name,
                "description":     detail[:400] if detail else name,
                "category":        dept or source_tag,
                "startup_types":   ["All"],
                "states_required": ["All"] if not scope else [source_tag],
                "official_portal": f"https://www.nsws.gov.in/",
                "source":          source_tag,
            })
    logger.info(f"Loaded {len(items)} items from {source_tag}.")
    return items


def load_all_sources() -> list[dict]:
    all_items = []

    # A: National requirements
    if os.path.exists(REQUIREMENTS_JSON):
        all_items.extend(load_national_requirements(REQUIREMENTS_JSON))
    else:
        logger.warning(f"Not found: {REQUIREMENTS_JSON}")

    # B: DPIIT / central ministry licenses
    if os.path.exists(DPIIT_JSON):
        all_items.extend(load_nsws_json(DPIIT_JSON, "DPIIT"))
    else:
        logger.warning(f"Not found: {DPIIT_JSON}")

    # C: State-level licenses
    if os.path.isdir(STATES_DIR):
        for fname in sorted(os.listdir(STATES_DIR)):
            if not fname.endswith(".json"):
                continue
            # e.g. nsws_maharashtra_data.json → "maharashtra"
            state_tag = fname.replace("nsws_", "").replace("_data.json", "")
            fpath = os.path.join(STATES_DIR, fname)
            all_items.extend(load_nsws_json(fpath, state_tag))
    else:
        logger.warning(f"States dir not found: {STATES_DIR}")

    logger.info(f"Total raw items loaded: {len(all_items)}")
    return all_items


raw_items = load_all_sources()
print(f"✅ Loaded {len(raw_items)} raw compliance items from all sources.")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 4: Embed all items
#
# We embed  name + " — " + description  (same strategy used for
# legal chunks in Notebook 03) so the vector captures both the
# label and the regulatory meaning of each item.
#
# At 512+ items × 384 dims × 4 bytes ≈ < 800 KB — trivially
# within free-tier driver memory limits.
# ─────────────────────────────────────────────────────────────

print(f"Loading embedding model: {EMBED_MODEL_NAME}  (CPU-only) ...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
print("✅ Model loaded.")

texts = [
    f"{item['name']} — {item['description'][:300]}"
    for item in raw_items
]

print(f"Embedding {len(texts)} items ...")
embeddings = embed_model.encode(
    texts,
    batch_size=64,
    convert_to_numpy=True,
    normalize_embeddings=True,   # unit vectors → cosine ≡ dot product
    show_progress_bar=True,
)
embeddings = embeddings.astype("float32")
print(f"✅ Embeddings shape: {embeddings.shape}")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 5: Cluster / merge near-duplicate requirements
#
# Strategy: greedy single-pass clustering.
#   - Iterate items in order.
#   - If the item's cosine similarity to an existing cluster
#     centroid exceeds MERGE_THRESHOLD, add it to that cluster.
#   - Otherwise, start a new cluster.
#
# This is O(n × k) where k = number of clusters formed.
# For n ≈ 1000 items this completes in < 2 seconds on CPU.
#
# Why not KMeans?  KMeans needs you to choose k in advance,
# which is awkward for compliance data.  The greedy approach
# is parameter-free and produces naturally-sized clusters.
# ─────────────────────────────────────────────────────────────

print("Clustering requirements ...")

cluster_centroids = []   # list of 384-dim numpy arrays
cluster_members   = []   # list of lists of raw_items indices

for i, emb in enumerate(embeddings):
    if not cluster_centroids:
        cluster_centroids.append(emb.copy())
        cluster_members.append([i])
        continue

    # Compute cosine similarity to all existing centroids
    centroid_matrix = np.stack(cluster_centroids, axis=0)  # (k, 384)
    sims = cosine_similarity(emb.reshape(1, -1), centroid_matrix)[0]  # (k,)
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    if best_sim >= MERGE_THRESHOLD:
        # Add to existing cluster; update centroid as running mean
        cluster_members[best_idx].append(i)
        n = len(cluster_members[best_idx])
        cluster_centroids[best_idx] = (
            cluster_centroids[best_idx] * (n - 1) / n + emb / n
        )
    else:
        cluster_centroids.append(emb.copy())
        cluster_members.append([i])

print(f"✅ Formed {len(cluster_members)} task clusters "
      f"from {len(raw_items)} raw items  "
      f"(merge threshold: {MERGE_THRESHOLD}).")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 6: Infer task metadata from each cluster
#
# For every cluster we pick:
#   task_name    → name of the most-representative member
#                  (the one with highest similarity to centroid)
#   description  → combined from up to 3 members
#   authority    → from departmentName / category heuristic
#   portal_url   → first non-empty official_portal in cluster
#   sector_filter→ union of all startup_types, mapped to sectors
#   size_filter  → ["all"] (size data not in source JSONs;
#                  can be extended with an employee-count field)
#   phase        → inferred from keywords in name+description
#   est_days     → phase-based heuristic
#   prereq_ids   → [] at this stage (filled in CELL 7)
# ─────────────────────────────────────────────────────────────

def infer_phase(name: str, description: str, category: str) -> str:
    """Return 'incorporation' | 'post-incorporation' | 'operations'."""
    combined = (name + " " + description + " " + category).lower()
    scores = {phase: 0 for phase in PHASE_KEYWORDS}
    for phase, keywords in PHASE_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                scores[phase] += 1
    best = max(scores, key=scores.get)
    # If nothing matched, default to post-incorporation
    return best if scores[best] > 0 else "post-incorporation"


def infer_sectors(startup_types: list[str]) -> list[str]:
    """Map startup_types strings → sector keys used by app.py."""
    sectors = set()
    for t in startup_types:
        mapped = STARTUP_TYPE_TO_SECTOR.get(t)
        if mapped:
            sectors.add(mapped)
    if not sectors or "all" in sectors:
        return ["all"]
    return sorted(sectors)


def infer_authority(category: str, name: str) -> str:
    """Derive the governing authority from category / name keywords."""
    text = (category + " " + name).upper()
    if "MCA" in text or "INCORPORAT" in text or "ROC" in text:
        return "MCA"
    if "GST" in text:
        return "GSTN"
    if "INCOME TAX" in text or "TDS" in text or "PAN" in text or "TAN" in text:
        return "Income Tax Dept"
    if "FSSAI" in text or "FOOD" in text:
        return "FSSAI"
    if "SEBI" in text:
        return "SEBI"
    if "RBI" in text or "NBFC" in text or "FEMA" in text:
        return "RBI"
    if "IRDAI" in text or "INSUR" in text:
        return "IRDAI"
    if "EPF" in text or "PROVIDENT" in text:
        return "EPFO"
    if "ESIC" in text:
        return "ESIC"
    if "TRADEMARK" in text or "PATENT" in text or "IPR" in text:
        return "IP India"
    if "DPIIT" in text or "STARTUP INDIA" in text:
        return "DPIIT"
    if "LABOUR" in text or "LABOR" in text or "SHOP" in text:
        return "State Labour Dept"
    if "TRAI" in text or "DOT" in text or "TELECOM" in text:
        return "TRAI / DoT"
    if "NSWS" in text or "MINISTRY" in text:
        return category.split("–")[-1].strip() if "–" in category else category
    return "Regulatory Authority"


def build_task_from_cluster(cluster_idx: int, member_indices: list[int],
                             centroid: np.ndarray) -> dict:
    members = [raw_items[i] for i in member_indices]

    # Pick representative: highest cosine sim to centroid
    member_embs   = embeddings[member_indices]           # (m, 384)
    sims_to_cent  = cosine_similarity(
        centroid.reshape(1, -1), member_embs
    )[0]
    rep_local_idx = int(np.argmax(sims_to_cent))
    rep           = members[rep_local_idx]

    task_name    = rep["name"]
    category     = rep["category"]
    portal_url   = next((m["official_portal"] for m in members
                         if m.get("official_portal")), "")

    # Combine descriptions (up to 3 most distinct members)
    descriptions = list({m["description"] for m in members if m["description"]})[:3]
    description  = " | ".join(descriptions)[:600]

    # Union startup_types → sectors
    all_types = []
    for m in members:
        all_types.extend(m.get("startup_types", []))
    sector_filter = infer_sectors(list(set(all_types)))

    phase     = infer_phase(task_name, description, category)
    authority = infer_authority(category, task_name)
    est_days  = PHASE_EST_DAYS.get(phase, 7)

    # Compact task_id — zero-padded cluster index
    task_id = f"T{cluster_idx + 1:03d}"

    return {
        "task_id":       task_id,
        "task_name":     task_name,
        "description":   description,
        "authority":     authority,
        "portal_url":    portal_url,
        "prereq_ids":    [],        # filled in CELL 7
        "sector_filter": sector_filter,
        "size_filter":   ["all"],   # extend here if size data is added
        "phase":         phase,
        "est_days":      est_days,
        # Internal — used for prereq inference; stripped before Delta write
        "_centroid":     centroid,
        "_category":     category,
        "_raw_name":     task_name.lower(),
    }


print("Building task dicts from clusters ...")
tasks = [
    build_task_from_cluster(idx, members, cluster_centroids[idx])
    for idx, members in enumerate(cluster_members)
]
print(f"✅ Built {len(tasks)} candidate tasks.")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 7: Infer prerequisite relationships via embeddings
#
# Approach: for every pair of tasks where one is in an earlier
# phase than the other, check if their embedding similarity
# exceeds PREREQ_SIM_THRESHOLD.  If so, the later task depends
# on the earlier one.
#
# Additionally we hard-wire a handful of well-known structural
# dependencies (e.g. "open bank account" always follows "get
# COI") that are too fundamental to leave to embeddings alone.
#
# PREREQ_SIM_THRESHOLD is deliberately LOW (0.55) because we
# want to capture loose thematic coupling (e.g. GST → all
# tax-related tasks), not just near-duplicates.
# ─────────────────────────────────────────────────────────────

PREREQ_SIM_THRESHOLD = 0.55

PHASE_ORDER = {"incorporation": 0, "post-incorporation": 1, "operations": 2}

# Keyword-based hard-wire rules:
# If task A's name contains KEY_A and task B's name contains KEY_B
# and A is in an earlier phase, B gets A as a prereq.
HARD_WIRE_RULES = [
    # (earlier_keyword, later_keyword)
    ("incorporat",          "gst"),
    ("incorporat",          "dpiit"),
    ("incorporat",          "shop"),
    ("incorporat",          "trademark"),
    ("incorporat",          "patent"),
    ("incorporat",          "udyam"),
    ("pan",                 "bank account"),
    ("bank account",        "gst"),
    ("certificate of incorp","pan"),
    ("spice",               "certificate of incorp"),
    ("name reserv",         "spice"),
    ("gst",                 "professional tax"),
    ("bank account",        "esic"),
    ("bank account",        "epf"),
    ("esic",                "gratuity"),
]

print("Inferring prerequisite edges ...")

task_centroids = np.stack([t["_centroid"] for t in tasks], axis=0)  # (T, 384)

for i, task_b in enumerate(tasks):
    phase_b = PHASE_ORDER.get(task_b["phase"], 1)
    prereqs = set()

    for j, task_a in enumerate(tasks):
        if i == j:
            continue
        phase_a = PHASE_ORDER.get(task_a["phase"], 1)
        if phase_a >= phase_b:
            continue  # only earlier-phase tasks can be prereqs

        # Embedding similarity check
        sim = float(cosine_similarity(
            task_b["_centroid"].reshape(1, -1),
            task_a["_centroid"].reshape(1, -1)
        )[0][0])
        if sim >= PREREQ_SIM_THRESHOLD:
            prereqs.add(task_a["task_id"])

    # Hard-wire rules
    name_b = task_b["_raw_name"]
    for kw_a, kw_b in HARD_WIRE_RULES:
        if kw_b in name_b:
            for task_a in tasks:
                if kw_a in task_a["_raw_name"] and \
                   PHASE_ORDER.get(task_a["phase"], 1) < phase_b:
                    prereqs.add(task_a["task_id"])

    task_b["prereq_ids"] = sorted(prereqs)

total_edges = sum(len(t["prereq_ids"]) for t in tasks)
print(f"✅ Inferred {total_edges} prerequisite edges across {len(tasks)} tasks.")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 8: Topological-sort sanity check
#
# The app runs Kahn's topological sort on the task graph.
# We verify there are no cycles here before writing to Delta.
# ─────────────────────────────────────────────────────────────

from collections import deque

def has_cycle(tasks: list[dict]) -> bool:
    id_to_task = {t["task_id"]: t for t in tasks}
    in_degree  = {t["task_id"]: 0 for t in tasks}
    adj        = defaultdict(list)

    for t in tasks:
        for p in t["prereq_ids"]:
            if p in id_to_task:
                adj[p].append(t["task_id"])
                in_degree[t["task_id"]] += 1

    queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for neighbour in adj[node]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    return visited != len(tasks)


if has_cycle(tasks):
    # Fallback: strip all prereq_ids and log a warning.
    # This is extremely unlikely given phase-gated inference above,
    # but better to write clean data than crash.
    logger.warning("⚠️  Cycle detected — clearing all prereq_ids as fallback.")
    for t in tasks:
        t["prereq_ids"] = []
else:
    print("✅ No cycles detected — DAG is valid.")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 9: Strip internal fields & build final task list
# ─────────────────────────────────────────────────────────────

FINAL_FIELDS = [
    "task_id", "task_name", "description", "authority",
    "portal_url", "prereq_ids", "sector_filter", "size_filter",
    "phase", "est_days",
]

final_tasks = []
for t in tasks:
    row = {k: t[k] for k in FINAL_FIELDS}

    # Truncate description to avoid Delta varchar limits
    row["description"] = row["description"][:800]

    # Ensure sector_filter / size_filter are clean lists
    row["sector_filter"] = [s.lower() for s in row["sector_filter"]]
    row["size_filter"]   = [s.lower() for s in row["size_filter"]]

    final_tasks.append(row)

print(f"✅ Final task list: {len(final_tasks)} tasks ready for Delta.")

# Quick stats
from collections import Counter
phase_dist = Counter(t["phase"] for t in final_tasks)
print(f"   Phase distribution: {dict(phase_dist)}")
sector_all  = sum(1 for t in final_tasks if t["sector_filter"] == ["all"])
print(f"   Tasks applicable to ALL sectors: {sector_all}")
print(f"   Tasks with sector-specific filter: {len(final_tasks) - sector_all}")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 10: Write to Delta table
#
# We REPLACE (overwrite) the existing task_graph table so the
# app always sees a consistent, freshly-generated task set.
# The schema is identical to what Notebook 02 wrote — the app
# code (db.py / app.py) doesn't need any changes.
# ─────────────────────────────────────────────────────────────

from pyspark.sql import Row
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, ArrayType
)

SCHEMA = StructType([
    StructField("task_id",       StringType(),               False),
    StructField("task_name",     StringType(),               True),
    StructField("description",   StringType(),               True),
    StructField("authority",     StringType(),               True),
    StructField("portal_url",    StringType(),               True),
    StructField("prereq_ids",    ArrayType(StringType()),    True),
    StructField("sector_filter", ArrayType(StringType()),    True),
    StructField("size_filter",   ArrayType(StringType()),    True),
    StructField("phase",         StringType(),               True),
    StructField("est_days",      IntegerType(),              True),
])

rows = [
    Row(
        task_id       = t["task_id"],
        task_name     = t["task_name"],
        description   = t["description"],
        authority     = t["authority"],
        portal_url    = t["portal_url"],
        prereq_ids    = t["prereq_ids"],
        sector_filter = t["sector_filter"],
        size_filter   = t["size_filter"],
        phase         = t["phase"],
        est_days      = int(t["est_days"]),
    )
    for t in final_tasks
]

df = spark.createDataFrame(rows, schema=SCHEMA)

(
    df.write
      .format("delta")
      .mode("overwrite")
      .option("overwriteSchema", "true")
      .saveAsTable(DELTA_TABLE)
)

print(f"✅ Written {len(final_tasks)} tasks to {DELTA_TABLE}.")
spark.sql(f"SELECT phase, COUNT(*) AS cnt FROM {DELTA_TABLE} GROUP BY phase ORDER BY cnt DESC").show()

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 11: Spot-check — show a sample of generated tasks
# ─────────────────────────────────────────────────────────────

spark.sql(f"""
    SELECT task_id, task_name, phase, sector_filter, prereq_ids, est_days
    FROM {DELTA_TABLE}
    ORDER BY phase, task_id
    LIMIT 25
""").show(truncate=50)

# COMMAND ----------

# COMMAND ----------
 
# ─────────────────────────────────────────────────────────────
# CELL 12: Export a lightweight JSON snapshot (optional)
#
# Useful for offline inspection or loading directly into the
# app as a fallback when Delta is unavailable.
#
# PATH NOTE — Unity Catalog Volumes on Databricks Free Edition:
#   CORRECT  →  /Volumes/<catalog>/<schema>/<volume>/file.json
#   WRONG    →  /dbfs/Volumes/...   (/dbfs prefix is for DBFS
#               legacy paths only — it does NOT work for UC Volumes
#               and raises OSError: [Errno 5] Input/output error)
#
# We try the direct Volume path first, then fall back to writing
# a Delta table (always available) so Cell 12 never blocks the run.
# ─────────────────────────────────────────────────────────────
 
import tempfile
 
# Direct UC Volume path — works on DBR 13+ with Unity Catalog
snapshot_path = f"{ARTIFACT_VOLUME}/task_graph_snapshot.json"
 
def _save_snapshot_to_volume(path: str, data: list) -> bool:
    """
    Write JSON directly to a Unity Catalog Volume path.
    Returns True on success, False on any I/O error.
 
    UC Volumes are accessible as a regular POSIX path
    (/Volumes/catalog/schema/volume/...) from notebook code
    running on DBR 13.3+ — no /dbfs prefix, no dbutils needed.
    """
    try:
        # os.makedirs() does NOT work on UC Volume paths on the free tier.
        # Volumes are not a real POSIX filesystem — subdirectories must
        # already exist (created via UI/CLI). We skip makedirs entirely
        # and let open() raise cleanly so the Delta fallback takes over.
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except OSError as e:
        logger.warning(f"Direct Volume write failed ({e}). Using Delta fallback.")
        return False
 
 
def _save_snapshot_via_spark(data: list) -> str:
    """
    Fallback: serialise the snapshot as a single-row Delta table
    (startup_hackathon.legal_data.task_graph_snapshot).
    The db.py fallback reader can query this table instead of
    reading a JSON file.  Always succeeds if Delta is available.
    """
    snapshot_table = "startup_hackathon.legal_data.task_graph_snapshot"
    payload = json.dumps(data, ensure_ascii=False)
    spark.sql(f"DROP TABLE IF EXISTS {snapshot_table}")
    spark.createDataFrame(
        [{"payload": payload}]
    ).write.format("delta").saveAsTable(snapshot_table)
    return snapshot_table
 
 
# ── Try direct Volume write ───────────────────────────────────
saved_to_volume = _save_snapshot_to_volume(snapshot_path, final_tasks)
 
if saved_to_volume:
    print(f"✅ Snapshot saved to Volume: {snapshot_path}")
    print("   db.py fallback will read this file directly.")
else:
    # ── Fallback: write to Delta table ────────────────────────
    snap_table = _save_snapshot_via_spark(final_tasks)
    print(f"✅ Snapshot saved to Delta table: {snap_table}")
    print("   Update SNAPSHOT_PATH in db.py to read from Delta instead:")
    print("   See db_patch_task_graph_fallback.py  →  _from_snapshot_delta()")
 
print(f"\n   Total tasks in snapshot: {len(final_tasks)}")
 
# ─────────────────────────────────────────────────────────────
# DONE
#
# Next steps:
#   1. Run Notebook 03 to process PDFs (unchanged)
#   2. Run Notebook 04 to rebuild the FAISS index (unchanged)
#   3. Restart the Streamlit app — it will pick up the new tasks
#      automatically on the next "Generate My Checklist" click.
#
# To regenerate tasks after adding new JSON data:
#   - Drop the new file(s) into the Volumes path above
#   - Re-run this notebook
#   - No changes needed in app.py, db.py, or graph.py
# ─────────────────────────────────────────────────────────────
 
print("\n🎉  Notebook 02b complete!  task_graph is now driven by your data.")
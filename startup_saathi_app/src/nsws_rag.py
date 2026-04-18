"""
nsws_rag.py — NSWS License Embedding Search & Checklist Generation
StartupSaathi | Bharat Bricks Hacks 2026

Companion module to rag.py.  This module handles the NEW embedding-driven
license search against:
  (a) Central DPIIT licenses — indexed per sector
  (b) State-specific licenses — one FAISS index per state

Search flow when a user submits their startup details:
  1. load_nsws_indices()     → loads all sector + state FAISS indices once
  2. search_nsws_all()       → encodes the query, searches sector index
                               (+ optionally "General"), and state index
  3. enrich_from_delta()     → fetches price, authority, details from
                               nsws_dpiit_licenses / nsws_state_licenses
                               Delta tables
  4. format_nsws_checklist() → converts results into the checklist format
                               already consumed by app.py

Databricks Free Edition constraints respected:
  - No GPU / CUDA — CPU only.
  - No Databricks Vector Search (paid).
  - faiss-cpu + sentence-transformers + numpy only.
  - All FAISS indices are tiny (<400 KB each) — no memory issues.
  - Uses @st.cache_resource for one-time loading across Streamlit reruns.

INCLUDE_GENERAL_IN_SECTOR_SEARCH:
  Toggle this boolean to control whether "General" sector licenses are
  mixed in when a user chooses a specific sector.
    True  → broader checklist, more coverage (recommended for MVPs)
    False → tighter results, less noise (useful for niche verticals)
"""

from __future__ import annotations

import os
import pickle
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════
# CONFIGURATION  (adjust paths / flags here — nowhere else needed)
# ════════════════════════════════════════════════════════════════

ARTIFACT_VOLUME  = "/Volumes/startup_hackathon/legal_data/model_artifacts"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM        = 384

# ── Search toggle ─────────────────────────────────────────────
# Flip to False to restrict DPIIT results to the chosen sector only.
INCLUDE_GENERAL_IN_SECTOR_SEARCH: bool = True

# ── Supported sectors (must match the sector field in the JSON) ──
KNOWN_DPIIT_SECTORS: list[str] = [
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

# ── Supported states ──────────────────────────────────────────
SUPPORTED_STATES: dict[str, str] = {
    "andhra_pradesh": "Andhra Pradesh",
    "delhi"         : "Delhi",
    "gujarat"       : "Gujarat",
    "madhya_pradesh": "Madhya Pradesh",
    "maharashtra"   : "Maharashtra",
}

# ── Local /tmp fallback (for Databricks Apps runtime) ────────
_TMP_DIR = "/tmp/nsws_artifacts"

# Delta table names (used for enrichment after vector search)
DPIIT_TABLE = "startup_hackathon.legal_data.nsws_dpiit_licenses"
STATE_TABLE = "startup_hackathon.legal_data.nsws_state_licenses"


# ════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ════════════════════════════════════════════════════════════════

def _sector_to_key(sector: str) -> str:
    """Convert sector display name to the filesystem-safe key used by Notebook 04b."""
    return sector.lower().replace("/", "-").replace(" ", "_")


def _resolve_artifact_path(filename: str) -> str:
    """
    Return a readable local path for a Volume artifact.

    Strategy:
      1. Try /Volumes/... directly (works in cluster notebooks).
      2. Fall back to /tmp cache (required in Databricks Apps runtime,
         which cannot resolve /Volumes/ via Python os.path).
         Downloads via the Databricks SDK Files API.
    """
    vol_path = f"{ARTIFACT_VOLUME}/{filename}"
    # Also validate size > 0 to guard against corrupt/empty Volume files.
    if os.path.exists(vol_path) and os.path.getsize(vol_path) > 0:
        return vol_path

    # Fallback: check /tmp cache.
    # IMPORTANT: only trust the cached file when it is non-empty.
    # A previous failed download leaves an empty file behind; returning
    # that path causes faiss.read_index to throw "0 != 1 (File exists)".
    os.makedirs(_TMP_DIR, exist_ok=True)
    tmp_path = f"{_TMP_DIR}/{filename}"
    if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
        return tmp_path

    # Attempt download via Databricks SDK Files API.
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        logger.info(f"Downloading {filename} via SDK → {tmp_path}")
        response = w.files.download(vol_path)
        # Use .read() on the BinaryIO object.
        # Iterating over a BinaryIO with `for chunk in response.contents`
        # reads line-by-line (splits on \n), which corrupts binary FAISS
        # index files and leaves a 0-byte file on disk.
        data = response.contents.read()
        with open(tmp_path, "wb") as f:
            f.write(data)
        if os.path.getsize(tmp_path) == 0:
            os.remove(tmp_path)
            raise IOError(
                f"Downloaded file '{filename}' is 0 bytes — "
                f"check that Notebook 04b ran successfully and the Volume path is correct."
            )
        return tmp_path
    except Exception as e:
        # Clean up any empty file so the next attempt triggers a fresh download.
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) == 0:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise FileNotFoundError(
            f"Cannot find artifact '{filename}' at '{vol_path}' or in /tmp cache. "
            f"Run Notebook 04b to generate all FAISS artifacts first. "
            f"SDK download error: {e}"
        )


# ════════════════════════════════════════════════════════════════
# INDEX LOADING  (call once; use @st.cache_resource in app.py)
# ════════════════════════════════════════════════════════════════

def load_embed_model():
    """
    Load the MiniLM sentence-transformer model.

    In app.py wrap this with @st.cache_resource so it is loaded
    only once per Streamlit server process:

        @st.cache_resource
        def _cached_model():
            return nsws_rag.load_embed_model()
    """
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    assert model.get_sentence_embedding_dimension() == EMBED_DIM
    logger.info("Embedding model loaded.")
    return model


def load_nsws_indices() -> tuple[dict, dict]:
    """
    Load all NSWS FAISS indices from the Unity Catalog Volume.

    Returns
    -------
    dpiit_indices : dict[sector_name, {"index": faiss.Index, "metadata": list}]
    state_indices : dict[state_key,   {"index": faiss.Index, "metadata": list}]

    In app.py wrap this with @st.cache_resource:

        @st.cache_resource
        def _cached_indices():
            return nsws_rag.load_nsws_indices()
    """
    import faiss

    dpiit_indices: dict[str, dict] = {}
    for sector in KNOWN_DPIIT_SECTORS:
        key        = _sector_to_key(sector)
        index_file = f"dpiit_faiss_{key}.bin"
        meta_file  = f"dpiit_meta_{key}.pkl"
        try:
            index_path = _resolve_artifact_path(index_file)
            meta_path  = _resolve_artifact_path(meta_file)
            index      = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            dpiit_indices[sector] = {"index": index, "metadata": meta}
            logger.info(f"Loaded DPIIT index [{sector}]: {index.ntotal} vectors")
        except FileNotFoundError as e:
            logger.warning(f"DPIIT index missing for sector '{sector}': {e}")

    state_indices: dict[str, dict] = {}
    for state_key in SUPPORTED_STATES:
        index_file = f"state_faiss_{state_key}.bin"
        meta_file  = f"state_meta_{state_key}.pkl"
        try:
            index_path = _resolve_artifact_path(index_file)
            meta_path  = _resolve_artifact_path(meta_file)
            index      = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            state_indices[state_key] = {"index": index, "metadata": meta}
            logger.info(f"Loaded state index [{state_key}]: {index.ntotal} vectors")
        except FileNotFoundError as e:
            logger.warning(f"State index missing for '{state_key}': {e}")

    loaded_dpiit = len(dpiit_indices)
    loaded_state = len(state_indices)
    logger.info(
        f"NSWS indices loaded: {loaded_dpiit}/{len(KNOWN_DPIIT_SECTORS)} sectors, "
        f"{loaded_state}/{len(SUPPORTED_STATES)} states."
    )
    return dpiit_indices, state_indices


# ════════════════════════════════════════════════════════════════
# DELTA ENRICHMENT  (joins vector search hits back to full details)
# ════════════════════════════════════════════════════════════════

def _enrich_dpiit_from_delta(license_ids: list[int]) -> dict[int, dict]:
    """
    Fetch full DPIIT license details from the nsws_dpiit_licenses Delta table.

    Returns dict keyed by licenseId with all stored fields:
      licenseName, sector, paymentType, price, details,
      ministryName, departmentName, referenceId, urlTitle,
      approvalType, availableOnNsws
    """
    if not license_ids:
        return {}
    try:
        from pyspark.sql import SparkSession
        spark  = SparkSession.builder.getOrCreate()
        id_csv = ", ".join(str(i) for i in license_ids)
        rows   = spark.sql(f"""
            SELECT
                licenseId, licenseName, sector, paymentType, price,
                details, ministryName, departmentName, referenceId,
                urlTitle, approvalType, availableOnNsws
            FROM {DPIIT_TABLE}
            WHERE licenseId IN ({id_csv})
        """).collect()
        return {r["licenseId"]: r.asDict() for r in rows}
    except Exception as e:
        logger.error(f"Delta enrichment (DPIIT) failed: {e}")
        return {}


def _enrich_state_from_delta(license_ids: list[int], state_key: str) -> dict[int, dict]:
    """
    Fetch full state license details from the nsws_state_licenses Delta table.

    Returns dict keyed by licenseId with all stored fields:
      licenseName, state_key, stateName, paymentType, price, details,
      authorityName, referenceId, urlTitle, approvalType, availableOnNsws
    """
    if not license_ids:
        return {}
    try:
        from pyspark.sql import SparkSession
        spark  = SparkSession.builder.getOrCreate()
        id_csv = ", ".join(str(i) for i in license_ids)
        rows   = spark.sql(f"""
            SELECT
                licenseId, licenseName, state_key, stateName,
                paymentType, price, details, authorityName,
                referenceId, urlTitle, approvalType, availableOnNsws
            FROM {STATE_TABLE}
            WHERE state_key = '{state_key}'
              AND licenseId IN ({id_csv})
        """).collect()
        return {r["licenseId"]: r.asDict() for r in rows}
    except Exception as e:
        logger.error(f"Delta enrichment (state={state_key}) failed: {e}")
        return {}


# ════════════════════════════════════════════════════════════════
# SEARCH FUNCTIONS
# ════════════════════════════════════════════════════════════════

def _encode(query: str, embed_model) -> np.ndarray:
    """Encode a query string to a normalised float32 vector (shape 1×384)."""
    vec = embed_model.encode(
        [query],
        convert_to_numpy     = True,
        normalize_embeddings = True,
    ).astype(np.float32)
    return vec  # (1, EMBED_DIM)


def search_nsws_dpiit(
    query: str,
    embed_model,
    sector: str,
    dpiit_indices: dict,
    top_k: int = 10,
    include_general: bool = INCLUDE_GENERAL_IN_SECTOR_SEARCH,
) -> list[dict]:
    """
    Search DPIIT central licenses for a given sector.

    Parameters
    ----------
    query           : Free-text startup description entered by the user.
    embed_model     : Loaded SentenceTransformer instance (from load_embed_model()).
    sector          : Must be one of KNOWN_DPIIT_SECTORS.
    dpiit_indices   : Dict returned by load_nsws_indices()[0].
    top_k           : Max results to return (after deduplication).
    include_general : If True, also search the "General" sector index
                      and merge results.  Controlled by
                      INCLUDE_GENERAL_IN_SECTOR_SEARCH.

    Returns
    -------
    List of dicts sorted by relevance score (desc).  Each dict contains:
      licenseId, licenseName, sector, paymentType, price (float),
      details (str), ministryName, departmentName, referenceId,
      urlTitle, approvalType, availableOnNsws (bool), score (float)
    """
    if sector not in dpiit_indices:
        logger.warning(
            f"Sector '{sector}' not found in loaded indices. "
            f"Available: {list(dpiit_indices.keys())}"
        )
        return []

    q_vec = _encode(query, embed_model)
    hits: dict[int, tuple[float, dict]] = {}  # licenseId → (best_score, meta)

    def _search_sector(sec_name: str):
        entry = dpiit_indices.get(sec_name)
        if entry is None:
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
            if lid not in hits or score > hits[lid][0]:
                hits[lid] = (float(score), meta)

    # Primary sector search
    _search_sector(sector)

    # Optionally blend in General results
    if include_general and sector != "General":
        _search_sector("General")

    if not hits:
        return []

    # Enrich from Delta
    enriched = _enrich_dpiit_from_delta(list(hits.keys()))

    results = []
    for lid, (score, meta) in hits.items():
        row = enriched.get(lid, {})
        results.append({
            "licenseId"      : lid,
            "licenseName"    : row.get("licenseName")    or meta["licenseName"],
            "sector"         : row.get("sector")         or meta.get("sector", ""),
            "paymentType"    : row.get("paymentType")    or "FREE",
            "price"          : float(row.get("price") or 0.0),
            "details"        : row.get("details")        or "",
            "ministryName"   : row.get("ministryName")   or "",
            "departmentName" : row.get("departmentName") or "",
            "referenceId"    : row.get("referenceId")    or "",
            "urlTitle"       : row.get("urlTitle")       or "",
            "approvalType"   : row.get("approvalType")   or "",
            "availableOnNsws": bool(row.get("availableOnNsws", False)),
            "score"          : round(score, 4),
            "source"         : "dpiit",
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def search_nsws_state(
    query: str,
    embed_model,
    state_key: str,
    state_indices: dict,
    top_k: int = 10,
) -> list[dict]:
    """
    Search state-specific licenses.

    Parameters
    ----------
    query         : Free-text startup description entered by the user.
    embed_model   : Loaded SentenceTransformer instance.
    state_key     : Must be one of SUPPORTED_STATES keys
                    (e.g. "gujarat", "maharashtra").
    state_indices : Dict returned by load_nsws_indices()[1].
    top_k         : Max results to return.

    Returns
    -------
    List of dicts sorted by relevance score (desc).  Each dict contains:
      licenseId, licenseName, state_key, stateName, paymentType,
      price (float), details (str), authorityName, referenceId,
      urlTitle, approvalType, availableOnNsws (bool), score (float)
    """
    if state_key not in state_indices:
        raise ValueError(
            f"State '{state_key}' not loaded.  "
            f"Valid options: {list(SUPPORTED_STATES.keys())}"
        )

    q_vec = _encode(query, embed_model)
    entry = state_indices[state_key]
    idx   = entry["index"]
    metas = entry["metadata"]
    k     = min(top_k, idx.ntotal)
    if k == 0:
        return []

    scores, positions = idx.search(q_vec, k)

    hits = [
        (float(scores[0][i]), metas[positions[0][i]])
        for i in range(k)
        if positions[0][i] >= 0
    ]
    if not hits:
        return []

    hit_ids  = [h[1]["licenseId"] for h in hits]
    enriched = _enrich_state_from_delta(hit_ids, state_key)

    results = []
    for score, meta in hits:
        lid = meta["licenseId"]
        row = enriched.get(lid, {})
        results.append({
            "licenseId"      : lid,
            "licenseName"    : row.get("licenseName")    or meta["licenseName"],
            "state_key"      : state_key,
            "stateName"      : row.get("stateName")      or SUPPORTED_STATES[state_key],
            "paymentType"    : row.get("paymentType")    or "FREE",
            "price"          : float(row.get("price") or 0.0),
            "details"        : row.get("details")        or "",
            "authorityName"  : row.get("authorityName")  or "",
            "referenceId"    : row.get("referenceId")    or "",
            "urlTitle"       : row.get("urlTitle")       or "",
            "approvalType"   : row.get("approvalType")   or "",
            "availableOnNsws": bool(row.get("availableOnNsws", False)),
            "score"          : round(score, 4),
            "source"         : "state",
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def search_nsws_all(
    query: str,
    embed_model,
    sector: str,
    state_key: str,
    dpiit_indices: dict,
    state_indices: dict,
    top_k: int = 10,
    include_general: bool = INCLUDE_GENERAL_IN_SECTOR_SEARCH,
) -> dict:
    """
    Combined NSWS search: DPIIT (sector-filtered) + state-specific.

    This is the main entry point called from app.py when the user
    submits their startup details.

    Parameters
    ----------
    query           : Free-text description (e.g. "food delivery cloud kitchen").
    embed_model     : Loaded SentenceTransformer.
    sector          : User-selected sector (one of KNOWN_DPIIT_SECTORS).
    state_key       : User-selected state (one of SUPPORTED_STATES keys).
    dpiit_indices   : From load_nsws_indices()[0].
    state_indices   : From load_nsws_indices()[1].
    top_k           : Results per category.
    include_general : Blend in General sector — defaults to
                      INCLUDE_GENERAL_IN_SECTOR_SEARCH module constant.

    Returns
    -------
    {
        "dpiit" : list[dict],   # Central licenses (DPIIT), sector-filtered
        "state" : list[dict],   # State-specific licenses
    }
    """
    dpiit_results = search_nsws_dpiit(
        query           = query,
        embed_model     = embed_model,
        sector          = sector,
        dpiit_indices   = dpiit_indices,
        top_k           = top_k,
        include_general = include_general,
    )
    state_results = search_nsws_state(
        query         = query,
        embed_model   = embed_model,
        state_key     = state_key,
        state_indices = state_indices,
        top_k         = top_k,
    )
    return {"dpiit": dpiit_results, "state": state_results}


# ════════════════════════════════════════════════════════════════
# CHECKLIST FORMATTER
# ════════════════════════════════════════════════════════════════

def format_nsws_checklist(search_results: dict) -> list[dict]:
    """
    Convert the raw search results dict (from search_nsws_all) into
    the flat checklist format already consumed by app.py.

    Each checklist item dict has:
      task_id        : "DPIIT-<licenseId>" or "STATE-<licenseId>"
      task_name      : licenseName
      description    : details text
      authority      : ministryName+departmentName (DPIIT) or authorityName (state)
      portal_url     : "https://www.nsws.gov.in/licenses/{urlTitle}"
                       (empty string if urlTitle is blank)
      fee_display    : "Free" / "₹{price}" / "Paid (fee not listed)"
      approval_type  : approvalType
      available_online: availableOnNsws
      source         : "dpiit" or "state"
      state_name     : stateName (only for state results, else "")
      sector         : sector (only for DPIIT results, else "")
      score          : relevance score (float)

    DPIIT results come first, ordered by score; state results follow.
    """
    checklist = []

    for item in search_results.get("dpiit", []):
        price = item.get("price", 0.0)
        if price and price > 0:
            fee = f"₹{price:.0f}"
        elif item.get("paymentType", "FREE").upper() == "FREE":
            fee = "Free"
        else:
            fee = "Paid (fee not listed)"

        authority = " — ".join(filter(None, [
            item.get("ministryName", ""),
            item.get("departmentName", ""),
        ]))
        url_title  = item.get("urlTitle", "")
        portal_url = f"https://www.nsws.gov.in/licenses/{url_title}" if url_title else ""

        checklist.append({
            "task_id"          : f"DPIIT-{item['licenseId']}",
            "task_name"        : item.get("licenseName", ""),
            "description"      : item.get("details", ""),
            "authority"        : authority,
            "portal_url"       : portal_url,
            "fee_display"      : fee,
            "approval_type"    : item.get("approvalType", ""),
            "available_online" : item.get("availableOnNsws", False),
            "source"           : "dpiit",
            "state_name"       : "",
            "sector"           : item.get("sector", ""),
            "score"            : item.get("score", 0.0),
        })

    for item in search_results.get("state", []):
        price = item.get("price", 0.0)
        if price and price > 0:
            fee = f"₹{price:.0f}"
        elif item.get("paymentType", "FREE").upper() == "FREE":
            fee = "Free"
        else:
            fee = "Paid (fee not listed)"

        url_title  = item.get("urlTitle", "")
        portal_url = f"https://www.nsws.gov.in/licenses/{url_title}" if url_title else ""

        checklist.append({
            "task_id"          : f"STATE-{item['licenseId']}",
            "task_name"        : item.get("licenseName", ""),
            "description"      : item.get("details", ""),
            "authority"        : item.get("authorityName", ""),
            "portal_url"       : portal_url,
            "fee_display"      : fee,
            "approval_type"    : item.get("approvalType", ""),
            "available_online" : item.get("availableOnNsws", False),
            "source"           : "state",
            "state_name"       : item.get("stateName", ""),
            "sector"           : "",
            "score"            : item.get("score", 0.0),
        })

    return checklist


# ════════════════════════════════════════════════════════════════
# APP.PY INTEGRATION SNIPPET
# ════════════════════════════════════════════════════════════════
#
# Paste this block into app.py to wire up the new search:
#
# ── app.py imports ─────────────────────────────────────────────
# import nsws_rag
#
# ── Cached resource loading (runs once per Streamlit server session) ──
# @st.cache_resource
# def _load_embed_model():
#     return nsws_rag.load_embed_model()
#
# @st.cache_resource
# def _load_nsws_indices():
#     return nsws_rag.load_nsws_indices()
#
# ── In the "Generate Checklist" button handler ────────────────
# embed_model               = _load_embed_model()
# dpiit_indices, state_idx  = _load_nsws_indices()
#
# # sector comes from st.selectbox("Sector", nsws_rag.KNOWN_DPIIT_SECTORS)
# # state_key from st.selectbox("State", list(nsws_rag.SUPPORTED_STATES.keys()),
# #                              format_func=lambda k: nsws_rag.SUPPORTED_STATES[k])
#
# raw_results = nsws_rag.search_nsws_all(
#     query         = startup_description,   # user input
#     embed_model   = embed_model,
#     sector        = selected_sector,
#     state_key     = selected_state_key,
#     dpiit_indices = dpiit_indices,
#     state_indices = state_idx,
#     top_k         = 12,
#     include_general = nsws_rag.INCLUDE_GENERAL_IN_SECTOR_SEARCH,
# )
# checklist = nsws_rag.format_nsws_checklist(raw_results)
#
# ── Render checklist ──────────────────────────────────────────
# for item in checklist:
#     st.markdown(f"**{item['task_name']}** ({item['fee_display']})")
#     if item["details"]:
#         st.caption(item["details"][:200])
#     label = f"{'🇮🇳 Central' if item['source'] == 'dpiit' else '🗺️ ' + item['state_name']}"
#     st.caption(f"{label} | {item['authority']}")
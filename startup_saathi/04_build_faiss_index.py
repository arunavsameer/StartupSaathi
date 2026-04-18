# Databricks notebook source
# ============================================================
# NOTEBOOK 04: Build FAISS Index
# StartupSaathi — Legal Navigator | Bharat Bricks Hacks 2026
#
# PURPOSE: Reads all embeddings from the legal_chunks Delta
#          table and builds a FAISS IndexFlatL2 saved to the
#          model_artifacts Volume. Also saves a parallel
#          metadata pickle file for chunk lookup.
#
# RUN: Every time after Notebook 03 runs (new PDFs added).
#      Takes ~5 minutes. Overwrites existing index files.
# ============================================================

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 1: Install FAISS (CPU version)
# ─────────────────────────────────────────────────────────────

%pip install faiss-cpu

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 2: Imports and path constants
# ─────────────────────────────────────────────────────────────

import os
import pickle
import numpy as np
import faiss

DELTA_TABLE      = "startup_hackathon.legal_data.legal_chunks"
ARTIFACT_VOLUME  = "/Volumes/startup_hackathon/legal_data/model_artifacts"
FAISS_INDEX_PATH = f"{ARTIFACT_VOLUME}/faiss_index.bin"
METADATA_PKL     = f"{ARTIFACT_VOLUME}/faiss_chunk_metadata.pkl"

# DBFS paths for actual file I/O
FAISS_INDEX_DBFS = f"/dbfs{FAISS_INDEX_PATH}"
METADATA_DBFS    = f"/dbfs{METADATA_PKL}"

EMBED_DIM = 384  # Must match the MiniLM model output dimension

print("✅ Imports complete.")
print(f"   Will write FAISS index to: {FAISS_INDEX_PATH}")
print(f"   Will write metadata to:    {METADATA_PKL}")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 3: Read all chunks from Delta table
# ─────────────────────────────────────────────────────────────

print(f"Reading from {DELTA_TABLE}...")

# Select only the columns needed — avoids loading chunk_text twice
raw_df = spark.sql(f"""
    SELECT
        chunk_id,
        source_file,
        page_number,
        chunk_text,
        embedding,
        sector_tag,
        phase_tag
    FROM {DELTA_TABLE}
    WHERE embedding IS NOT NULL
    ORDER BY ingested_at ASC, chunk_id ASC
""")

total_rows = raw_df.count()
print(f"✅ Found {total_rows} chunks with embeddings in Delta table.")

if total_rows == 0:
    raise RuntimeError(
        "❌ No chunks found in legal_chunks. "
        "Please run Notebook 03 (03_process_pdfs_and_chunks) first."
    )

# Collect to driver — all data needed in memory for FAISS
# At 384 floats × 4 bytes × 5000 chunks ≈ 7.7 MB — very safe
rows = raw_df.collect()
print(f"✅ Collected {len(rows)} rows to driver memory.")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 4: Build numpy embedding matrix
# ─────────────────────────────────────────────────────────────

print("Building embedding matrix...")

# Convert ARRAY<FLOAT> (Python list from Delta) → numpy float32 matrix
embedding_matrix = np.array(
    [list(row["embedding"]) for row in rows],
    dtype=np.float32
)

print(f"✅ Embedding matrix shape: {embedding_matrix.shape}")
print(f"   Expected: ({len(rows)}, {EMBED_DIM})")

# Sanity check: confirm dimensions match expected embedding size
if embedding_matrix.shape[1] != EMBED_DIM:
    raise ValueError(
        f"❌ Embedding dimension mismatch: "
        f"got {embedding_matrix.shape[1]}, expected {EMBED_DIM}. "
        f"Check that the same MiniLM model was used in Notebook 03."
    )

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 5: Build FAISS index
#
# IndexFlatL2: exact L2 search — no approximation.
# Fast enough for < 10,000 chunks (well within our range).
# Embeddings are already L2-normalized from Notebook 03
# (normalize_embeddings=True), so L2 distance ≡ cosine distance.
# ─────────────────────────────────────────────────────────────

print("Building FAISS IndexFlatL2...")

index = faiss.IndexFlatL2(EMBED_DIM)
index.add(embedding_matrix)

print(f"✅ FAISS index built.")
print(f"   Total vectors in index: {index.ntotal}")

# Quick smoke test — search for the first vector's nearest neighbors
test_query = embedding_matrix[0:1]
distances, indices = index.search(test_query, k=3)
print(f"   Smoke test (nearest to chunk 0): indices={indices[0]}, distances={distances[0]}")
# First result should be index 0 with distance ~0.0

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 6: Build parallel metadata list
#
# This is a list of dicts aligned with the FAISS index.
# faiss_metadata[i] corresponds to embedding_matrix[i].
# The app uses this to retrieve chunk text and source after search.
# ─────────────────────────────────────────────────────────────

print("Building metadata list...")

faiss_metadata = [
    {
        "chunk_id":    row["chunk_id"],
        "chunk_text":  row["chunk_text"],
        "source_file": row["source_file"],
        "page_number": row["page_number"],
        "sector_tag":  row["sector_tag"],
        "phase_tag":   row["phase_tag"],
    }
    for row in rows
]

print(f"✅ Metadata list built: {len(faiss_metadata)} entries")
print(f"   Sample entry[0]: source={faiss_metadata[0]['source_file']}, "
      f"page={faiss_metadata[0]['page_number']}, "
      f"text_preview={faiss_metadata[0]['chunk_text'][:80]}...")

# COMMAND ----------

# DBTITLE 1,Cell 8
# ─────────────────────────────────────────────────────────────
# CELL 7: Save FAISS index locally (NO VOLUME WRITE)
# ─────────────────────────────────────────────────────────────

import tempfile

# Local paths
FAISS_INDEX_LOCAL = "/tmp/faiss_index.bin"
METADATA_LOCAL    = "/tmp/faiss_metadata.pkl"

# ── Save FAISS index ──────────────────────────────────────────
print(f"Saving FAISS index locally: {FAISS_INDEX_LOCAL}")
faiss.write_index(index, FAISS_INDEX_LOCAL)

index_size_mb = os.path.getsize(FAISS_INDEX_LOCAL) / (1024 * 1024)
print(f"✅ FAISS index saved locally. Size: {index_size_mb:.2f} MB")


# ── Save metadata ─────────────────────────────────────────────
print(f"Saving metadata locally: {METADATA_LOCAL}")
with open(METADATA_LOCAL, "wb") as f:
    pickle.dump(faiss_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

meta_size_kb = os.path.getsize(METADATA_LOCAL) / 1024
print(f"✅ Metadata saved locally. Size: {meta_size_kb:.2f} KB")


print("\n⚠️ NOTE:")
print("Files are saved in /tmp (ephemeral).")
print("Download them or move them before cluster stops.")

# COMMAND ----------

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 7b: Copy FAISS artifacts from /tmp → Unity Catalog Volume
#
# Cell 7 saves to /tmp (local to the cluster driver — ephemeral).
# This cell copies those files to the persistent Unity Catalog Volume
# so the Databricks App can read them at runtime.
#
# Uses shutil.copy which works directly with /Volumes/ paths
# on Databricks Runtime 13.3+.
# ─────────────────────────────────────────────────────────────

import shutil, os

FAISS_INDEX_LOCAL = "/tmp/faiss_index.bin"
METADATA_LOCAL    = "/tmp/faiss_metadata.pkl"

# Destination on the Unity Catalog Volume (persistent)
ARTIFACT_VOLUME  = "/Volumes/startup_hackathon/legal_data/model_artifacts"
FAISS_INDEX_PATH = f"{ARTIFACT_VOLUME}/faiss_index.bin"
METADATA_PKL     = f"{ARTIFACT_VOLUME}/faiss_chunk_metadata.pkl"

# Verify source files exist before copying
assert os.path.exists(FAISS_INDEX_LOCAL), f"Source not found: {FAISS_INDEX_LOCAL} — run Cell 7 first"
assert os.path.exists(METADATA_LOCAL),    f"Source not found: {METADATA_LOCAL} — run Cell 7 first"

print(f"Copying FAISS index  → {FAISS_INDEX_PATH}")
shutil.copy(FAISS_INDEX_LOCAL, FAISS_INDEX_PATH)
print(f"  ✅ Done. Size: {os.path.getsize(FAISS_INDEX_PATH) / 1024 / 1024:.2f} MB")

print(f"Copying metadata pkl → {METADATA_PKL}")
shutil.copy(METADATA_LOCAL, METADATA_PKL)
print(f"  ✅ Done. Size: {os.path.getsize(METADATA_PKL) / 1024:.1f} KB")

print("\n✅ Both files are now on the Volume and ready for the app.")
print(f"   App will read from: {FAISS_INDEX_PATH}")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 8: Verify saved artifacts (LOCAL VERSION)
# ─────────────────────────────────────────────────────────────

print("Verifying saved artifacts (simulating app load)...")

# Local paths (same as previous cell)
FAISS_INDEX_LOCAL = "/tmp/faiss_index.bin"
METADATA_LOCAL    = "/tmp/faiss_metadata.pkl"

# ── Load index ────────────────────────────────────────────────
test_index = faiss.read_index(FAISS_INDEX_LOCAL)
print(f"✅ FAISS index reloaded. Vectors: {test_index.ntotal}")

# ── Load metadata ─────────────────────────────────────────────
with open(METADATA_LOCAL, "rb") as f:
    test_meta = pickle.load(f)

print(f"✅ Metadata reloaded. Entries: {len(test_meta)}")

# ── Validate alignment ────────────────────────────────────────
assert test_index.ntotal == len(test_meta), (
    f"❌ Mismatch: FAISS has {test_index.ntotal} vectors but metadata has {len(test_meta)} entries!"
)

print("✅ Index and metadata are aligned.")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL: Generate download links from /tmp
# ─────────────────────────────────────────────────────────────

from IPython.display import HTML

def download_link(path, filename):
    with open(path, "rb") as f:
        data = f.read()
    import base64
    b64 = base64.b64encode(data).decode()
    return f'<a download="{filename}" href="data:application/octet-stream;base64,{b64}">Download {filename}</a>'

index_link = download_link("/tmp/faiss_index.bin", "faiss_index.bin")
meta_link  = download_link("/tmp/faiss_metadata.pkl", "faiss_metadata.pkl")

HTML(f"""
<h3>Download your files:</h3>
<ul>
<li>{index_link}</li>
<li>{meta_link}</li>
</ul>
""")

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 9: Summary and source breakdown
# ─────────────────────────────────────────────────────────────

from collections import Counter

source_counts = Counter(m["source_file"] for m in faiss_metadata)
print("\n📊 Chunks by source document:")
for source, count in sorted(source_counts.items()):
    print(f"   {source:<55} {count:>5} chunks")

print(f"\n{'─'*60}")
print(f"  TOTAL VECTORS IN FAISS INDEX: {index.ntotal}")
print(f"{'─'*60}")
print(f"\n✅ FAISS index build complete!")
print(f"   Index path:    {FAISS_INDEX_PATH}")
print(f"   Metadata path: {METADATA_PKL}")
print(f"\n🚀 Next step: Deploy the Streamlit app (see SETUP_AND_RUN_GUIDE.md).")

# COMMAND ----------

!pip install sentence_transformers

# COMMAND ----------

# ─────────────────────────────────────────────────────────────
# CELL 10 (OPTIONAL): Full retrieval smoke test
# Tests the complete retrieval pipeline that the app will use.
# Run this to confirm the index returns sensible results.
# ─────────────────────────────────────────────────────────────

from sentence_transformers import SentenceTransformer

print("Loading MiniLM for retrieval smoke test...")
embed_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="cpu"
)

TEST_QUERIES = [
    "How do I register a company in India?",
    "What is GST registration threshold for startups?",
    "FSSAI license requirements for food business",
    "Kya main apne startup ko DPIIT mein register kar sakta hoon?",  # Hindi
]

for query in TEST_QUERIES:
    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = test_index.search(q_emb.astype(np.float32), k=3)
    print(f"\nQuery: '{query}'")
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        m = test_meta[idx]
        print(f"  [{rank}] {m['source_file']} (p.{m['page_number']}) | dist={dist:.4f}")
        print(f"      {m['chunk_text'][:120]}...")

print("\n✅ Retrieval smoke test complete.")
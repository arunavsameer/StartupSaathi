# 🚀 StartupSaathi — AI-Powered Startup Legal Navigator

> **BharatBricks Hacks 2026** · Track: **Swatantra (Open / Any Indic AI Use Case)**
> Built on Databricks Free Edition · IIT Indore · April 17–18, 2026

**Team StartupSaathi:** Arunav Sameer · Srinidhi Sai · Arihant Jain · Tanishq Godha

---

## 📌 What It Does

StartupSaathi is a multilingual AI assistant that guides Indian startup founders through every compliance step required to incorporate and operate a business in India — in their own language. It combines a dependency-aware task graph (topological sort via Kahn's algorithm), a FAISS-backed RAG pipeline over curated legal documents, and Sarvam AI's multilingual LLM to deliver grounded, step-by-step legal guidance across 22 Indian languages. Founders get a personalised compliance checklist ordered by legal prerequisites, an AI chat assistant that cites sources, and a live scraper for government startup opportunities.

---

### Screenshots

<img width="1920" height="1080" alt="Screenshot 2026-04-18 151254" src="https://github.com/user-attachments/assets/6e8f79b0-9877-45e4-97b0-9e30ac95f18e" />

*Figure 1: Dashboard overview*


<img width="1920" height="1080" alt="Screenshot 2026-04-18 151516" src="https://github.com/user-attachments/assets/79f226c0-cd83-40ea-8865-d8753c3d7a54" />

*Figure 2: Customized Checklist* 


<img width="1920" height="1080" alt="Screenshot 2026-04-18 151833" src="https://github.com/user-attachments/assets/8f2354df-1ac5-4856-9aa7-76591b4ce3d6" />

*Figure 3: Legal Q&A* 


<img width="1920" height="1080" alt="Screenshot 2026-04-18 151845" src="https://github.com/user-attachments/assets/1b382086-3385-489c-a587-0073a9142ce2" />

*Figure 4: Response to Question* 


<img width="1920" height="1080" alt="Screenshot 2026-04-18 151900" src="https://github.com/user-attachments/assets/41e48e0e-6a13-4b12-a646-55afda36a31b" />

*Figure 5: Applicable government schemes* 


<img width="1920" height="1080" alt="Screenshot 2026-04-18 151942" src="https://github.com/user-attachments/assets/d0f27a50-7371-4816-8460-5c8b800b1930" />

*Figure 6: Incubation Opportunities* 


<img width="1920" height="1080" alt="Screenshot 2026-04-18 151954" src="https://github.com/user-attachments/assets/dc05c5b0-20c3-4af2-9e5c-50bdb2cb2388" />

*Figure 7: Incubation Center Information* 


<img width="1920" height="1080" alt="Screenshot 2026-04-18 152536" src="https://github.com/user-attachments/assets/280a45d2-6bdb-402c-ac49-d4f57c4cbddc" />

*Figure 8: Example of the AI answering a query in different languages* 

<!-- PLACEHOLDER: Add screenshots to docs/images/ and update paths above -->

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATABRICKS FREE EDITION                             │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    PIPELINE (Data Engineering)                        │  │
│  │                                                                      │  │
│  │  Raw Docs ──► [01_setup_catalog_and_tables.sql]                      │  │
│  │  (PDFs/JSON)      CREATE Delta tables in Unity Catalog               │  │
│  │       │                                                              │  │
│  │       ▼                                                              │  │
│  │  [02_populate_task_graph.py]                                         │  │
│  │   17 compliance tasks + DAG written → task_graph (Gold)             │  │
│  │       │                                                              │  │
│  │       ▼                                                              │  │
│  │  [02b_generate_task_graph_from_data.py]                             │  │
│  │   Auto-generate extended tasks from raw data sources                │  │
│  │       │                                                              │  │
│  │       ▼                                                              │  │
│  │  [03_process_pdfs_and_chunks.py]                                     │  │
│  │   Extract text → chunk (~300 words) → embed (MiniLM-384)            │  │
│  │   Write to → legal_chunks (Silver) in Delta Lake                    │  │
│  │       │                                                              │  │
│  │       ▼                                                              │  │
│  │  [04_build_faiss_index.py]                                           │  │
│  │   Read embeddings from Delta → build FAISS IndexFlatIP              │  │
│  │   Save → /Volumes/startup_hackathon/legal_data/model_artifacts/     │  │
│  │       │                                                              │  │
│  │       ▼                                                              │  │
│  │  [04b_build_nsws_license_embeddings.py]                             │  │
│  │   NSWS license catalogue → separate FAISS index                     │  │
│  │       │                                                              │  │
│  │       ▼                                                              │  │
│  │  [05_opportunity_scraper.py]  ◄── Databricks Job (scheduled)        │  │
│  │   Scrape gov startup schemes → opportunities (Gold) Delta table      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │              UNITY CATALOG  (startup_hackathon.legal_data)           │  │
│  │                                                                      │  │
│  │  Delta Tables:                                                       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  ┌──────────┐ │  │
│  │  │ legal_chunks│  │ task_graph  │  │user_profiles │  │query_logs│ │  │
│  │  │  (Silver)   │  │  (Gold)     │  │  (Gold)      │  │  (Gold)  │ │  │
│  │  └─────────────┘  └─────────────┘  └──────────────┘  └──────────┘ │  │
│  │                                                                      │  │
│  │  UC Volume:  /Volumes/.../model_artifacts/                           │  │
│  │  ┌───────────────────┐   ┌──────────────────────────┐               │  │
│  │  │  faiss_index.bin  │   │ faiss_chunk_metadata.pkl │               │  │
│  │  └───────────────────┘   └──────────────────────────┘               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    DATABRICKS APP (Streamlit)                         │  │
│  │                                                                      │  │
│  │  app.py ──► src/                                                     │  │
│  │    ├─ rag.py          FAISS load (Volume → /tmp fallback) +          │  │
│  │    │                  query expansion + BM25 rerank + prompt build   │  │
│  │    ├─ models.py       Sarvam AI API (sarvam-30b) → Databricks FMAPI │  │
│  │    │                  MiniLM embedder (CPU, ~120 MB)                 │  │
│  │    ├─ graph.py        Kahn's topological sort → compliance order     │  │
│  │    ├─ translate.py    IndicTrans2 / Sarvam translate wrappers        │  │
│  │    ├─ db.py           Delta Lake read/write via Databricks SDK       │  │
│  │    ├─ nsws_rag.py     NSWS licence RAG (separate FAISS index)        │  │
│  │    ├─ constants.py    Sectors, sizes, phases, config                 │  │
│  │    └─ ui_helpers.py   Streamlit component helpers                    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    EXTERNAL AI SERVICES                               │  │
│  │  Sarvam AI API ──► sarvam-30b (primary LLM, Indian-origin)          │  │
│  │                    sarvam-m   (fallback)                             │  │
│  │  Databricks FMAPI ► llama-3.3-70b / dbrx (free, no key needed)     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

```
User Query (any Indian language)
        │
        ▼
  Language Detection
        │
        ▼
  Query Expansion + Variant Generation (pure Python, zero LLM cost)
        │
        ├──► FAISS Semantic Search (MiniLM 384-dim)
        │         │
        │         ▼
        │    BM25-lite Re-ranking
        │         │
        └──► Top-K Chunks Retrieved from legal_chunks (Delta)
                  │
                  ▼
           Grounded Prompt Assembly (query-type-aware template)
                  │
                  ▼
           Sarvam AI sarvam-30b  →  (fallback) Databricks FMAPI
                  │
                  ▼
           Response (translated back to user language if non-English)
                  │
                  ▼
           Logged to query_logs Delta table
```

---

## 🗂️ Repository Structure

```
startup-saathi/
│
├── README.md                          ← You are here
├── docs/
│   └── images/
│       ├── checklist.png              ← PLACEHOLDER: add screenshot
│       ├── chat.png                   ← PLACEHOLDER: add screenshot
│       └── opportunities.png          ← PLACEHOLDER: add screenshot
│
├── startup_saathi/                    ← Databricks Pipeline Notebooks
│   ├── 01_setup_catalog_and_tables.sql      Step 1: Create Delta tables
│   ├── 02_populate_task_graph.py            Step 2: Write 17 compliance tasks
│   ├── 02b_generate_task_graph_from_data.py Step 2b: Extended task generation
│   ├── 03_process_pdfs_and_chunks.py        Step 3: PDF → chunks → embeddings
│   ├── 04_build_faiss_index.py              Step 4: Build FAISS index
│   ├── 04b_build_nsws_license_embeddings.py Step 4b: NSWS licence index
│   └── 05_opportunity_scraper.py            Step 5: Scrape gov schemes (Job)
│
└── startup_saathi_app/                ← Databricks App (Streamlit)
    ├── app.py                               Main Streamlit entrypoint
    ├── app.yaml                             Databricks App manifest
    ├── requirements.txt                     Python dependencies
    └── src/
        ├── __init__.py
        ├── constants.py                     Sectors, sizes, phases, config
        ├── db.py                            Delta Lake read/write (SDK)
        ├── graph.py                         Kahn's topological sort + DAG
        ├── models.py                        LLM + embedding model loaders
        ├── nsws_rag.py                      NSWS licence RAG pipeline
        ├── rag.py                           Core RAG: FAISS + retrieval + prompts
        ├── translate.py                     Multilingual translation wrappers
        └── ui_helpers.py                    Streamlit UI component helpers
```

---

## ⚙️ How to Run

### Prerequisites

- Databricks Free Edition workspace — sign up at [bharatbricks.org/free-edition](https://bharatbricks.org/free-edition)
- Sarvam AI API key — get one at [dashboard.sarvam.ai](https://dashboard.sarvam.ai) *(free tier available)*
- Python 3.10+ on your local machine (only needed to import notebooks)

---

### Step 1 — Clone this Repository

```bash
git clone https://github.com/<your-org>/startup-saathi.git
cd startup-saathi
```

---

### Step 2 — Import Notebooks into Databricks

1. Open your Databricks Free Edition workspace
2. Go to **Workspace → Import**
3. Import all files from `startup_saathi/` as individual notebooks
4. Recommended import method: drag-and-drop `.py` / `.sql` files into a `startup_saathi` folder in your workspace

---

### Step 3 — Run the Pipeline (in order)

Open each notebook and run it top-to-bottom using a **Serverless** or **Single-node** cluster.

```
01_setup_catalog_and_tables.sql       ← Run once. Creates Unity Catalog tables.
02_populate_task_graph.py             ← Run once. Writes 17 compliance tasks.
02b_generate_task_graph_from_data.py  ← Run once. Adds extended tasks.
03_process_pdfs_and_chunks.py         ← Run once. Chunks PDFs + writes embeddings.
04_build_faiss_index.py               ← Run once. Builds FAISS index on Volume.
04b_build_nsws_license_embeddings.py  ← Run once. Builds NSWS FAISS index.
05_opportunity_scraper.py             ← Run manually or schedule as a Databricks Job.
```

> ⚠️ **Wait for each notebook to complete** before running the next. Steps 3 and 4 depend on the Delta tables created in steps 1 and 2.

---

### Step 4 — Configure Secrets

Store your Sarvam API key as a Databricks secret (recommended) or use the App environment variable:

**Option A — Databricks Secret (recommended):**
```bash
# Using Databricks CLI
databricks secrets create-scope startup-saathi
databricks secrets put --scope startup-saathi --key sarvam-api-key
```
Then reference it in your notebook/app: `dbutils.secrets.get("startup-saathi", "sarvam-api-key")`

**Option B — App environment variable:**
Set `SARVAM_API_KEY` directly in `app.yaml` under the `env` section (already templated):
```yaml
env:
  - name: SARVAM_API_KEY
    value: "your-key-here"
```

---

### Step 5 — Deploy the Databricks App

1. In your workspace, go to **Apps → Create App**
2. Upload the entire `startup_saathi_app/` directory, or link to this Git repo
3. Databricks will detect `app.yaml` and run:
   ```
   streamlit run app.py --server.headless=true --server.port=8000 --server.address=0.0.0.0
   ```
4. Click **Deploy** — your app will be live at a `*.databricksapps.com` URL within minutes

**Or deploy via CLI:**
```bash
databricks apps deploy startup-saathi --source-code-path ./startup_saathi_app
```

---

### Step 6 — Schedule the Opportunity Scraper (Databricks Job)

The opportunity scraper (`05_opportunity_scraper.py`) fetches live government startup schemes and should run on a schedule:

1. Go to **Workflows → Jobs → Create Job**
2. Select notebook: `startup_saathi/05_opportunity_scraper.py`
3. Set cluster: **Serverless**
4. Set schedule: **Daily at 6:00 AM IST** (or as needed)
5. Add widget parameter: `MAX_RECORDS = 100`
6. Click **Create**

The job writes results to `startup_hackathon.legal_data.opportunities` (Delta), which the app reads live.

---

### Demo Steps (What to Click)

Once the app is deployed:

1. **Open the app URL** in your browser
2. **Select your startup profile**: choose sector (e.g., Food Tech), company size (Micro), and state
3. **View your compliance checklist**: tasks appear in dependency order (Kahn's topological sort)
4. **Mark tasks as complete**: the checklist updates and unlocks next steps
5. **Ask a legal question** in the chat (any language): e.g., *"GST registration ke liye kya documents chahiye?"*
6. **Switch to Opportunities tab**: see live government schemes for your profile
7. **Try NSWS licence lookup**: search for licences required for your business activity

---

## 🛠️ Technologies Used

### Databricks Platform
| Component | Usage |
|---|---|
| **Delta Lake** | Silver layer (`legal_chunks`) and Gold layer (`task_graph`, `user_profiles`, `query_logs`, `opportunities`) with Change Data Feed |
| **Unity Catalog** | Catalog `startup_hackathon`, schema `legal_data`, Volume for FAISS artifacts |
| **Apache Spark / PySpark** | PDF processing, chunking, embedding writes, opportunity scraping |
| **Databricks Apps** | Streamlit app hosting (CPU-only, no GPU required) |
| **Databricks Jobs** | Scheduled opportunity scraper (Notebook Job) |
| **Databricks SDK** | App-side Delta table reads/writes and Volume file access |
| **Databricks Foundation Models API** | Free LLM fallback (Llama 3.3 70B, DBRX) |
| **MLflow** | Experiment tracking for embedding runs |

### AI / ML Models
| Model | Type | Role |
|---|---|---|
| **Sarvam AI sarvam-30b** | LLM (Indian-origin) | Primary answer generation |
| **Sarvam AI sarvam-m** | LLM (Indian-origin) | LLM fallback |
| **paraphrase-multilingual-MiniLM-L12-v2** | Embedding (384-dim) | Semantic search, CPU-only, ~120 MB |
| **FAISS IndexFlatIP** | Vector index | Fast similarity search over legal chunks |
| **IndicTrans2** | Translation | Hindi ↔ English and 22 Indian languages |

### Open-Source Libraries
```
streamlit>=1.32.0
sentence-transformers>=2.7.0
faiss-cpu>=1.8.0
networkx>=3.2.0
langdetect>=1.0.9
databricks-sdk>=0.25.0
openai>=1.30.0          # Sarvam AI is OpenAI-compatible
pydeck>=0.8.0
requests>=2.31.0
```

---

## 📊 Datasets Used

| Dataset | Source | Usage |
|---|---|---|
| BNS 2023 Full Text | [Kaggle](https://www.kaggle.com/datasets/nandr39/bharatiya-nyaya-sanhita-dataset-bns) | Legal clause retrieval |
| India Government Schemes (MyScheme) | [data.gov.in](https://data.gov.in) | Scheme eligibility matching |
| NSWS Licence Catalogue | National Single Window System | Licence lookup by business activity |
| Startup India / DPIIT Guidelines | Public PDFs | Compliance guidance |
| gov.in Opportunity Feed | Live scrape via `05_opportunity_scraper.py` | Real-time scheme updates |

---

## 🏆 Judging Criteria Alignment

| Criteria | How We Address It |
|---|---|
| **Databricks Usage (30%)** | Delta Lake (4 tables + CDF), Unity Catalog Volumes, Spark for ETL, Databricks Apps for serving, Databricks Jobs for scheduled scraping, Databricks FMAPI as LLM fallback |
| **Accuracy & Effectiveness (25%)** | Multi-query RAG with BM25 reranking, query-type-aware prompts, extractive fallback, source citations in every answer |
| **Innovation (25%)** | Kahn's topological sort for dependency-aware compliance checklists; CPU-only FAISS serving via Volume → /tmp fallback; Indian-language-first UX using Sarvam |
| **Presentation & Demo (20%)** | Live working app at Databricks Apps URL; reproducible from this README in under 30 minutes |

---


## 👥 Team StartupSaathi

| Name | Role |
|---|---|
| Arunav Sameer | Pipeline & Data Engineering |
| Srinidhi Sai | RAG & AI / LLM Integration |
| Arihant Jain | App & UI Development |
| Tanishq Godha | Task Graph & Backend Logic |

---

*Built with ❤️ at IIT Indore · BharatBricks Hacks 2026*

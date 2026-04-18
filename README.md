# 🚀 SancharSathi — AI Legal & Startup Compliance Assistant

**BharatBricks 2026 Hackathon Submission**

---

## 🧠 What it does

SancharSathi is an AI-powered assistant that helps Indian startups understand legal requirements and discover opportunities using a Databricks-based RAG pipeline with automated data ingestion.

---

## 🎥 Demo

* 📹 Demo Video: *(Add link here)*
* 🌐 Live App: *(Add link here)*

---

## 📸 Screenshots

*(Add screenshots of your UI, results, dashboard here)*

---

# 🏗️ Architecture (Databricks-Centric)

## 🔴 Full System Architecture

```mermaid
flowchart TD

    A[Legal PDFs / Govt Websites]
    B[Startup Opportunity Websites]

    A --> C[Databricks Pipeline: Data Ingestion]
    B --> D[Databricks Job: Daily Scraper]

    C --> E[Databricks SQL / Unity Catalog]
    D --> F[Databricks SQL / Unity Catalog]

    E --> G[Embedding + Chunking Pipeline]
    G --> H[Vector Index (FAISS)]

    I[User Query] --> J[Databricks App / Streamlit]
    J --> K[RAG Pipeline]

    K --> H
    H --> K

    K --> L[LLM Inference]
    L --> M[Response to User]
```

---

## 🧩 Databricks Components Breakdown

### 🔹 Pipeline Scripts (`startup_saathi/`)

* `01_setup_catalog_and_tables.sql` → Sets up the database catalog and tables
* `02_populate_task_graph.py` & `02b_generate_task_graph_from_data.py` → Generates task graph data
* `03_process_pdfs_and_chunks.py` → Chunking and preprocessing PDFs
* `04_build_faiss_index.py` & `04b_build_nsws_license_embeddings.py` → Embedding generation and FAISS index build
* `05_opportunity_scraper.py` → Scrapes latest startup opportunities

### 🔹 Application Layer (`startup_saathi_app/`)

* `app.py` → Streamlit / Databricks App entry point
* `src/` → RAG and DB logic (`rag.py`, `nsws_rag.py`, `db.py`)
* Handles user queries and displays results.

---

# ⚙️ How to Run (Exact Steps)

## 1. Setup Database

Run the SQL script to initialize tables:
```bash
# Execute in Databricks SQL / Notebook
01_setup_catalog_and_tables.sql
```

## 2. Run Data Pipeline

Execute the python scripts sequentially in your Databricks environment:
```bash
python startup_saathi/02_populate_task_graph.py
python startup_saathi/03_process_pdfs_and_chunks.py
python startup_saathi/04_build_faiss_index.py
python startup_saathi/04b_build_nsws_license_embeddings.py
```

## 3. Setup Daily Job (Startup Opportunities)

Go to **Workflows → Jobs → Create Job** in Databricks to schedule the scraper:
* **Script path**: `startup_saathi/05_opportunity_scraper.py`
* **Schedule**: Every 24 hours

## 4. Run Application

Navigate to the app directory and install dependencies:
```bash
cd startup_saathi_app
pip install -r requirements.txt
```

Run the Streamlit app:
```bash
streamlit run app.py
```
*(Or deploy it directly using Databricks Apps via `app.yaml`)*

---

# 🧪 Demo Steps (For Judges)

1. Open the app
2. Enter queries:

**Legal:**
```
How do I register a startup in India?
```
```
What are tax exemptions under Startup India?
```

**Opportunities:**
```
Show me current startup schemes
```

---

## ✅ Expected Output

* Relevant legal context retrieved
* AI-generated explanation
* Latest startup opportunities

---

# 📁 Repository Structure

```text
.
├── startup_saathi/                  # Databricks Data Pipeline & Jobs
│   ├── 01_setup_catalog_and_tables.sql
│   ├── 02_populate_task_graph.py
│   ├── 02b_generate_task_graph_from_data.py
│   ├── 03_process_pdfs_and_chunks.py
│   ├── 04_build_faiss_index.py
│   ├── 04b_build_nsws_license_embeddings.py
│   ├── 05_opportunity_scraper.py
│   └── manifest.mf
└── startup_saathi_app/              # Databricks App (Streamlit)
    ├── app.py
    ├── app.yaml
    ├── manifest.mf
    ├── requirements.txt
    └── src/
        ├── __init__.py
        ├── constants.py
        ├── db.py
        ├── graph.py
        ├── models.py
        ├── nsws_rag.py
        ├── rag.py
        ├── translate.py
        └── ui_helpers.py
```

---

# 📊 Features

* AI-powered legal assistant
* RAG-based document retrieval
* Automated startup opportunity updates
* Databricks-native pipeline
* Build with Databricks Apps & Asset Bundles

---

# ⚡ Key Design Decisions

* Built for Databricks architecture
* RAG instead of fine-tuning (faster, scalable)
* Automated ingestion via Databricks Jobs
* Modular pipeline for easy extension

---

# 🌍 Impact

* Makes legal compliance accessible to startups
* Reduces dependency on legal experts
* Provides real-time opportunity discovery

---

# 🔮 Future Work

* Multilingual support (Indic languages)
* Personalized recommendations
* Dashboard analytics
* More datasets integration

---

# 📜 License

MIT License

---

# 👥 Team SancharSathi

* Arunav Sameer
* Srinidhi Sai
* Arihant Jain
* Tanishq Godha

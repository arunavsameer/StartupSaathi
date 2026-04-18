-- Databricks notebook source
-- DBTITLE 1,Header
-- ============================================================
-- NOTEBOOK 01: Setup Catalog & Delta Tables
-- StartupSaathi — Legal Navigator | Bharat Bricks Hacks 2026
-- RUN THIS ONCE after creating Unity Catalog objects in GUI.
-- ============================================================

-- COMMAND ----------

-- DBTITLE 1,Set catalog and schema
-- Switch to the correct catalog and schema
USE CATALOG startup_hackathon;
USE SCHEMA legal_data;

-- COMMAND ----------

-- DBTITLE 1,Create legal_chunks table
-- ─────────────────────────────────────────────────────────────
-- TABLE 1: legal_chunks (Silver Layer)
-- Stores all text chunks extracted from PDFs, CSVs, JSON.
-- Change Data Feed MUST be enabled for incremental reads.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS startup_hackathon.legal_data.legal_chunks (
    chunk_id STRING NOT NULL COMMENT 'UUID — primary key for each text chunk',
    source_file STRING NOT NULL COMMENT 'Filename of the origin document (PDF / CSV / JSON)',
    page_number INT COMMENT 'PDF page number, CSV row number, or JSON index',
    chunk_text STRING COMMENT 'Raw chunk text, ~300 words per chunk',
    embedding ARRAY<FLOAT> COMMENT '384-dim MiniLM embedding vector',
    sector_tag STRING COMMENT 'Comma-separated sector tags: all / food_tech / manufacturing / etc.',
    phase_tag STRING
      COMMENT 'Comma-separated phase tags: incorporation / post-incorporation / operations / all',
    ingested_at TIMESTAMP COMMENT 'Timestamp of ingestion run'
  )
  TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true');

-- COMMAND ----------

-- DBTITLE 1,Create task_graph table
-- ─────────────────────────────────────────────────────────────
-- TABLE 2: task_graph (Gold Layer)
-- Stores the compliance task DAG. App reads this dynamically.
-- Never hardcode task data in the app — always read from here.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS startup_hackathon.legal_data.task_graph (
  task_id       STRING         NOT NULL COMMENT 'e.g., T001',
  task_name     STRING                  COMMENT 'Human-readable task name',
  description   STRING                  COMMENT '1–3 sentence explanation',
  authority     STRING                  COMMENT 'Governing body: MCA / GSTN / FSSAI / etc.',
  portal_url    STRING                  COMMENT 'Official government portal link',
  prereq_ids    ARRAY<STRING>           COMMENT 'task_ids that must be completed first',
  sector_filter ARRAY<STRING>           COMMENT 'Applicable sectors: ["all"] or ["food_tech","manufacturing"]',
  size_filter   ARRAY<STRING>           COMMENT 'Applicable sizes: ["all"] or ["micro","small","medium"]',
  phase         STRING                  COMMENT 'incorporation | post-incorporation | operations',
  est_days      INT                     COMMENT 'Estimated days to complete this task'
);

-- COMMAND ----------

-- DBTITLE 1,Create user_profiles table
-- ─────────────────────────────────────────────────────────────
-- TABLE 3: user_profiles (Gold Layer)
-- Tracks per-session compliance progress. Written by the app.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS startup_hackathon.legal_data.user_profiles (
  session_id       STRING         NOT NULL COMMENT 'UUID generated per browser session',
  sector           STRING                  COMMENT 'Selected sector by user',
  size             STRING                  COMMENT 'Selected company size',
  location         STRING                  COMMENT 'State of incorporation',
  completed_tasks  ARRAY<STRING>           COMMENT 'List of checked-off task_ids',
  last_updated     TIMESTAMP               COMMENT 'Last write timestamp'
);

-- COMMAND ----------

-- ─────────────────────────────────────────────────────────────
-- TABLE 4: query_logs (Gold Layer)
-- Logs every Q&A interaction for the Databricks SQL Dashboard.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS startup_hackathon.legal_data.query_logs (
  log_id            STRING         NOT NULL COMMENT 'UUID for this log entry',
  session_id        STRING                  COMMENT 'Session that made the query',
  query_text        STRING                  COMMENT 'The user question',
  retrieved_chunks  ARRAY<STRING>           COMMENT 'chunk_ids retrieved by FAISS',
  answer_text       STRING                  COMMENT 'LLM-generated answer',
  sector_context    STRING                  COMMENT 'Sector active when query was made',
  response_time_ms  INT                     COMMENT 'End-to-end latency in milliseconds',
  logged_at         TIMESTAMP               COMMENT 'Timestamp of the log entry'
);

-- COMMAND ----------

-- DBTITLE 1,Create query_logs table
-- ─────────────────────────────────────────────────────────────
-- TABLE 4: query_logs (Gold Layer)
-- Logs every Q&A interaction for the Databricks SQL Dashboard.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS startup_hackathon.legal_data.query_logs (
  log_id            STRING         NOT NULL COMMENT 'UUID for this log entry',
  session_id        STRING                  COMMENT 'Session that made the query',
  query_text        STRING                  COMMENT 'The user question',
  retrieved_chunks  ARRAY<STRING>           COMMENT 'chunk_ids retrieved by FAISS',
  answer_text       STRING                  COMMENT 'LLM-generated answer',
  sector_context    STRING                  COMMENT 'Sector active when query was made',
  response_time_ms  INT                     COMMENT 'End-to-end latency in milliseconds',
  logged_at         TIMESTAMP               COMMENT 'Timestamp of the log entry'
);

-- COMMAND ----------

-- DBTITLE 1,Verify tables created
-- ─────────────────────────────────────────────────────────────
-- VERIFICATION: Confirm all four tables exist
-- ─────────────────────────────────────────────────────────────
SHOW TABLES IN startup_hackathon.legal_data;

-- COMMAND ----------

-- DBTITLE 1,Describe legal_chunks schema
-- Optional: Describe the most complex table to confirm schema
DESCRIBE TABLE startup_hackathon.legal_data.legal_chunks;
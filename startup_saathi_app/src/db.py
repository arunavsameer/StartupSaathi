"""
db.py — Delta Table Read/Write Helpers
StartupSaathi | Bharat Bricks Hacks 2026

Uses the Databricks SDK's StatementExecutionAPI to query Delta tables via a
SQL warehouse. This is the correct approach for Databricks Apps on Free Edition:

  - No Spark Connect / cluster ID required
  - SQL warehouse is always available in Free Edition (starter warehouse)
  - DATABRICKS_HOST and DATABRICKS_TOKEN are auto-injected by Databricks Apps
  - WorkspaceClient() picks them up with zero configuration

Do NOT use SparkSession / DatabricksSession here — those require a running
cluster or serverless compute attached to the app.
"""

import json
import os
import uuid
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

CATALOG = "startup_hackathon"
SCHEMA  = "legal_data"
FQN     = f"{CATALOG}.{SCHEMA}"

# ── Snapshot paths (written by Notebook 02b Cell 12) ──────────
# Used as a fallback when the SQL warehouse is cold-starting.
# The Volume path is the happy-path; the Delta table is the
# fallback-of-fallback if the Volume write failed in 02b.
ARTIFACT_VOLUME = "/Volumes/startup_hackathon/legal_data/model_artifacts"
SNAPSHOT_PATH   = f"{ARTIFACT_VOLUME}/task_graph_snapshot.json"
SNAPSHOT_TABLE  = f"{CATALOG}.{SCHEMA}.task_graph_snapshot"

_workspace_client = None
_warehouse_id     = None


def _get_client():
    global _workspace_client
    if _workspace_client is None:
        from databricks.sdk import WorkspaceClient
        _workspace_client = WorkspaceClient()
        logger.info("Databricks WorkspaceClient initialised.")
    return _workspace_client


def _get_warehouse_id() -> str:
    global _warehouse_id
    if _warehouse_id is not None:
        return _warehouse_id
    w = _get_client()
    warehouses = list(w.warehouses.list())
    if not warehouses:
        raise RuntimeError(
            "No SQL warehouses found. Create one in Databricks SQL → Warehouses, "
            "then restart the app."
        )
    running = [wh for wh in warehouses if str(getattr(wh.state, "value", "")).upper() == "RUNNING"]
    chosen  = running[0] if running else warehouses[0]
    _warehouse_id = chosen.id
    logger.info(f"Using SQL warehouse: {chosen.name} (id={_warehouse_id})")
    return _warehouse_id


def _run_sql(statement: str, wait_timeout: str = "30s"):
    """Execute SQL via the warehouse. Returns (columns, rows)."""
    from databricks.sdk.service.sql import StatementState, Disposition, Format
    w     = _get_client()
    wh_id = _get_warehouse_id()
    resp  = w.statement_execution.execute_statement(
        warehouse_id=wh_id,
        statement=statement,
        disposition=Disposition.INLINE,
        format=Format.JSON_ARRAY,
        wait_timeout=wait_timeout,
    )
    if resp.status.state != StatementState.SUCCEEDED:
        err = resp.status.error.message if (resp.status and resp.status.error) else "unknown"
        raise RuntimeError(f"SQL failed [{resp.status.state}]: {err}")
    columns = []
    if resp.manifest and resp.manifest.schema and resp.manifest.schema.columns:
        columns = [col.name for col in resp.manifest.schema.columns]
    rows = resp.result.data_array if (resp.result and resp.result.data_array) else []
    return columns, rows


def _parse_array(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def _esc(s: str) -> str:
    return str(s).replace("'", "''")


# ── task_graph ────────────────────────────────────────────────

def get_task_graph(sector: str, size: str) -> list[dict]:
    """
    Return compliance tasks filtered by sector and size.

    Read strategy (free-tier safe):
      Level 1 — Delta table via SQL warehouse (primary, same as before).
      Level 2 — JSON snapshot at SNAPSHOT_PATH on the UC Volume.
      Level 3 — Delta snapshot table (single-row, payload column).

    Levels 2 and 3 are only reached if Level 1 fails or returns 0 rows,
    which happens when the SQL warehouse is still cold-starting (~30-60 s
    on free edition). The snapshot is written by Notebook 02b Cell 12.

    The returned list shape is identical regardless of which level served
    it, so graph.py and app.py need no changes.
    """

    def _parse_task_row(t: dict) -> dict:
        """Normalise array fields and est_days — shared by all levels."""
        for col in ("prereq_ids", "sector_filter", "size_filter"):
            t[col] = _parse_array(t.get(col))
        if t.get("est_days") is not None:
            try:
                t["est_days"] = int(t["est_days"])
            except (TypeError, ValueError):
                t["est_days"] = 0
        return t

    def _filter_tasks(all_tasks: list[dict]) -> list[dict]:
        """Filter a flat task list by sector and size in Python."""
        filtered = []
        for t in all_tasks:
            sf = [s.lower() for s in t.get("sector_filter", ["all"])]
            zf = [s.lower() for s in t.get("size_filter",   ["all"])]
            if ("all" in sf or sector.lower() in sf) and \
               ("all" in zf or size.lower()   in zf):
                filtered.append(_parse_task_row(t))
        return filtered

    # ── Level 1: Delta table ──────────────────────────────────
    try:
        sql = f"""
            SELECT task_id, task_name, description, authority, portal_url,
                   prereq_ids, sector_filter, size_filter, phase, est_days
            FROM {FQN}.task_graph
            WHERE (array_contains(sector_filter, 'all') OR array_contains(sector_filter, '{_esc(sector)}'))
              AND (array_contains(size_filter,   'all') OR array_contains(size_filter,   '{_esc(size)}'))
        """
        columns, rows = _run_sql(sql)
        tasks = [_parse_task_row(dict(zip(columns, row))) for row in rows]
        if tasks:
            logger.info(f"Delta: returned {len(tasks)} tasks for sector={sector}, size={size}.")
            return tasks
        logger.warning("Delta returned 0 tasks — trying snapshot fallback.")
    except Exception as exc:
        logger.warning(f"Delta read failed ({exc}) — trying snapshot fallback.")

    # ── Level 2: JSON file on UC Volume ──────────────────────
    try:
        if os.path.exists(SNAPSHOT_PATH):
            with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
                all_tasks = json.load(f)
            filtered = _filter_tasks(all_tasks)
            logger.info(f"JSON snapshot: returned {len(filtered)} tasks for sector={sector}, size={size}.")
            return filtered
    except Exception as exc:
        logger.warning(f"JSON snapshot read failed ({exc}) — trying Delta snapshot table.")

    # ── Level 3: Delta snapshot table ────────────────────────
    try:
        columns, rows = _run_sql(f"SELECT payload FROM {SNAPSHOT_TABLE} LIMIT 1")
        if rows:
            all_tasks = json.loads(rows[0][0])
            filtered = _filter_tasks(all_tasks)
            logger.info(f"Delta snapshot table: returned {len(filtered)} tasks for sector={sector}, size={size}.")
            return filtered
    except Exception as exc:
        logger.error(f"Delta snapshot table read failed ({exc}). All fallbacks exhausted.")

    return []


# ── user_profiles ─────────────────────────────────────────────

def upsert_user_profile(
    session_id: str,
    sector: str,
    size: str,
    location: str,
    completed_tasks: list[str],
) -> None:
    try:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        if completed_tasks:
            arr = "array(" + ", ".join(f"'{_esc(t)}'" for t in completed_tasks) + ")"
        else:
            arr = "array()"
        sql = f"""
            MERGE INTO {FQN}.user_profiles AS target
            USING (
                SELECT '{_esc(session_id)}' AS session_id,
                       '{_esc(sector)}'     AS sector,
                       '{_esc(size)}'       AS size,
                       '{_esc(location)}'   AS location,
                       {arr}               AS completed_tasks,
                       TIMESTAMP '{now}'   AS last_updated
            ) AS source
            ON target.session_id = source.session_id
            WHEN MATCHED     THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
        """
        _run_sql(sql, wait_timeout="20s")
    except Exception as e:
        logger.error(f"upsert_user_profile failed: {e}")


def get_user_profile(session_id: str) -> dict | None:
    try:
        columns, rows = _run_sql(f"""
            SELECT session_id, sector, size, location, completed_tasks
            FROM {FQN}.user_profiles
            WHERE session_id = '{_esc(session_id)}'
            LIMIT 1
        """)
        if not rows:
            return None
        r = dict(zip(columns, rows[0]))
        r["completed_tasks"] = _parse_array(r.get("completed_tasks"))
        return r
    except Exception as e:
        logger.error(f"get_user_profile failed: {e}")
        return None


# ── query_logs ────────────────────────────────────────────────

def log_query(
    session_id: str,
    query_text: str,
    retrieved_chunk_ids: list[str],
    answer_text: str,
    sector_context: str,
    response_time_ms: int,
) -> None:
    try:
        now    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        log_id = str(uuid.uuid4())
        if retrieved_chunk_ids:
            chunks = "array(" + ", ".join(f"'{_esc(c)}'" for c in retrieved_chunk_ids) + ")"
        else:
            chunks = "array()"
        sql = f"""
            INSERT INTO {FQN}.query_logs
            (log_id, session_id, query_text, retrieved_chunks,
             answer_text, sector_context, response_time_ms, logged_at)
            VALUES (
                '{log_id}', '{_esc(session_id)}', '{_esc(query_text[:2000])}',
                {chunks}, '{_esc(answer_text[:8000])}', '{_esc(sector_context)}',
                {int(response_time_ms)}, TIMESTAMP '{now}'
            )
        """
        _run_sql(sql, wait_timeout="20s")
    except Exception as e:
        logger.error(f"log_query failed: {e}")


# ── legal_chunks (schemes) ────────────────────────────────────

def get_schemes(limit: int = 15) -> list[dict]:
    """
    Return scheme chunks from legal_chunks where source is myscheme_startup.json.
    Uses the same SQL warehouse path as every other db function.
    """
    try:
        columns, rows = _run_sql(f"""
            SELECT chunk_text, source_file, sector_tag
            FROM {FQN}.legal_chunks
            WHERE source_file = 'myscheme_startup.json'
            ORDER BY page_number ASC
            LIMIT {int(limit)}
        """)
        return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        logger.error(f"get_schemes failed: {e}")
        return []


# ── startup_opportunities ─────────────────────────────────────
def get_opportunities(limit: int = 50, sector: str = "all", state: str = "") -> list[dict]:
    """
    Return startup incubation opportunities, filtered to match the user's
    startup profile (sector and state).

    Filtering logic (done in Python after fetch — sector is a free-text
    comma-separated string so SQL LIKE is unreliable for multi-value matching):
      - sector: "all" → show everything; otherwise keyword-match against
                the stored sector string using SECTOR_KEYWORDS map.
      - state:  empty → show everything; otherwise exact state match in SQL.

    Returns a list of dicts with keys:
        name, description, sector, link, discovered_at, state, city
    Returns [] on any error.
    """

    # ── Sector keyword map ────────────────────────────────────
    # Maps the app's internal sector key → keywords to look for in the
    # stored sector string (which comes straight from the API, e.g.
    # "Clean Tech, Enterprise Mobility, Technology Hardware").
    SECTOR_KEYWORDS: dict[str, list[str]] = {
        "tech":         ["technology", "software", "saas", "it", "digital",
                         "mobility", "enterprise", "ar vr", "augmented",
                         "virtual", "internet", "iot", "ai", "machine learning",
                         "data", "cyber", "telecom", "media"],
        "food_tech":    ["food", "agri", "agriculture", "fmcg", "nutrition",
                         "retail", "supply chain"],
        "manufacturing":["manufacturing", "hardware", "defence", "aerospace",
                         "automotive", "robotics", "3d print", "materials",
                         "textile", "chemical", "construction"],
    }

    try:
        # State filter goes into SQL (exact match) — keeps result set small
        state_clause = ""
        if state and state.strip():
            state_clause = f"AND LOWER(state) = LOWER('{_esc(state)}')"

        columns, rows = _run_sql(f"""
            SELECT name, description, sector, link, discovered_at, state, city
            FROM {FQN}.startup_opportunities
            WHERE 1=1 {state_clause}
            ORDER BY discovered_at DESC
            LIMIT {int(limit) * 5}
        """, wait_timeout="30s")
        # Fetch 5× limit so Python-side sector filtering still returns enough rows

        results = []
        keywords = SECTOR_KEYWORDS.get(sector, [])   # empty → no filter

        for row in rows:
            record = dict(zip(columns, row))

            # ── Sector filter (Python, keyword match) ─────────
            if keywords:
                stored_sector = (record.get("sector") or "").lower()
                if not any(kw in stored_sector for kw in keywords):
                    continue   # doesn't match user's sector

            # ── Normalise timestamp ───────────────────────────
            raw_ts = record.get("discovered_at")
            if raw_ts:
                try:
                    record["discovered_at"] = (
                        str(raw_ts).replace("T", " ").split(".")[0] + " UTC"
                    )
                except Exception:
                    pass

            results.append(record)
            if len(results) >= limit:
                break

        logger.info(
            f"get_opportunities: {len(results)} records "
            f"(sector={sector}, state={state or 'all'})."
        )
        return results

    except Exception as e:
        logger.error(f"get_opportunities failed: {e}")
        return []
#!/usr/bin/env python3
"""
Agent 2 (Claude) per riconciliazione MDR: valida i candidati RACI per ogni task e scrive
le decisioni in MdrReconciliationAgentDecisions.

Config: config.txt (MOTHERDUCK_*, ANTHROPIC_API_KEY, PROMPT_VERSION). Vedi config.example.txt.

Uso in tempo reale (default)
  Elabora i task pending con N worker paralleli (possibile rate limit 429).
  python 3.2_run_agent2_claude.py [--prompt-version v1] [--embedding-model text-embedding-3-small] [--limit N] [--workers 4] [--model claude-sonnet-4-6]

Uso con Batch API (niente rate limit, ~50% costo)
  1) Submit: invia i task pending in batch chunked automatici.
     python 3.2_run_agent2_claude.py --batch [--limit N] [--batch-max-bytes 240000000]
     Ogni chunk genera un batch id; la lista viene salvata in .agent2_last_batch_ids.json.
  2) Collect: quando i batch sono terminati, scarica i risultati e scrive in DB.
     python 3.2_run_agent2_claude.py --batch-collect
     Colleziona automaticamente tutti i batch dell'ultimo submit (anche se e' uno solo).

Stati Agent2Status (batch): pending | submitted_{batch_id} | in_batch_{batch_id} | done | error_{batch_id}
  (solo 'done' senza suffisso; gli altri stati batch hanno _batch_id per tracciare il batch.)
"""

import json
import argparse
import queue
import threading
import time
import random
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import anthropic
import httpx

# -----------------------------
# Config da file (stesso formato degli altri script)
# -----------------------------
CONFIG_PATH = Path(__file__).resolve().parent / "config.txt"
BATCH_IDS_FILE = Path(__file__).resolve().parent / ".agent2_last_batch_ids.json"
TEST_TASK_FILE = Path(__file__).resolve().parent / ".recon_test_tasks.json"
CLAUDE_BATCH_REQUEST_HARD_LIMIT_BYTES = 256_000_000
DEFAULT_BATCH_TARGET_BYTES = 240_000_000
DEFAULT_ADAPTIVE_BATCH_INITIAL_LIMIT = 7000
DEFAULT_ADAPTIVE_BATCH_MIN_LIMIT = 500
DEFAULT_ADAPTIVE_BATCH_MAX_LIMIT = 12000
DEFAULT_ADAPTIVE_BATCH_BACKOFF_FACTOR = 0.5
DEFAULT_ADAPTIVE_BATCH_GROWTH_FACTOR = 1.1


def load_config(path: Optional[Path] = None) -> Dict[str, str]:
    p = path or CONFIG_PATH
    config: Dict[str, str] = {}
    if not p.exists():
        raise FileNotFoundError(
            f"File di configurazione non trovato: {p}\n"
            "Crea config.txt (puoi copiare config.example.txt) e inserisci le chiavi richieste."
        )
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                config[key.strip()] = value.strip()
    return config


def get_config() -> Dict[str, str]:
    if not hasattr(get_config, "_cache"):
        get_config._cache = load_config()  # type: ignore[attr-defined]
    return get_config._cache  # type: ignore[attr-defined]


def _cfg(key: str, default: Optional[str] = None) -> str:
    return get_config().get(key, default or "").strip()


AGENT_NAME = "claude"
DEFAULT_MODEL = "claude-opus-4-6"

client = anthropic.Anthropic(api_key=_cfg("ANTHROPIC_API_KEY") or None)


def now_ts_naive_utc():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def norm(s):
    if s is None:
        return ""
    return " ".join(str(s).strip().split())


def connect_motherduck() -> duckdb.DuckDBPyConnection:
    token = _cfg("MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError("Manca MOTHERDUCK_TOKEN nel file di configurazione (config.txt)")
    dbname = _cfg("MOTHERDUCK_DB", "my_db")
    return duckdb.connect(f"md:{dbname}?token={token}")


AGENT_TOP_CANDIDATES_TABLE = "my_db.mdr_reconciliation.MdrReconciliationAgentTopCandidates"


def ensure_agent_eval_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
    CREATE TABLE IF NOT EXISTS my_db.mdr_reconciliation.MdrReconciliationAgentDecisions (
      TaskId              VARCHAR NOT NULL,
      AgentName           VARCHAR NOT NULL,
      AgentModel          VARCHAR,
      Document_title      VARCHAR NOT NULL,
      PromptVersion       VARCHAR NOT NULL,
      EmbeddingModel      VARCHAR NOT NULL,
      SelectedTitleKey    VARCHAR,
      SelectedRaciTitle   VARCHAR,
      DecisionType        VARCHAR NOT NULL,
      Confidence          DOUBLE,
      ReasoningSummary    VARCHAR NOT NULL,
      CreatedAt           TIMESTAMP NOT NULL,
      PRIMARY KEY (TaskId, AgentName)
    );
    """)


def load_or_create_test_tasks(tasks: List[Dict[str, Any]], test_count: int) -> List[Dict[str, Any]]:
    """
    Test mode: share a fixed set of TaskId across agents/judge.
    - If TEST_TASK_FILE exists: filter tasks by stored task_ids.
    - If it does not exist: randomly pick test_count TaskId from tasks, save them, and return only those.
    """
    path = TEST_TASK_FILE
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            ids = set(data.get("task_ids") or [])
        except Exception:
            return tasks
        if not ids:
            return tasks
        return [t for t in tasks if t.get("TaskId") in ids]

    if not tasks:
        return tasks

    shuffled = list(tasks)
    random.shuffle(shuffled)
    count = max(1, int(test_count or 1))
    selected = shuffled[:count]
    task_ids = [t.get("TaskId") for t in selected if t.get("TaskId")]
    if not task_ids:
        return tasks

    payload = {
        "task_ids": task_ids,
    }
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Test mode: created shared task list with {len(task_ids)} TaskId in {path.name}.")
    except Exception as e:
        print(f"Warning: could not write test task file {path}: {e}")
    ids_set = set(task_ids)
    return [t for t in tasks if t.get("TaskId") in ids_set]


def ensure_agent_top_candidates_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"""
    CREATE TABLE IF NOT EXISTS {AGENT_TOP_CANDIDATES_TABLE} (
      TaskId                    VARCHAR NOT NULL,
      AgentName                 VARCHAR NOT NULL,
      PromptVersion             VARCHAR NOT NULL,
      ModelName                 VARCHAR,
      CandidateRankWithinAgent  INTEGER NOT NULL,
      TitleKey                  VARCHAR NOT NULL,
      RaciTitle                 VARCHAR,
      CandidateConfidence       DOUBLE,
      WhyPlausible              VARCHAR,
      CreatedAt                 TIMESTAMP NOT NULL,
      PRIMARY KEY (TaskId, AgentName, CandidateRankWithinAgent)
    );
    """)


def fetch_pending_tasks(
    con: duckdb.DuckDBPyConnection,
    prompt_version: str,
    embedding_model: str,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    params: List[Any] = [prompt_version, embedding_model]
    sql = """
        SELECT
          TaskId,
          Document_title,
          PromptVersion,
          EmbeddingModel,
          CandidateCount,
          Agent1Status,
          Agent2Status,
          JudgeStatus,
          FinalStatus
        FROM my_db.mdr_reconciliation.MdrReconciliationTasks
        WHERE PromptVersion = ?
          AND EmbeddingModel = ?
          AND Agent2Status = 'pending'
        ORDER BY Document_title
    """
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    rows = con.execute(sql, params).fetchall()

    cols = [
        "TaskId", "Document_title", "PromptVersion", "EmbeddingModel",
        "CandidateCount", "Agent1Status", "Agent2Status",
        "JudgeStatus", "FinalStatus"
    ]
    return [dict(zip(cols, r)) for r in rows]


def claim_task_agent2(con: duckdb.DuckDBPyConnection, task_id: str) -> bool:
    ts = now_ts_naive_utc()
    con.execute("BEGIN;")
    try:
        rows = con.execute("""
            UPDATE my_db.mdr_reconciliation.MdrReconciliationTasks
            SET
              Agent2Status = 'running',
              FinalStatus = CASE
                WHEN FinalStatus = 'pending' THEN 'in_progress'
                ELSE FinalStatus
              END,
              UpdatedAt = ?
            WHERE TaskId = ?
              AND Agent2Status = 'pending'
            RETURNING TaskId
        """, [ts, task_id]).fetchall()
        con.execute("COMMIT;")
        return len(rows) == 1
    except Exception:
        con.execute("ROLLBACK;")
        raise


def fetch_candidates_for_task(
    con: duckdb.DuckDBPyConnection,
    document_title: str,
    prompt_version: str,
    embedding_model: str
) -> List[Dict[str, Any]]:
    rows = con.execute("""
        SELECT
          Rank,
          Similarity,
          TitleKey,
          RaciTitle,
          EffectiveDescription,
          DisciplineName,
          TypeName,
          CategoryDescription,
          ChapterName
        FROM my_db.mdr_reconciliation.v_MdrReconciliationAgentInput
        WHERE Document_title = ?
          AND PromptVersion = ?
          AND EmbeddingModel = ?
        ORDER BY Rank
    """, [document_title, prompt_version, embedding_model]).fetchall()

    cols = [
        "Rank", "Similarity", "TitleKey", "RaciTitle", "EffectiveDescription",
        "DisciplineName", "TypeName", "CategoryDescription", "ChapterName"
    ]
    return [dict(zip(cols, r)) for r in rows]


def fetch_mdr_context(
    con: duckdb.DuckDBPyConnection,
    document_title: str
) -> Dict[str, Any]:
    row = con.execute("""
        SELECT
          Document_title,
          Discipline_Normalized,
          Discipline_Status,
          Type_L1,
          Type_L1_Status
        FROM my_db.historical_mdr_normalization.v_MdrPreviousRecords_Normalized_All
        WHERE Document_title = ?
        LIMIT 1
    """, [document_title]).fetchone()

    if not row:
        return {
            "Document_title": document_title,
            "Discipline_Normalized": None,
            "Discipline_Status": None,
            "Type_L1": None,
            "Type_L1_Status": None,
        }

    cols = [
        "Document_title",
        "Discipline_Normalized",
        "Discipline_Status",
        "Type_L1",
        "Type_L1_Status",
    ]
    return dict(zip(cols, row))


# Expected JSON shape (for validation; Claude does not use schema in API)
EXPECTED_AGENT_OUTPUT = {
    "decision_type": "MATCH | NO_MATCH",
    "selected_titlekey": "string | null",
    "selected_raci_title": "string | null",
    "confidence": 0.0,
    "reasoning_summary": "string",
    "top_candidates": [{"rank": 1, "titlekey": "string", "raci_title": "string", "confidence": 0.0, "why_plausible": "string"}],
}

SYSTEM_PROMPT = """
You are a specialist agent for document title reconciliation in EPC project documentation.

You will receive:
- one historical MDR title with normalized metadata
- a list of 50 candidate standard RACI documents

Your task:
1. Determine whether a sufficiently credible semantic match exists.
2. Return MATCH or NO_MATCH.
3. Return an ordered shortlist (top_candidates) of up to 3 most plausible candidates.

Allowed outcomes:
- MATCH: exactly one candidate is clearly the best semantic match.
- NO_MATCH: no candidate is sufficiently credible.

Rules:
- MATCH only if one candidate is clearly the best. If several are plausible but none is clearly superior, return NO_MATCH.
- Similarity score and rank are retrieval hints only, not proof of equivalence.
- Prefer semantic equivalence over lexical overlap. Use discipline, type, category, chapter as strong evidence; strong metadata incompatibility is negative evidence.
- Do not invent information; only use provided candidates.
- top_candidates: maximum 3 distinct candidates, ordered by your preference (rank 1 = best). Ranks must be 1, 2, 3 with no gaps. No duplicates.
- If decision_type is MATCH: selected_titlekey and selected_raci_title must equal the rank-1 candidate in top_candidates (same titlekey and raci_title).
- If decision_type is NO_MATCH: selected_titlekey and selected_raci_title must be null. top_candidates may be empty or list up to 3 closest-but-insufficient candidates.
- confidence between 0 and 1. reasoning_summary and why_plausible must be brief and factual.

Output JSON only with: decision_type, selected_titlekey, selected_raci_title, confidence, reasoning_summary, top_candidates (array of {rank, titlekey, raci_title, confidence, why_plausible}).
"""


def build_user_prompt(mdr_ctx: Dict[str, Any], candidates: List[Dict[str, Any]]) -> str:

    blocks = []

    blocks.append("HISTORICAL MDR RECORD")
    blocks.append(f"Document_title: {norm(mdr_ctx.get('Document_title'))}")
    blocks.append(f"Discipline_Normalized: {norm(mdr_ctx.get('Discipline_Normalized'))}")
    blocks.append(f"Discipline_Status: {norm(mdr_ctx.get('Discipline_Status'))}")
    blocks.append(f"Type_L1: {norm(mdr_ctx.get('Type_L1'))}")
    blocks.append(f"Type_L1_Status: {norm(mdr_ctx.get('Type_L1_Status'))}")
    blocks.append("")

    blocks.append("EVALUATION GUIDANCE")
    blocks.append("- SimilarityScore is a retrieval hint only, not proof of equivalence")
    blocks.append("- Prefer semantic equivalence between MDR title and candidate meaning")
    blocks.append("- Use discipline, type, category, and chapter as supporting evidence")
    blocks.append("- Strong metadata incompatibility is a negative signal")
    blocks.append("- Choose MATCH only if one candidate is clearly stronger than the others")
    blocks.append("- Choose NO_MATCH if the best candidate is still weak, generic, or not clearly equivalent")
    blocks.append("")

    blocks.append("CANDIDATES")

    for c in candidates:
        blocks.append("----")
        blocks.append(f"Rank: {c['Rank']}")
        blocks.append(f"SimilarityScore: {float(c['Similarity']):.4f}")
        blocks.append(f"TitleKey: {norm(c['TitleKey'])}")
        blocks.append(f"RaciTitle: {norm(c['RaciTitle'])}")
        blocks.append(f"EffectiveDescription: {norm(c['EffectiveDescription'])}")
        blocks.append(f"DisciplineName: {norm(c['DisciplineName'])}")
        blocks.append(f"TypeName: {norm(c['TypeName'])}")
        blocks.append(f"CategoryDescription: {norm(c['CategoryDescription'])}")
        blocks.append(f"ChapterName: {norm(c['ChapterName'])}")

    blocks.append("")
    blocks.append("Return JSON only.")

    return "\n".join(blocks)


def extract_text_from_anthropic_message(message) -> str:
    parts = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts).strip()


def extract_json_payload(raw_text: str) -> str:
    text = raw_text.strip()

    # Caso 1: fenced JSON block
    if text.startswith("```"):
        lines = text.splitlines()

        # rimuove prima riga ``` o ```json
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        # rimuove ultima riga ``` se presente
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # Caso 2: testo extra prima/dopo il JSON
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return text


def _is_rate_limit_error(ex: BaseException) -> bool:
    msg = str(ex).lower()
    return "429" in msg or "rate_limit" in msg


def call_agent(model: str, mdr_ctx: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    user_prompt = build_user_prompt(mdr_ctx, candidates)

    max_retries = 5
    base_wait = 60  # seconds

    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=model,
                # token budget aumentato per permettere risposte JSON complete
                max_tokens=1200,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )
            break
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < max_retries - 1:
                wait = base_wait * (2 ** attempt)
                time.sleep(wait)
                continue
            raise

    raw_text = extract_text_from_anthropic_message(message)
    cleaned_json = extract_json_payload(raw_text)
    try:
        return json.loads(cleaned_json)
    except json.JSONDecodeError as e:
        # Log raw Anthropic output to help debug JSON formatting issues
        print("==== CLAUDE RAW TEXT BEGIN ====")
        print(raw_text)
        print("==== CLAUDE RAW TEXT END ====")
        print("==== CLAUDE CLEANED JSON BEGIN ====")
        print(cleaned_json)
        print("==== CLAUDE CLEANED JSON END ====")
        raise


def validate_agent_output(result: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    candidate_map = {norm(c["TitleKey"]): c for c in candidates}

    decision_type = result["decision_type"]
    selected_titlekey = result.get("selected_titlekey")
    selected_raci_title = result.get("selected_raci_title")
    confidence = float(result["confidence"])
    reasoning_summary = norm(result.get("reasoning_summary") or "")
    raw_top = result.get("top_candidates") or []

    if confidence < 0:
        confidence = 0.0
    if confidence > 1:
        confidence = 1.0

    # Validate top_candidates: max 3, distinct titlekeys, ranks 1..N no gaps
    top_candidates: List[Dict[str, Any]] = []
    seen_keys: set = set()
    for i, item in enumerate(raw_top[:3]):
        rank = int(item.get("rank", i + 1))
        titlekey = norm(item.get("titlekey") or "")
        if not titlekey or titlekey in seen_keys:
            raise ValueError("top_candidates must contain distinct titlekeys; rank must be 1..N without gaps")
        seen_keys.add(titlekey)
        if titlekey not in candidate_map:
            raise ValueError(f"top_candidates titlekey not in provided candidates: {titlekey}")
        if rank != i + 1:
            raise ValueError("top_candidates ranks must be 1, 2, 3 with no gaps")
        top_candidates.append({
            "CandidateRankWithinAgent": rank,
            "TitleKey": titlekey,
            "RaciTitle": norm(item.get("raci_title") or candidate_map[titlekey]["RaciTitle"]),
            "CandidateConfidence": max(0.0, min(1.0, float(item.get("confidence", 0)))),
            "WhyPlausible": norm(item.get("why_plausible") or ""),
        })

    if decision_type == "NO_MATCH":
        return {
            "DecisionType": "NO_MATCH",
            "SelectedTitleKey": None,
            "SelectedRaciTitle": None,
            "Confidence": confidence,
            "ReasoningSummary": reasoning_summary or "No credible match among the provided candidates.",
            "TopCandidates": top_candidates,
        }

    if decision_type != "MATCH":
        raise ValueError(f"Invalid decision_type: {decision_type}")

    if not selected_titlekey:
        raise ValueError("MATCH requires selected_titlekey")

    selected_titlekey = norm(selected_titlekey)

    if selected_titlekey not in candidate_map:
        raise ValueError(f"SelectedTitleKey not in provided candidates: {selected_titlekey}")

    candidate = candidate_map[selected_titlekey]

    if not selected_raci_title:
        selected_raci_title = norm(candidate["RaciTitle"])

    # MATCH: must have at least one top_candidate and rank 1 must equal selected_titlekey
    if not top_candidates or norm(top_candidates[0]["TitleKey"]) != selected_titlekey:
        raise ValueError("When MATCH, top_candidates must contain selected_titlekey as rank 1")

    return {
        "DecisionType": "MATCH",
        "SelectedTitleKey": selected_titlekey,
        "SelectedRaciTitle": norm(selected_raci_title),
        "Confidence": confidence,
        "ReasoningSummary": reasoning_summary or "Selected best semantic match among provided candidates.",
        "TopCandidates": top_candidates,
    }


def save_agent2_evaluation(
    con: duckdb.DuckDBPyConnection,
    task: Dict[str, Any],
    model: str,
    result: Dict[str, Any],
    manage_transaction: bool = True,
) -> None:
    ts = now_ts_naive_utc()

    if manage_transaction:
        con.execute("BEGIN;")
    try:
        con.execute("""
            INSERT INTO my_db.mdr_reconciliation.MdrReconciliationAgentDecisions
              (TaskId, AgentName, AgentModel, Document_title, PromptVersion, EmbeddingModel,
               SelectedTitleKey, SelectedRaciTitle, DecisionType, Confidence, ReasoningSummary, CreatedAt)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (TaskId, AgentName) DO UPDATE SET
              AgentModel = excluded.AgentModel,
              Document_title = excluded.Document_title,
              PromptVersion = excluded.PromptVersion,
              EmbeddingModel = excluded.EmbeddingModel,
              SelectedTitleKey = excluded.SelectedTitleKey,
              SelectedRaciTitle = excluded.SelectedRaciTitle,
              DecisionType = excluded.DecisionType,
              Confidence = excluded.Confidence,
              ReasoningSummary = excluded.ReasoningSummary,
              CreatedAt = excluded.CreatedAt
        """, [
            task["TaskId"],
            AGENT_NAME,
            model,
            task["Document_title"],
            task["PromptVersion"],
            task["EmbeddingModel"],
            result["SelectedTitleKey"],
            result["SelectedRaciTitle"],
            result["DecisionType"],
            result["Confidence"],
            result["ReasoningSummary"],
            ts
        ])

        # Write top_candidates to MdrReconciliationAgentTopCandidates
        con.execute(f"""
            DELETE FROM {AGENT_TOP_CANDIDATES_TABLE}
            WHERE TaskId = ? AND AgentName = ?
        """, [task["TaskId"], AGENT_NAME])
        for tc in result.get("TopCandidates") or []:
            con.execute(f"""
                INSERT INTO {AGENT_TOP_CANDIDATES_TABLE}
                  (TaskId, AgentName, PromptVersion, ModelName, CandidateRankWithinAgent,
                   TitleKey, RaciTitle, CandidateConfidence, WhyPlausible, CreatedAt)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                task["TaskId"],
                AGENT_NAME,
                task["PromptVersion"],
                model,
                tc["CandidateRankWithinAgent"],
                tc["TitleKey"],
                tc.get("RaciTitle"),
                tc.get("CandidateConfidence"),
                tc.get("WhyPlausible"),
                ts,
            ])

        con.execute("""
            UPDATE my_db.mdr_reconciliation.MdrReconciliationTasks
            SET
              Agent2Status = 'done',
              FinalStatus = CASE
                WHEN Agent1Status = 'done' THEN 'ready_for_judge'
                ELSE 'in_progress'
              END,
              UpdatedAt = ?
            WHERE TaskId = ?
        """, [ts, task["TaskId"]])

        if manage_transaction:
            con.execute("COMMIT;")
    except Exception:
        if manage_transaction:
            con.execute("ROLLBACK;")
        raise

def mark_agent2_error(
    con: duckdb.DuckDBPyConnection,
    task_id: str,
    batch_id: Optional[str] = None,
) -> None:
    """Set Agent2Status to 'error' or 'error_{batch_id}' and FinalStatus to 'error'."""
    ts = now_ts_naive_utc()
    status = f"error_{batch_id}" if batch_id else "error"
    con.execute("""
        UPDATE my_db.mdr_reconciliation.MdrReconciliationTasks
        SET
          Agent2Status = ?,
          FinalStatus = 'error',
          UpdatedAt = ?
        WHERE TaskId = ?
    """, [status, ts, task_id])


def reset_agent2_batch_statuses_to_pending(
    con: duckdb.DuckDBPyConnection,
    batch_id: str,
) -> int:
    """
    Reset submitted/in_batch/error statuses for one failed batch back to pending,
    only for tasks that do not already have an Agent2 decision saved.
    """
    ts = now_ts_naive_utc()
    status_submitted = f"submitted_{batch_id}"
    status_in_batch = f"in_batch_{batch_id}"
    status_error = f"error_{batch_id}"
    rows = con.execute(
        """
        UPDATE my_db.mdr_reconciliation.MdrReconciliationTasks t
        SET
          Agent2Status = 'pending',
          FinalStatus = CASE
            WHEN t.FinalStatus IN ('in_progress', 'error') THEN 'pending'
            ELSE t.FinalStatus
          END,
          UpdatedAt = ?
        FROM (
          SELECT t2.TaskId
          FROM my_db.mdr_reconciliation.MdrReconciliationTasks t2
          LEFT JOIN my_db.mdr_reconciliation.MdrReconciliationAgentDecisions d2
            ON d2.TaskId = t2.TaskId
           AND d2.AgentName = ?
          WHERE t2.Agent2Status IN (?, ?, ?)
            AND d2.TaskId IS NULL
        ) x
        WHERE t.TaskId = x.TaskId
        RETURNING t.TaskId
        """,
        [ts, AGENT_NAME, status_submitted, status_in_batch, status_error],
    ).fetchall()
    return len(rows)


def _extract_text_from_batch_message(message: Any) -> str:
    """Extract plain text from batch result message (object or dict)."""
    content = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else [])
    parts = []
    for block in content or []:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text") or "")
        elif getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts).strip()


def run_batch_submit(
    con: duckdb.DuckDBPyConnection,
    tasks: List[Dict[str, Any]],
    model: str,
) -> List[str]:
    return run_batch_submit_chunked(
        con=con,
        tasks=tasks,
        model=model,
        target_max_bytes=DEFAULT_BATCH_TARGET_BYTES,
    )


def run_batch_submit_chunked(
    con: duckdb.DuckDBPyConnection,
    tasks: List[Dict[str, Any]],
    model: str,
    target_max_bytes: int,
) -> List[str]:
    """Build and submit multiple Claude batches under byte target."""
    if target_max_bytes <= 0:
        raise ValueError("target_max_bytes must be > 0")
    if target_max_bytes > CLAUDE_BATCH_REQUEST_HARD_LIMIT_BYTES:
        raise ValueError(
            f"target_max_bytes cannot exceed hard limit {CLAUDE_BATCH_REQUEST_HARD_LIMIT_BYTES}"
        )

    current_requests: List[Dict[str, Any]] = []
    current_task_ids: List[str] = []
    current_bytes = 0
    batch_ids: List[str] = []
    valid_rows = 0
    total_bytes = 0
    skipped_too_large = 0
    ts = now_ts_naive_utc()

    def flush_chunk() -> None:
        nonlocal current_requests, current_task_ids, current_bytes
        if not current_requests:
            return
        batch = client.beta.messages.batches.create(requests=current_requests)
        batch_id = batch.id
        status_submitted = f"submitted_{batch_id}"
        for task_id in current_task_ids:
            con.execute("""
                UPDATE my_db.mdr_reconciliation.MdrReconciliationTasks
                SET Agent2Status = ?,
                    FinalStatus = CASE WHEN FinalStatus = 'pending' THEN 'in_progress' ELSE FinalStatus END,
                    UpdatedAt = ?
                WHERE TaskId = ?
            """, [status_submitted, ts, task_id])
        batch_ids.append(batch_id)
        # Persist after each submitted chunk so partial progress is recoverable on later failures.
        BATCH_IDS_FILE.write_text(
            json.dumps(batch_ids, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"Submitted chunk {len(batch_ids)}: batch_id={batch_id}, "
            f"tasks={len(current_task_ids)}, payload_bytes~={current_bytes}"
        )
        current_requests = []
        current_task_ids = []
        current_bytes = 0

    total_tasks = len(tasks)
    for i, task in enumerate(tasks, start=1):
        candidates = fetch_candidates_for_task(
            con=con,
            document_title=task["Document_title"],
            prompt_version=task["PromptVersion"],
            embedding_model=task["EmbeddingModel"],
        )
        if i % 100 == 0 or i == total_tasks:
            print(f"Batch build progress: {i}/{total_tasks}")
        if not candidates:
            continue
        mdr_ctx = fetch_mdr_context(con, task["Document_title"])
        user_prompt = build_user_prompt(mdr_ctx, candidates)
        req = {
            "custom_id": task["TaskId"],
            "params": {
                "model": model,
                "max_tokens": 1200,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_prompt}],
            },
        }
        req_bytes = len(json.dumps(req, ensure_ascii=False).encode("utf-8")) + 2
        if req_bytes > target_max_bytes:
            mark_agent2_error(con, str(task["TaskId"]))
            skipped_too_large += 1
            print(
                f"  {task['TaskId']}: skipped (single request {req_bytes} bytes exceeds target {target_max_bytes})"
            )
            continue
        if current_requests and (current_bytes + req_bytes > target_max_bytes):
            flush_chunk()
        current_requests.append(req)
        current_task_ids.append(str(task["TaskId"]))
        current_bytes += req_bytes
        valid_rows += 1
        total_bytes += req_bytes

    flush_chunk()

    if valid_rows == 0:
        raise RuntimeError("No valid tasks to submit (all missing candidates?).")

    avg_bytes = total_bytes / max(valid_rows, 1)
    estimated_tasks_per_batch = max(1, int(target_max_bytes // max(avg_bytes, 1)))
    estimated_batch_count = math.ceil(valid_rows / estimated_tasks_per_batch)
    print(
        "Batch sizing summary: "
        f"valid_tasks={valid_rows}, avg_request_bytes={avg_bytes:.0f}, "
        f"target_max_bytes={target_max_bytes}, "
        f"estimated_tasks_per_batch={estimated_tasks_per_batch}, "
        f"estimated_batches={estimated_batch_count}, "
        f"submitted_batches={len(batch_ids)}, skipped_too_large={skipped_too_large}"
    )
    BATCH_IDS_FILE.write_text(
        json.dumps(batch_ids, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return batch_ids


def run_batch_collect(
    con: duckdb.DuckDBPyConnection,
    batch_id: str,
    model: str,
    poll_interval: int = 60,
    skip_done: bool = False,
    db_commit_every: int = 200,
) -> Dict[str, Any]:
    """Poll batch until ended, then stream results and write each to DB."""
    while True:
        batch = client.beta.messages.batches.retrieve(message_batch_id=batch_id)
        status = getattr(batch, "processing_status", None) or (batch.get("processing_status") if isinstance(batch, dict) else None)
        if status == "ended":
            break
        print(f"Batch {batch_id} still processing (status={status}), waiting {poll_interval}s...")
        time.sleep(poll_interval)
    # Passaggio submitted_{id} -> in_batch_{id} (batch terminato, in scrittura risultati)
    status_submitted = f"submitted_{batch_id}"
    status_in_batch = f"in_batch_{batch_id}"
    con.execute("""
        UPDATE my_db.mdr_reconciliation.MdrReconciliationTasks
        SET Agent2Status = ?, UpdatedAt = ?
        WHERE Agent2Status = ?
    """, [status_in_batch, now_ts_naive_utc(), status_submitted])
    saved = 0
    errors = 0
    skipped_done = 0
    writes_since_commit = 0
    transaction_open = False
    candidates_cache: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}

    if db_commit_every <= 0:
        db_commit_every = 200

    con.execute("BEGIN;")
    transaction_open = True

    def commit_if_needed(force: bool = False) -> None:
        nonlocal writes_since_commit, transaction_open
        if not transaction_open:
            return
        if writes_since_commit <= 0:
            return
        if force or writes_since_commit >= db_commit_every:
            con.execute("COMMIT;")
            con.execute("BEGIN;")
            writes_since_commit = 0

    # Stream can occasionally drop on large result sets; retry and skip already handled tasks.
    handled_task_ids: set[str] = set()
    stream_attempt = 0
    max_stream_retries = 5
    while True:
        try:
            for item in client.beta.messages.batches.results(message_batch_id=batch_id):
                custom_id = getattr(item, "custom_id", None) or (item.get("custom_id") if isinstance(item, dict) else None)
                result = getattr(item, "result", None) or (item.get("result") if isinstance(item, dict) else None)
                if not custom_id or not result:
                    continue
                task_id = str(custom_id)
                if task_id in handled_task_ids:
                    continue
                result_type = getattr(result, "type", None) or (result.get("type") if isinstance(result, dict) else None)
                if result_type == "succeeded":
                    row = con.execute("""
                        SELECT TaskId, Document_title, PromptVersion, EmbeddingModel, Agent2Status
                        FROM my_db.mdr_reconciliation.MdrReconciliationTasks
                        WHERE TaskId = ?
                    """, [task_id]).fetchone()
                    if not row:
                        errors += 1
                        handled_task_ids.add(task_id)
                        continue
                    if skip_done and row[4] == "done":
                        skipped_done += 1
                        handled_task_ids.add(task_id)
                        if skipped_done % 100 == 0:
                            print(f"  skipped done: {skipped_done}")
                        continue
                    task = {"TaskId": row[0], "Document_title": row[1], "PromptVersion": row[2], "EmbeddingModel": row[3]}
                    msg = getattr(result, "message", None) or (result.get("message") if isinstance(result, dict) else None)
                    if not msg:
                        mark_agent2_error(con, task_id, batch_id=batch_id)
                        errors += 1
                        writes_since_commit += 1
                        commit_if_needed()
                        handled_task_ids.add(task_id)
                        continue
                    raw_text = _extract_text_from_batch_message(msg)
                    try:
                        cleaned = extract_json_payload(raw_text)
                        raw_result = json.loads(cleaned)
                    except (json.JSONDecodeError, ValueError) as e:
                        mark_agent2_error(con, task_id, batch_id=batch_id)
                        errors += 1
                        writes_since_commit += 1
                        commit_if_needed()
                        handled_task_ids.add(task_id)
                        print(f"  {task_id}: parse error {e}")
                        print("    ---- CLAUDE BATCH RAW TEXT BEGIN ----")
                        print(raw_text)
                        print("    ---- CLAUDE BATCH RAW TEXT END ----")
                        print("    ---- CLAUDE BATCH CLEANED JSON BEGIN ----")
                        print(cleaned)
                        print("    ---- CLAUDE BATCH CLEANED JSON END ----")
                        continue
                    candidate_key = (
                        task["Document_title"],
                        task["PromptVersion"],
                        task["EmbeddingModel"],
                    )
                    candidates = candidates_cache.get(candidate_key)
                    if candidates is None:
                        candidates = fetch_candidates_for_task(
                            con,
                            task["Document_title"],
                            task["PromptVersion"],
                            task["EmbeddingModel"],
                        )
                        candidates_cache[candidate_key] = candidates
                    if not candidates:
                        mark_agent2_error(con, task_id, batch_id=batch_id)
                        errors += 1
                        writes_since_commit += 1
                        commit_if_needed()
                        handled_task_ids.add(task_id)
                        continue
                    try:
                        validated = validate_agent_output(raw_result, candidates)
                        save_agent2_evaluation(
                            con=con,
                            task=task,
                            model=model,
                            result=validated,
                            manage_transaction=False,
                        )
                        saved += 1
                        writes_since_commit += 1
                        commit_if_needed()
                        handled_task_ids.add(task_id)
                        print(f"  {task_id} -> {validated['DecisionType']} (saved)")
                    except Exception as e:
                        mark_agent2_error(con, task_id, batch_id=batch_id)
                        errors += 1
                        writes_since_commit += 1
                        commit_if_needed()
                        handled_task_ids.add(task_id)
                        print(f"  {task_id}: validation/save error {e}")
                else:
                    mark_agent2_error(con, task_id, batch_id=batch_id)
                    errors += 1
                    writes_since_commit += 1
                    commit_if_needed()
                    handled_task_ids.add(task_id)
                    print(f"  {task_id}: result type={result_type}")
            break
        except Exception as e:
            is_stream_drop = isinstance(e, httpx.RemoteProtocolError) or "incomplete chunked read" in str(e).lower()
            if is_stream_drop and stream_attempt < max_stream_retries:
                stream_attempt += 1
                wait_s = min(5 * stream_attempt, 30)
                print(
                    f"Batch result stream interrupted ({e}); retry {stream_attempt}/{max_stream_retries} in {wait_s}s..."
                )
                time.sleep(wait_s)
                continue
            if transaction_open:
                con.execute("ROLLBACK;")
                transaction_open = False
            raise
    if transaction_open:
        commit_if_needed(force=True)
        con.execute("COMMIT;")
    print(f"Batch collect done: {saved} saved, {errors} errors, {skipped_done} skipped_done.")
    return {
        "batch_id": batch_id,
        "status": "completed",
        "saved": saved,
        "errors": errors,
        "skipped_done": skipped_done,
    }


def run_batch_and_collect_adaptive(
    con: duckdb.DuckDBPyConnection,
    prompt_version: str,
    embedding_model: str,
    model: str,
    target_max_bytes: int,
    initial_limit: int,
    min_limit: int,
    max_limit: int,
    backoff_factor: float,
    growth_factor: float,
    skip_done: bool,
    max_rounds: Optional[int],
    test_fixed_tasks: bool = False,
    test_task_count: int = 150,
) -> None:
    current_limit = max(min_limit, min(max_limit, int(initial_limit)))
    rounds = 0
    fixed_task_ids: Optional[set] = None

    if test_fixed_tasks:
        all_pending = fetch_pending_tasks(
            con=con,
            prompt_version=prompt_version,
            embedding_model=embedding_model,
            limit=None,
        )
        fixed_tasks = load_or_create_test_tasks(all_pending, test_task_count)
        fixed_task_ids = {str(t.get("TaskId")) for t in fixed_tasks if t.get("TaskId")}
        print(f"Test mode: fixed TaskId loaded ({len(fixed_task_ids)} task).")

    while True:
        if max_rounds is not None and rounds >= max_rounds:
            print(f"Stopping: reached max rounds ({max_rounds}).")
            return

        if fixed_task_ids is None:
            tasks = fetch_pending_tasks(
                con=con,
                prompt_version=prompt_version,
                embedding_model=embedding_model,
                limit=current_limit,
            )
        else:
            all_pending = fetch_pending_tasks(
                con=con,
                prompt_version=prompt_version,
                embedding_model=embedding_model,
                limit=None,
            )
            tasks = [t for t in all_pending if str(t.get("TaskId")) in fixed_task_ids][:current_limit]

        if not tasks:
            print("No pending tasks for adaptive batch-and-collect. Done.")
            return

        rounds += 1
        print(
            f"[Adaptive round {rounds}] submitting {len(tasks)} task(s) "
            f"(current_limit={current_limit}, min={min_limit}, max={max_limit})"
        )
        try:
            batch_ids = run_batch_submit_chunked(
                con=con,
                tasks=tasks,
                model=model,
                target_max_bytes=target_max_bytes,
            )
        except Exception as e:
            next_limit = max(min_limit, int(max(1, current_limit) * float(backoff_factor)))
            if next_limit == current_limit and current_limit > min_limit:
                next_limit = current_limit - 1
            current_limit = max(min_limit, next_limit)
            print(
                f"[Adaptive round {rounds}] submit failed ({e}). "
                f"Reducing limit to {current_limit} and retrying."
            )
            continue

        all_completed = True
        for i, batch_id in enumerate(batch_ids, start=1):
            print(f"[Adaptive round {rounds}] collecting batch {i}/{len(batch_ids)}: {batch_id}")
            try:
                collect_result = run_batch_collect(
                    con=con,
                    batch_id=batch_id,
                    model=model,
                    skip_done=skip_done,
                )
            except Exception as e:
                reset_n = reset_agent2_batch_statuses_to_pending(con, batch_id)
                print(
                    f"[Adaptive round {rounds}] batch {batch_id} collect exception ({e}), "
                    f"reset to pending: {reset_n} task(s)."
                )
                next_limit = max(min_limit, int(max(1, current_limit) * float(backoff_factor)))
                if next_limit == current_limit and current_limit > min_limit:
                    next_limit = current_limit - 1
                current_limit = max(min_limit, next_limit)
                all_completed = False
                break
            if collect_result["status"] != "completed":
                reset_n = reset_agent2_batch_statuses_to_pending(con, batch_id)
                print(
                    f"[Adaptive round {rounds}] batch {batch_id} failed "
                    f"({collect_result['status']}), reset to pending: {reset_n} task(s)."
                )
                next_limit = max(min_limit, int(max(1, current_limit) * float(backoff_factor)))
                if next_limit == current_limit and current_limit > min_limit:
                    next_limit = current_limit - 1
                current_limit = max(min_limit, next_limit)
                all_completed = False
                break

        if all_completed:
            next_limit = min(max_limit, int(max(1, current_limit) * float(growth_factor)))
            if next_limit == current_limit and current_limit < max_limit:
                next_limit = current_limit + 1
            current_limit = min(max_limit, next_limit)
            print(f"[Adaptive round {rounds}] completed successfully. Next limit: {current_limit}")


def process_one_agent2_task(
    con: duckdb.DuckDBPyConnection,
    task: Dict[str, Any],
    model: str
) -> Optional[Dict[str, Any]]:
    """Run agent2 pipeline for one task (caller must have claimed it). Returns result on success, None on skip/error."""
    candidates = fetch_candidates_for_task(
        con=con,
        document_title=task["Document_title"],
        prompt_version=task["PromptVersion"],
        embedding_model=task["EmbeddingModel"]
    )
    if not candidates:
        mark_agent2_error(con, task["TaskId"])
        return None
    mdr_ctx = fetch_mdr_context(con, task["Document_title"])
    raw_result = call_agent(model=model, mdr_ctx=mdr_ctx, candidates=candidates)
    validated_result = validate_agent_output(raw_result, candidates)
    save_agent2_evaluation(con=con, task=task, model=model, result=validated_result)
    return validated_result


def _worker(
    task_queue: queue.Queue,
    model: str,
    print_lock: threading.Lock,
    total_tasks: int,
    completed_count: List[int],
) -> None:
    con = connect_motherduck()
    try:
        while True:
            task = task_queue.get()
            processed = False
            try:
                if task is None:
                    return
                claimed = claim_task_agent2(con, task["TaskId"])
                if not claimed:
                    continue
                processed = True
                with print_lock:
                    print(f"[{threading.current_thread().name}] Processing: {task['Document_title']}")
                result = process_one_agent2_task(con, task, model)
                if result:
                    with print_lock:
                        print(
                            f"  Saved -> decision={result['DecisionType']} "
                            f"titlekey={result['SelectedTitleKey']} "
                            f"confidence={result['Confidence']:.3f}"
                        )
            except Exception as e:
                mark_agent2_error(con, task["TaskId"])
                with print_lock:
                    print(f"  ERROR: {e}")
            finally:
                task_queue.task_done()
                if task is not None and processed:
                    with print_lock:
                        completed_count[0] += 1
                        n = completed_count[0]
                        print(f"Progress: {n}/{total_tasks} (remaining: {total_tasks - n})")
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-version", default=None, help="PromptVersion (default: from config.txt PROMPT_VERSION).")
    ap.add_argument("--embedding-model", default=None, help="EmbeddingModel (default: from config or text-embedding-3-small).")
    ap.add_argument("--limit", type=int, default=None, help="Max number of tasks to process (default: no limit, process all pending).")
    ap.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4).")
    ap.add_argument("--model", default=None, help="Anthropic model for agent (default: claude-sonnet-4-6).")
    ap.add_argument("--batch", action="store_true", help="Use Batch API: submit pending tasks and exit (no rate limit; collect later with --batch-collect).")
    ap.add_argument("--batch-collect", action="store_true", help="Poll and collect all batch ids saved by latest submit in .agent2_last_batch_ids.json.")
    ap.add_argument(
        "--batch-and-collect",
        action="store_true",
        help=(
            "Adaptive loop: submit one tranche, collect immediately, and continue automatically. "
            "On failed collect/submit, reset related statuses to pending and retry with lower limit."
        ),
    )
    ap.add_argument(
        "--batch-collect-skip-done",
        action="store_true",
        help="When used with --batch-collect, skip tasks already marked Agent2Status='done'.",
    )
    ap.add_argument(
        "--batch-max-bytes",
        type=int,
        default=DEFAULT_BATCH_TARGET_BYTES,
        help=(
            "Target max serialized bytes per submitted Claude batch chunk "
            f"(default: {DEFAULT_BATCH_TARGET_BYTES}, hard max: {CLAUDE_BATCH_REQUEST_HARD_LIMIT_BYTES})."
        ),
    )
    ap.add_argument(
        "--batch-initial-limit",
        type=int,
        default=DEFAULT_ADAPTIVE_BATCH_INITIAL_LIMIT,
        help=f"Initial tranche size for --batch-and-collect (default: {DEFAULT_ADAPTIVE_BATCH_INITIAL_LIMIT}).",
    )
    ap.add_argument(
        "--batch-min-limit",
        type=int,
        default=DEFAULT_ADAPTIVE_BATCH_MIN_LIMIT,
        help=f"Minimum tranche size for --batch-and-collect backoff (default: {DEFAULT_ADAPTIVE_BATCH_MIN_LIMIT}).",
    )
    ap.add_argument(
        "--batch-max-limit",
        type=int,
        default=DEFAULT_ADAPTIVE_BATCH_MAX_LIMIT,
        help=f"Maximum tranche size for --batch-and-collect growth (default: {DEFAULT_ADAPTIVE_BATCH_MAX_LIMIT}).",
    )
    ap.add_argument(
        "--batch-backoff-factor",
        type=float,
        default=DEFAULT_ADAPTIVE_BATCH_BACKOFF_FACTOR,
        help=f"Backoff multiplier after failed round in --batch-and-collect (default: {DEFAULT_ADAPTIVE_BATCH_BACKOFF_FACTOR}).",
    )
    ap.add_argument(
        "--batch-growth-factor",
        type=float,
        default=DEFAULT_ADAPTIVE_BATCH_GROWTH_FACTOR,
        help=f"Growth multiplier after successful round in --batch-and-collect (default: {DEFAULT_ADAPTIVE_BATCH_GROWTH_FACTOR}).",
    )
    ap.add_argument(
        "--batch-max-rounds",
        type=int,
        default=None,
        help="Optional max adaptive rounds for --batch-and-collect (default: no limit).",
    )
    ap.add_argument("--test-fixed-tasks", action="store_true", help="Test mode: use a shared fixed list of TaskId stored in .recon_test_tasks.json.")
    ap.add_argument("--test-task-count", type=int, default=150, help="Number of random TaskId to pick when creating the shared test list (default: 150).")
    args = ap.parse_args()

    args.prompt_version = args.prompt_version or _cfg("PROMPT_VERSION", "v1")
    args.embedding_model = args.embedding_model or "text-embedding-3-small"
    args.model = args.model or DEFAULT_MODEL
    if args.batch_and_collect and (args.batch or args.batch_collect):
        raise RuntimeError("Use --batch-and-collect alone (do not combine with --batch or --batch-collect).")
    if args.batch_and_collect and args.batch_min_limit <= 0:
        raise RuntimeError("--batch-min-limit must be > 0.")
    if args.batch_and_collect and args.batch_max_limit < args.batch_min_limit:
        raise RuntimeError("--batch-max-limit must be >= --batch-min-limit.")
    if args.batch_and_collect and not (0 < args.batch_backoff_factor <= 1):
        raise RuntimeError("--batch-backoff-factor must be in (0, 1].")
    if args.batch_and_collect and args.batch_growth_factor < 1:
        raise RuntimeError("--batch-growth-factor must be >= 1.")

    con = connect_motherduck()
    ensure_agent_eval_table(con)
    ensure_agent_top_candidates_table(con)
    if args.batch_and_collect:
        run_batch_and_collect_adaptive(
            con=con,
            prompt_version=args.prompt_version,
            embedding_model=args.embedding_model,
            model=args.model,
            target_max_bytes=args.batch_max_bytes,
            initial_limit=args.batch_initial_limit,
            min_limit=args.batch_min_limit,
            max_limit=args.batch_max_limit,
            backoff_factor=args.batch_backoff_factor,
            growth_factor=args.batch_growth_factor,
            skip_done=args.batch_collect_skip_done,
            max_rounds=args.batch_max_rounds,
            test_fixed_tasks=args.test_fixed_tasks,
            test_task_count=args.test_task_count,
        )
        con.close()
        return

    if args.batch_collect:
        if not BATCH_IDS_FILE.exists():
            print("Error: no .agent2_last_batch_ids.json file. Run --batch first.")
            con.close()
            return
        try:
            batch_ids = json.loads(BATCH_IDS_FILE.read_text(encoding="utf-8"))
            if not isinstance(batch_ids, list):
                raise ValueError("invalid JSON shape")
            batch_ids = [str(x).strip() for x in batch_ids if str(x).strip()]
        except Exception as e:
            print(f"Error reading {BATCH_IDS_FILE.name}: {e}")
            con.close()
            return
        if not batch_ids:
            print("No batch ids to collect.")
            con.close()
            return
        print(f"Collecting {len(batch_ids)} batch(es) from {BATCH_IDS_FILE.name}...")
        for i, batch_id in enumerate(batch_ids, start=1):
            print(f"[{i}/{len(batch_ids)}] Collecting results for batch: {batch_id}")
            run_batch_collect(
                con=con,
                batch_id=batch_id,
                model=args.model,
                skip_done=args.batch_collect_skip_done,
            )
        con.close()
        return

    if args.batch:
        tasks = fetch_pending_tasks(
            con=con,
            prompt_version=args.prompt_version,
            embedding_model=args.embedding_model,
            limit=args.limit,
        )
        if args.test_fixed_tasks:
            tasks = load_or_create_test_tasks(tasks, args.test_task_count)
        if not tasks:
            print("No pending tasks for batch.")
            con.close()
            return
        batch_ids = run_batch_submit_chunked(
            con=con,
            tasks=tasks,
            model=args.model,
            target_max_bytes=args.batch_max_bytes,
        )
        con.close()
        if len(batch_ids) == 1:
            print(f"Batch submitted: {batch_ids[0]}")
        else:
            print(
                f"Batches submitted: {len(batch_ids)} (last={batch_ids[-1]}). "
                "Use --batch-collect to collect all results."
            )
        print("Run with --batch-collect later to write results to DB.")
        return

    tasks = fetch_pending_tasks(
        con=con,
        prompt_version=args.prompt_version,
        embedding_model=args.embedding_model,
        limit=args.limit
    )
    con.close()

    if args.test_fixed_tasks:
        tasks = load_or_create_test_tasks(tasks, args.test_task_count)

    print(f"Pending tasks fetched for {AGENT_NAME}: {len(tasks)} (workers={args.workers})")

    if not tasks:
        print("Done.")
        return

    total_tasks = len(tasks)
    task_queue: queue.Queue = queue.Queue()
    for task in tasks:
        task_queue.put(task)
    print_lock = threading.Lock()
    completed_count: List[int] = [0]

    workers = [
        threading.Thread(
            target=_worker,
            args=(task_queue, args.model, print_lock, total_tasks, completed_count),
            name=f"agent2-{i}",
            daemon=True,
        )
        for i in range(args.workers)
    ]
    for t in workers:
        t.start()
    for _ in range(args.workers):
        task_queue.put(None)
    try:
        task_queue.join()
        for t in workers:
            t.join()
        print("Done.")
    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl+C). Exiting...")


if __name__ == "__main__":
    main()
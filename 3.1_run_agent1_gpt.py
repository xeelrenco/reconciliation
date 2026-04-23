#!/usr/bin/env python3
"""
Agent 1 (GPT) per riconciliazione MDR: valida i candidati RACI per ogni task e scrive
le decisioni in MdrReconciliationAgentDecisions.

Config: config.txt (MOTHERDUCK_*, OPENAI_API_KEY, PROMPT_VERSION). Vedi config.example.txt.

Uso in tempo reale (default)
  Elabora i task pending con N worker paralleli.
  python 3.1_run_agent1_gpt.py [--prompt-version v1] [--embedding-model ...] [--limit N] [--workers 4] [--model ...]

Uso con Batch API (rate limit separati, ~50% costo)
  1) Submit: invia i task pending in batch chunked automatici (elaborazione asincrona, entro 24h).
     python 3.1_run_agent1_gpt.py --batch [--limit N] [--batch-max-bytes 120000000]
     Ogni chunk genera un batch id; la lista viene salvata in .agent1_last_batch_ids.json.
  2) Collect: quando il batch è completato, scarica i risultati e scrive in DB.
     python 3.1_run_agent1_gpt.py --batch-collect
     Colleziona automaticamente tutti i batch dell'ultimo submit (anche se è uno solo).

Stati Agent1Status (batch): pending | submitted_{batch_id} | in_batch_{batch_id} | done | error_{batch_id}
  (solo 'done' senza suffisso; gli altri stati batch hanno _batch_id per tracciare il batch.)
"""

import json
import argparse
import queue
import threading
import tempfile
import time
import random
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
from openai import OpenAI

# -----------------------------
# Config da file (stesso formato degli altri script)
# -----------------------------
CONFIG_PATH = Path(__file__).resolve().parent / "config.txt"
BATCH_IDS_FILE = Path(__file__).resolve().parent / ".agent1_last_batch_ids.json"
BATCH_ENDPOINT = "/v1/responses"
TEST_TASK_FILE = Path(__file__).resolve().parent / ".recon_test_tasks.json"
OPENAI_BATCH_INPUT_FILE_HARD_LIMIT_BYTES = 209_715_200  # 200 MB
DEFAULT_BATCH_TARGET_BYTES = 120_000_000  # conservative default to reduce token-queue failures
DEFAULT_ADAPTIVE_BATCH_INITIAL_LIMIT = 800
DEFAULT_ADAPTIVE_BATCH_MIN_LIMIT = 100
DEFAULT_ADAPTIVE_BATCH_MAX_LIMIT = 1200
DEFAULT_ADAPTIVE_BATCH_BACKOFF_FACTOR = 0.5
DEFAULT_ADAPTIVE_BATCH_GROWTH_FACTOR = 1.25


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


AGENT_NAME = "gpt5mini"
DEFAULT_MODEL = "gpt-5-mini"

client = OpenAI(api_key=_cfg("OPENAI_API_KEY"))


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
      DecisionType        VARCHAR NOT NULL,   -- MATCH | NO_MATCH
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
    # Filter original tasks by selected ids to preserve potential ordering
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
          AND Agent1Status = 'pending'
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


def claim_task_agent1(con: duckdb.DuckDBPyConnection, task_id: str) -> bool:
    ts = now_ts_naive_utc()
    con.execute("BEGIN;")
    try:
        rows = con.execute("""
            UPDATE my_db.mdr_reconciliation.MdrReconciliationTasks
            SET
              Agent1Status = 'running',
              FinalStatus = CASE
                WHEN FinalStatus = 'pending' THEN 'in_progress'
                ELSE FinalStatus
              END,
              UpdatedAt = ?
            WHERE TaskId = ?
              AND Agent1Status = 'pending'
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


TOP_CANDIDATE_ITEM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "rank": {"type": "integer"},
        "titlekey": {"type": "string"},
        "raci_title": {"type": "string"},
        "confidence": {"type": "number"},
        "why_plausible": {"type": "string"},
    },
    "required": ["rank", "titlekey", "raci_title", "confidence", "why_plausible"],
}

RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision_type": {
            "type": "string",
            "enum": ["MATCH", "NO_MATCH"]
        },
        "selected_titlekey": {
            "type": ["string", "null"]
        },
        "selected_raci_title": {
            "type": ["string", "null"]
        },
        "confidence": {
            "type": "number"
        },
        "reasoning_summary": {
            "type": "string"
        },
        "top_candidates": {
            "type": "array",
            "items": TOP_CANDIDATE_ITEM_SCHEMA,
            "maxItems": 3,
        },
    },
    "required": [
        "decision_type",
        "selected_titlekey",
        "selected_raci_title",
        "confidence",
        "reasoning_summary",
        "top_candidates",
    ]
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


def call_agent(model: str, mdr_ctx: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    user_prompt = build_user_prompt(mdr_ctx, candidates)

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "mdr_title_agent_evaluation",
                "schema": RESPONSE_SCHEMA,
                "strict": True,
            }
        },
    )

    return json.loads(resp.output_text)


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


def save_agent1_evaluation(
    con: duckdb.DuckDBPyConnection,
    task: Dict[str, Any],
    model: str,
    result: Dict[str, Any]
) -> None:
    ts = now_ts_naive_utc()

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
              Agent1Status = 'done',
              FinalStatus = CASE
                WHEN Agent2Status = 'done' THEN 'ready_for_judge'
                ELSE 'in_progress'
              END,
              UpdatedAt = ?
            WHERE TaskId = ?
        """, [ts, task["TaskId"]])

        con.execute("COMMIT;")
    except Exception:
        con.execute("ROLLBACK;")
        raise

def mark_agent1_error(
    con: duckdb.DuckDBPyConnection,
    task_id: str,
    batch_id: Optional[str] = None,
) -> None:
    """Set Agent1Status to 'error' or 'error_{batch_id}' and FinalStatus to 'error'."""
    ts = now_ts_naive_utc()
    status = f"error_{batch_id}" if batch_id else "error"
    con.execute("""
        UPDATE my_db.mdr_reconciliation.MdrReconciliationTasks
        SET
          Agent1Status = ?,
          FinalStatus = 'error',
          UpdatedAt = ?
        WHERE TaskId = ?
    """, [status, ts, task_id])


def reset_agent1_batch_statuses_to_pending(
    con: duckdb.DuckDBPyConnection,
    batch_id: str,
) -> int:
    """
    Reset submitted/in_batch/error statuses for one failed batch back to pending,
    only for tasks that do not already have an Agent1 decision saved.
    """
    ts = now_ts_naive_utc()
    status_submitted = f"submitted_{batch_id}"
    status_in_batch = f"in_batch_{batch_id}"
    status_error = f"error_{batch_id}"
    rows = con.execute(
        """
        UPDATE my_db.mdr_reconciliation.MdrReconciliationTasks t
        SET
          Agent1Status = 'pending',
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
          WHERE t2.Agent1Status IN (?, ?, ?)
            AND d2.TaskId IS NULL
        ) x
        WHERE t.TaskId = x.TaskId
        RETURNING t.TaskId
        """,
        [ts, AGENT_NAME, status_submitted, status_in_batch, status_error],
    ).fetchall()
    return len(rows)


def _responses_batch_request_body(model: str, user_prompt: str) -> Dict[str, Any]:
    """Body for one /v1/responses request in the batch JSONL."""
    return {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "mdr_title_agent_evaluation",
                "schema": RESPONSE_SCHEMA,
                "strict": True,
            }
        },
    }


def _extract_output_text_from_batch_response_body(body: Dict[str, Any]) -> Optional[str]:
    """Extract output text from Responses API batch result response.body."""
    if not isinstance(body, dict):
        return None
    if body.get("output_text"):
        return body["output_text"]
    output = body.get("output") or []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "output_text":
                return block.get("text")
    return None


def _batch_obj_to_debug_json(batch: Any) -> str:
    """Serialize SDK batch object for readable debug logging."""
    try:
        if hasattr(batch, "model_dump"):
            return json.dumps(batch.model_dump(), ensure_ascii=False, indent=2, default=str)
        if hasattr(batch, "to_dict"):
            return json.dumps(batch.to_dict(), ensure_ascii=False, indent=2, default=str)
        if isinstance(batch, dict):
            return json.dumps(batch, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass
    return repr(batch)


def _build_batch_line_for_task(
    con: duckdb.DuckDBPyConnection,
    task: Dict[str, Any],
    model: str,
) -> Optional[Dict[str, Any]]:
    candidates = fetch_candidates_for_task(
        con=con,
        document_title=task["Document_title"],
        prompt_version=task["PromptVersion"],
        embedding_model=task["EmbeddingModel"],
    )
    if not candidates:
        return None
    mdr_ctx = fetch_mdr_context(con, task["Document_title"])
    user_prompt = build_user_prompt(mdr_ctx, candidates)
    body = _responses_batch_request_body(model, user_prompt)
    line = json.dumps({
        "custom_id": task["TaskId"],
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": body,
    })
    line_bytes = len((line + "\n").encode("utf-8"))
    return {"task_id": task["TaskId"], "line": line, "line_bytes": line_bytes}


def _submit_batch_chunk(lines: List[str]) -> Any:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
        tmp_path = f.name
    try:
        with open(tmp_path, "rb") as f:
            batch_file = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=batch_file.id,
            endpoint=BATCH_ENDPOINT,
            completion_window="24h",
        )
        return batch
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def run_batch_submit(
    con: duckdb.DuckDBPyConnection,
    tasks: List[Dict[str, Any]],
    model: str,
) -> List[str]:
    """Create one or more batch submissions under size target; returns submitted batch ids."""
    return run_batch_submit_chunked(con=con, tasks=tasks, model=model, target_max_bytes=DEFAULT_BATCH_TARGET_BYTES)


def run_batch_submit_chunked(
    con: duckdb.DuckDBPyConnection,
    tasks: List[Dict[str, Any]],
    model: str,
    target_max_bytes: int,
) -> List[str]:
    """Build/submit chunked batch files to stay below OpenAI file-size limits."""
    if target_max_bytes <= 0:
        raise ValueError("target_max_bytes must be > 0")
    if target_max_bytes > OPENAI_BATCH_INPUT_FILE_HARD_LIMIT_BYTES:
        raise ValueError(
            f"target_max_bytes cannot exceed hard limit {OPENAI_BATCH_INPUT_FILE_HARD_LIMIT_BYTES}"
        )

    current_lines: List[str] = []
    current_task_ids: List[str] = []
    current_bytes = 0
    batch_ids: List[str] = []
    valid_rows = 0
    total_bytes = 0
    skipped_too_large = 0
    ts = now_ts_naive_utc()

    def flush_chunk() -> None:
        nonlocal current_lines, current_task_ids, current_bytes
        if not current_lines:
            return
        batch = _submit_batch_chunk(current_lines)
        batch_id = batch.id
        status_submitted = f"submitted_{batch_id}"
        for task_id in current_task_ids:
            con.execute("""
                UPDATE my_db.mdr_reconciliation.MdrReconciliationTasks
                SET Agent1Status = ?,
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
            f"tasks={len(current_task_ids)}, jsonl_bytes={current_bytes}"
        )
        current_lines = []
        current_task_ids = []
        current_bytes = 0

    total_tasks = len(tasks)
    for i, task in enumerate(tasks, start=1):
        row = _build_batch_line_for_task(con=con, task=task, model=model)
        if i % 100 == 0 or i == total_tasks:
            print(f"Batch build progress: {i}/{total_tasks}")
        if not row:
            continue
        line = row["line"]
        line_bytes = int(row["line_bytes"])
        task_id = str(row["task_id"])
        if line_bytes > target_max_bytes:
            mark_agent1_error(con, task_id)
            skipped_too_large += 1
            print(
                f"  {task_id}: skipped (single request {line_bytes} bytes exceeds target {target_max_bytes})"
            )
            continue
        if current_lines and (current_bytes + line_bytes > target_max_bytes):
            flush_chunk()
        current_lines.append(line)
        current_task_ids.append(task_id)
        current_bytes += line_bytes
        valid_rows += 1
        total_bytes += line_bytes

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
) -> Dict[str, Any]:
    """Poll batch until completed, then download results and write each to DB."""
    while True:
        batch = client.batches.retrieve(batch_id)
        status = getattr(batch, "status", None) or (batch.get("status") if isinstance(batch, dict) else None)
        if status == "completed":
            break
        if status in ("failed", "canceled", "expired"):
            print(f"Batch {batch_id} ended with status={status}. Cannot collect results.")
            print("Batch API response:")
            print(_batch_obj_to_debug_json(batch))
            return {
                "batch_id": batch_id,
                "status": str(status),
                "saved": 0,
                "errors": 0,
                "skipped_done": 0,
            }
        print(f"Batch {batch_id} status={status}, waiting {poll_interval}s...")
        time.sleep(poll_interval)
    status_submitted = f"submitted_{batch_id}"
    status_in_batch = f"in_batch_{batch_id}"
    ts = now_ts_naive_utc()
    con.execute("""
        UPDATE my_db.mdr_reconciliation.MdrReconciliationTasks
        SET Agent1Status = ?, UpdatedAt = ?
        WHERE Agent1Status = ?
    """, [status_in_batch, ts, status_submitted])
    saved = 0
    errors = 0
    skipped_done = 0
    output_file_id = getattr(batch, "output_file_id", None) or (batch.get("output_file_id") if isinstance(batch, dict) else None)
    error_file_id = getattr(batch, "error_file_id", None) or (batch.get("error_file_id") if isinstance(batch, dict) else None)
    if output_file_id:
        content = client.files.content(output_file_id).content
        for line in content.decode("utf-8").strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue
            custom_id = row.get("custom_id")
            response = row.get("response") or {}
            body = response.get("body") if isinstance(response, dict) else {}
            if not custom_id:
                errors += 1
                continue
            task_id = str(custom_id)
            db_row = con.execute("""
                SELECT TaskId, Document_title, PromptVersion, EmbeddingModel, Agent1Status
                FROM my_db.mdr_reconciliation.MdrReconciliationTasks
                WHERE TaskId = ?
            """, [task_id]).fetchone()
            if not db_row:
                errors += 1
                continue
            if skip_done and db_row[4] == "done":
                skipped_done += 1
                continue
            output_text = _extract_output_text_from_batch_response_body(body) if body else None
            if not output_text:
                mark_agent1_error(con, task_id, batch_id=batch_id)
                errors += 1
                print(f"  {task_id}: no output text in response")
                continue
            try:
                raw_result = json.loads(output_text)
            except json.JSONDecodeError as e:
                mark_agent1_error(con, task_id, batch_id=batch_id)
                errors += 1
                print(f"  {task_id}: JSON parse error {e}")
                continue
            task = {"TaskId": db_row[0], "Document_title": db_row[1], "PromptVersion": db_row[2], "EmbeddingModel": db_row[3]}
            candidates = fetch_candidates_for_task(con, task["Document_title"], task["PromptVersion"], task["EmbeddingModel"])
            if not candidates:
                mark_agent1_error(con, task_id, batch_id=batch_id)
                errors += 1
                continue
            try:
                validated = validate_agent_output(raw_result, candidates)
                save_agent1_evaluation(con=con, task=task, model=model, result=validated)
                saved += 1
                print(f"  {task_id} -> {validated['DecisionType']} (saved)")
            except Exception as e:
                mark_agent1_error(con, task_id, batch_id=batch_id)
                errors += 1
                print(f"  {task_id}: validation/save error {e}")
    if error_file_id:
        err_content = client.files.content(error_file_id).content
        for line in err_content.decode("utf-8").strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            custom_id = row.get("custom_id")
            if custom_id:
                task_id = str(custom_id)
                if skip_done:
                    status_row = con.execute("""
                        SELECT Agent1Status
                        FROM my_db.mdr_reconciliation.MdrReconciliationTasks
                        WHERE TaskId = ?
                    """, [task_id]).fetchone()
                    if status_row and status_row[0] == "done":
                        skipped_done += 1
                        continue
                mark_agent1_error(con, task_id, batch_id=batch_id)
                errors += 1
                print(f"  {custom_id}: batch error (see error file)")
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
        batch_ids = run_batch_submit_chunked(
            con=con,
            tasks=tasks,
            model=model,
            target_max_bytes=target_max_bytes,
        )

        all_completed = True
        for i, batch_id in enumerate(batch_ids, start=1):
            print(f"[Adaptive round {rounds}] collecting batch {i}/{len(batch_ids)}: {batch_id}")
            collect_result = run_batch_collect(
                con=con,
                batch_id=batch_id,
                model=model,
                skip_done=skip_done,
            )
            if collect_result["status"] != "completed":
                reset_n = reset_agent1_batch_statuses_to_pending(con, batch_id)
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


def process_one_agent1_task(
    con: duckdb.DuckDBPyConnection,
    task: Dict[str, Any],
    model: str
) -> Optional[Dict[str, Any]]:
    """Run agent1 pipeline for one task (caller must have claimed it). Returns result on success, None on skip/error."""
    candidates = fetch_candidates_for_task(
        con=con,
        document_title=task["Document_title"],
        prompt_version=task["PromptVersion"],
        embedding_model=task["EmbeddingModel"]
    )
    if not candidates:
        mark_agent1_error(con, task["TaskId"])
        return None
    mdr_ctx = fetch_mdr_context(con, task["Document_title"])
    raw_result = call_agent(model=model, mdr_ctx=mdr_ctx, candidates=candidates)
    validated_result = validate_agent_output(raw_result, candidates)
    save_agent1_evaluation(con=con, task=task, model=model, result=validated_result)
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
                claimed = claim_task_agent1(con, task["TaskId"])
                if not claimed:
                    continue
                processed = True
                with print_lock:
                    print(f"[{threading.current_thread().name}] Processing: {task['Document_title']}")
                result = process_one_agent1_task(con, task, model)
                if result:
                    with print_lock:
                        print(
                            f"  Saved -> decision={result['DecisionType']} "
                            f"titlekey={result['SelectedTitleKey']} "
                            f"confidence={result['Confidence']:.3f}"
                        )
            except Exception as e:
                mark_agent1_error(con, task["TaskId"])
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
    ap.add_argument("--embedding-model", default=None, help="EmbeddingModel (default: from config.txt EMBEDDING_MODEL).")
    ap.add_argument("--limit", type=int, default=None, help="Max number of tasks to process (default: no limit, process all pending).")
    ap.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4).")
    ap.add_argument("--model", default=None, help="OpenAI model for agent (default: from config OPENAI_AGENT1_MODEL or gpt-5-mini).")
    ap.add_argument("--batch", action="store_true", help="Use Batch API: submit pending tasks and exit (collect later with --batch-collect).")
    ap.add_argument("--batch-collect", action="store_true", help="Poll and collect all batch ids saved by latest submit in .agent1_last_batch_ids.json.")
    ap.add_argument(
        "--batch-and-collect",
        action="store_true",
        help=(
            "Adaptive loop: submit one tranche, collect immediately, and continue automatically. "
            "On failed batch, reset related statuses to pending and retry with lower limit."
        ),
    )
    ap.add_argument(
        "--batch-collect-skip-done",
        action="store_true",
        help="When used with --batch-collect, skip tasks already marked Agent1Status='done'.",
    )
    ap.add_argument(
        "--batch-max-bytes",
        type=int,
        default=DEFAULT_BATCH_TARGET_BYTES,
        help=(
            "Target max JSONL bytes per submitted batch chunk "
            f"(default: {DEFAULT_BATCH_TARGET_BYTES}, hard max: {OPENAI_BATCH_INPUT_FILE_HARD_LIMIT_BYTES})."
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
        help=f"Backoff multiplier after failed batch in --batch-and-collect (default: {DEFAULT_ADAPTIVE_BATCH_BACKOFF_FACTOR}).",
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
            print("Error: no .agent1_last_batch_ids.json file. Run --batch first.")
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

    print(f"Pending tasks fetched: {len(tasks)} (workers={args.workers})")

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
            name=f"agent1-{i}",
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
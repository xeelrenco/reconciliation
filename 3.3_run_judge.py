#!/usr/bin/env python3
"""
Judge per riconciliazione MDR: legge le decisioni dei due agenti, emette il giudizio finale
e scrive in MdrReconciliationAgentDecisions e MdrReconciliationResults.

Config: config.txt (MOTHERDUCK_*, OPENAI_API_KEY, PROMPT_VERSION). Vedi config.example.txt.

Uso in tempo reale (default)
  Elabora i task ready_for_judge con N worker paralleli.
  python 3.3_run_judge.py [--prompt-version v1] [--embedding-model ...] [--limit N] [--workers 4] [--output-prompt-version ...] [--model ...]

Uso con Batch API (rate limit separati, ~50% costo)
  1) Submit: invia i task ready in un batch (elaborazione asincrona, entro 24h).
     python 3.3_run_judge.py --batch [--limit N]
     L'id del batch viene salvato in .judge_last_batch_id.
  2) Collect: quando il batch è completato, scarica i risultati e scrive in DB.
     python 3.3_run_judge.py --batch-collect [--batch-id <id>] [--output-prompt-version ...]

Stati JudgeStatus (batch): pending | submitted_{batch_id} | in_batch_{batch_id} | done | error_{batch_id}
  (solo 'done' senza suffisso; gli altri stati batch hanno _batch_id per tracciare il batch.)
"""

import json
import argparse
import queue
import threading
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
from openai import OpenAI

# -----------------------------
# Config da file (stesso formato degli altri script)
# -----------------------------
CONFIG_PATH = Path(__file__).resolve().parent / "config.txt"
BATCH_ID_FILE = Path(__file__).resolve().parent / ".judge_last_batch_id"
BATCH_ENDPOINT = "/v1/responses"


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


# --------------------------------------------------
# Config
# --------------------------------------------------
JUDGE_AGENT_NAME = "judge"
DEFAULT_MODEL = "gpt-5-mini"

DB_SCHEMA = "my_db.mdr_reconciliation"
TASKS_TABLE = f"{DB_SCHEMA}.MdrReconciliationTasks"
AGENT_DECISIONS_TABLE = f"{DB_SCHEMA}.MdrReconciliationAgentDecisions"
FINAL_RESULTS_TABLE = f"{DB_SCHEMA}.MdrReconciliationResults"
AGENT_INPUT_VIEW = f"{DB_SCHEMA}.v_MdrReconciliationAgentInput"
MDR_VIEW = "my_db.historical_mdr_normalization.v_MdrPreviousRecords_Normalized_All"

client = OpenAI(api_key=_cfg("OPENAI_API_KEY"))

# --------------------------------------------------
# Helpers
# --------------------------------------------------
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


# --------------------------------------------------
# Bootstrap final results table.
# Chiave (TaskId, PromptVersion, EmbeddingModel): la PromptVersion in output
# distingue le esecuzioni per test/comparazione; righe con versione diversa non
# si sovrascrivono. Se la tabella esisteva con PK solo su TaskId, ricrearla:
#   DROP TABLE my_db.mdr_reconciliation.MdrReconciliationResults;
# --------------------------------------------------
def ensure_final_results_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"""
    CREATE TABLE IF NOT EXISTS {FINAL_RESULTS_TABLE} (
      TaskId               VARCHAR NOT NULL,
      Document_title       VARCHAR NOT NULL,
      PromptVersion        VARCHAR NOT NULL,
      EmbeddingModel       VARCHAR NOT NULL,
      FinalTitleKey        VARCHAR,
      FinalRaciTitle       VARCHAR,
      FinalDecisionType    VARCHAR NOT NULL,   -- MATCH | NO_MATCH | MANUAL_REVIEW
      FinalConfidence      DOUBLE,
      ResolutionMode       VARCHAR NOT NULL,
      FinalReason          VARCHAR NOT NULL,
      CreatedAt            TIMESTAMP NOT NULL,
      UpdatedAt            TIMESTAMP NOT NULL,
      PRIMARY KEY (TaskId, PromptVersion, EmbeddingModel)
    );
    """)


# --------------------------------------------------
# Fetch tasks ready for judge
# --------------------------------------------------
def fetch_ready_tasks(
    con: duckdb.DuckDBPyConnection,
    prompt_version: str,
    embedding_model: str,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    params: List[Any] = [prompt_version, embedding_model]
    sql = f"""
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
        FROM {TASKS_TABLE}
        WHERE PromptVersion = ?
          AND EmbeddingModel = ?
          AND Agent1Status = 'done'
          AND Agent2Status = 'done'
          AND JudgeStatus = 'pending'
          AND FinalStatus = 'ready_for_judge'
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


def claim_task_judge(con: duckdb.DuckDBPyConnection, task_id: str) -> bool:
    ts = now_ts_naive_utc()
    con.execute("BEGIN;")
    try:
        rows = con.execute(f"""
            UPDATE {TASKS_TABLE}
            SET
              JudgeStatus = 'running',
              FinalStatus = CASE
                WHEN FinalStatus = 'ready_for_judge' THEN 'in_progress'
                ELSE FinalStatus
              END,
              UpdatedAt = ?
            WHERE TaskId = ?
              AND JudgeStatus = 'pending'
              AND Agent1Status = 'done'
              AND Agent2Status = 'done'
              AND FinalStatus = 'ready_for_judge'
            RETURNING TaskId
        """, [ts, task_id]).fetchall()
        con.execute("COMMIT;")
        return len(rows) == 1
    except Exception:
        con.execute("ROLLBACK;")
        raise


# --------------------------------------------------
# Candidate set + MDR context
# --------------------------------------------------
def fetch_candidates_for_task(
    con: duckdb.DuckDBPyConnection,
    document_title: str,
    prompt_version: str,
    embedding_model: str
) -> List[Dict[str, Any]]:
    rows = con.execute(f"""
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
        FROM {AGENT_INPUT_VIEW}
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
    row = con.execute(f"""
        SELECT
          Document_title,
          Discipline_Normalized,
          Discipline_Status,
          Type_L1,
          Type_L1_Status
        FROM {MDR_VIEW}
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


def fetch_agent_decisions(con: duckdb.DuckDBPyConnection, task_id: str) -> Dict[str, Dict[str, Any]]:
    rows = con.execute(f"""
        SELECT
          AgentName,
          AgentModel,
          SelectedTitleKey,
          SelectedRaciTitle,
          DecisionType,
          Confidence,
          ReasoningSummary
        FROM {AGENT_DECISIONS_TABLE}
        WHERE TaskId = ?
          AND AgentName IN ('gpt5mini', 'claude')
    """, [task_id]).fetchall()

    out = {}
    for r in rows:
        out[r[0]] = {
            "AgentName": r[0],
            "AgentModel": r[1],
            "SelectedTitleKey": r[2],
            "SelectedRaciTitle": r[3],
            "DecisionType": r[4],
            "Confidence": float(r[5]) if r[5] is not None else None,
            "ReasoningSummary": r[6],
        }
    return out


# --------------------------------------------------
# Judge prompt
# --------------------------------------------------
RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision_type": {
            "type": "string",
            "enum": ["MATCH", "NO_MATCH", "MANUAL_REVIEW"]
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
        "resolution_mode": {
            "type": "string",
            "enum": [
                "agent_consensus",
                "judge_override",
                "no_credible_candidate",
                "ambiguous_candidates"
            ]
        }
    },
    "required": [
        "decision_type",
        "selected_titlekey",
        "selected_raci_title",
        "confidence",
        "reasoning_summary",
        "resolution_mode"
    ]
}

SYSTEM_PROMPT = """
You are the final judge for EPC document title reconciliation.

You will receive:
- one historical MDR title with normalized metadata
- 50 candidate RACI documents
- the decisions of two independent agents

Your task:
Determine the final reconciliation outcome.

Allowed outcomes:
- MATCH
- NO_MATCH
- MANUAL_REVIEW

Definitions:

MATCH
Select MATCH only if one candidate is clearly the best semantic match.

NO_MATCH
Choose NO_MATCH when none of the candidates is sufficiently credible.

MANUAL_REVIEW
Use MANUAL_REVIEW only when genuine ambiguity exists between plausible candidates or when a plausible candidate cannot be confidently confirmed or rejected.

Core principles:
- Similarity scores are retrieval hints, not proof of equivalence.
- Prefer semantic equivalence over lexical overlap.
- Metadata compatibility is important supporting evidence.
- Strong metadata incompatibility is negative evidence.
- Do not overuse MANUAL_REVIEW.

Use agent decisions as evidence, not authority.

Decision logic:

1. If both agents selected the same candidate and the candidate is semantically consistent → MATCH is strongly supported.

2. If agents selected different candidates:
   - Choose MATCH only if one candidate is clearly superior.

3. If neither agent found a credible candidate and the candidates appear weak → choose NO_MATCH.

4. Choose MANUAL_REVIEW only when:
   - two or more candidates appear genuinely plausible, or
   - a candidate appears plausible but evidence is insufficient to safely accept or reject it.

Output JSON only with:
- decision_type
- selected_titlekey
- selected_raci_title
- confidence
- resolution_mode
- reasoning_summary

Decision logic:
- If both agents selected the same candidate and the candidate is semantically consistent, MATCH is strongly supported.
- If agents disagree, choose MATCH only if one candidate is clearly better supported.
- Choose NO_MATCH when the best available candidate is still weak, generic, metadata-incompatible, or not clearly equivalent.
- Choose MANUAL_REVIEW only when there is real ambiguity between plausible candidates or a plausible-but-inconclusive candidate that warrants human review.

Rules:
- confidence must be between 0 and 1
- selected_titlekey and selected_raci_title must be null for NO_MATCH and MANUAL_REVIEW
- reasoning_summary maximum 100 words
"""


def build_user_prompt(
    mdr_ctx: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    agent1: Dict[str, Any],
    agent2: Dict[str, Any]
) -> str:

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
    blocks.append("- Choose MATCH only if one candidate is clearly the best")
    blocks.append("- Choose NO_MATCH if the best candidate is still weak, generic, metadata-incompatible, or not clearly equivalent")
    blocks.append("- Do not use MANUAL_REVIEW as a default fallback")
    blocks.append("- Choose MANUAL_REVIEW only when there is genuine ambiguity between plausible candidates or a plausible-but-inconclusive candidate worth human review")
    blocks.append("")

    blocks.append("AGENT DECISIONS")

    blocks.append("Agent 1 (gpt5mini)")
    blocks.append(f"DecisionType: {norm(agent1.get('DecisionType'))}")
    blocks.append(f"SelectedTitleKey: {norm(agent1.get('SelectedTitleKey'))}")
    blocks.append(f"SelectedRaciTitle: {norm(agent1.get('SelectedRaciTitle'))}")
    blocks.append(f"Confidence: {agent1.get('Confidence')}")
    blocks.append(f"ReasoningSummary: {norm(agent1.get('ReasoningSummary'))}")
    blocks.append("")

    blocks.append("Agent 2 (claude)")
    blocks.append(f"DecisionType: {norm(agent2.get('DecisionType'))}")
    blocks.append(f"SelectedTitleKey: {norm(agent2.get('SelectedTitleKey'))}")
    blocks.append(f"SelectedRaciTitle: {norm(agent2.get('SelectedRaciTitle'))}")
    blocks.append(f"Confidence: {agent2.get('Confidence')}")
    blocks.append(f"ReasoningSummary: {norm(agent2.get('ReasoningSummary'))}")
    blocks.append("")

    blocks.append("CANDIDATES")

    for c in candidates:

        selected_by = []

        if str(c["TitleKey"]) == str(agent1.get("SelectedTitleKey")):
            selected_by.append("Agent1")

        if str(c["TitleKey"]) == str(agent2.get("SelectedTitleKey")):
            selected_by.append("Agent2")

        blocks.append("----")
        blocks.append(f"Rank: {c['Rank']}")
        blocks.append(f"SimilarityScore: {float(c['Similarity']):.4f}")
        blocks.append(f"SelectedByAgents: {', '.join(selected_by) if selected_by else 'None'}")
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


# --------------------------------------------------
# OpenAI judge call
# --------------------------------------------------
def call_judge(
    model: str,
    mdr_ctx: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    agent1: Dict[str, Any],
    agent2: Dict[str, Any]
) -> Dict[str, Any]:
    user_prompt = build_user_prompt(mdr_ctx, candidates, agent1, agent2)

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "mdr_reconciliation_judge",
                "schema": RESPONSE_SCHEMA,
                "strict": True,
            }
        },
    )

    return json.loads(resp.output_text)


# --------------------------------------------------
# Validation
# --------------------------------------------------
def validate_judge_output(result: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    candidate_map = {norm(c["TitleKey"]): c for c in candidates}

    decision_type = result["decision_type"]
    selected_titlekey = result["selected_titlekey"]
    selected_raci_title = result["selected_raci_title"]
    confidence = float(result["confidence"])
    reasoning_summary = norm(result["reasoning_summary"])
    resolution_mode = norm(result["resolution_mode"])

    if confidence < 0:
        confidence = 0.0
    if confidence > 1:
        confidence = 1.0

    if decision_type in ("NO_MATCH", "MANUAL_REVIEW"):
        return {
            "FinalDecisionType": decision_type,
            "FinalTitleKey": None,
            "FinalRaciTitle": None,
            "FinalConfidence": confidence,
            "FinalReason": reasoning_summary or "No final candidate selected.",
            "ResolutionMode": resolution_mode or ("no_credible_candidate" if decision_type == "NO_MATCH" else "ambiguous_candidates")
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

    return {
        "FinalDecisionType": "MATCH",
        "FinalTitleKey": selected_titlekey,
        "FinalRaciTitle": norm(selected_raci_title),
        "FinalConfidence": confidence,
        "FinalReason": reasoning_summary or "Judge selected the best supported candidate.",
        "ResolutionMode": resolution_mode or "judge_override"
    }


# --------------------------------------------------
# Save outputs
# --------------------------------------------------
def save_judge_result(
    con: duckdb.DuckDBPyConnection,
    task: Dict[str, Any],
    model: str,
    judge_result: Dict[str, Any],
    output_prompt_version: Optional[str] = None,
) -> None:
    ts = now_ts_naive_utc()
    prompt_version_for_save = (output_prompt_version and output_prompt_version.strip()) or task["PromptVersion"]

    con.execute("BEGIN;")
    try:
        # Save judge decision in agent decisions table
        con.execute(f"""
            INSERT INTO {AGENT_DECISIONS_TABLE}
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
            JUDGE_AGENT_NAME,
            model,
            task["Document_title"],
            prompt_version_for_save,
            task["EmbeddingModel"],
            judge_result["FinalTitleKey"],
            judge_result["FinalRaciTitle"],
            judge_result["FinalDecisionType"],
            judge_result["FinalConfidence"],
            judge_result["FinalReason"],
            ts
        ])

        # Save final result (PromptVersion in output = versione usata per distinguere i test; non si sovrascrivono righe con versione diversa)
        con.execute(f"""
            INSERT INTO {FINAL_RESULTS_TABLE}
              (TaskId, Document_title, PromptVersion, EmbeddingModel,
               FinalTitleKey, FinalRaciTitle, FinalDecisionType, FinalConfidence,
               ResolutionMode, FinalReason, CreatedAt, UpdatedAt)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (TaskId, PromptVersion, EmbeddingModel) DO UPDATE SET
              Document_title = excluded.Document_title,
              FinalTitleKey = excluded.FinalTitleKey,
              FinalRaciTitle = excluded.FinalRaciTitle,
              FinalDecisionType = excluded.FinalDecisionType,
              FinalConfidence = excluded.FinalConfidence,
              ResolutionMode = excluded.ResolutionMode,
              FinalReason = excluded.FinalReason,
              UpdatedAt = excluded.UpdatedAt
        """, [
            task["TaskId"],
            task["Document_title"],
            prompt_version_for_save,
            task["EmbeddingModel"],
            judge_result["FinalTitleKey"],
            judge_result["FinalRaciTitle"],
            judge_result["FinalDecisionType"],
            judge_result["FinalConfidence"],
            judge_result["ResolutionMode"],
            judge_result["FinalReason"],
            ts,
            ts
        ])

        # Update task workflow
        next_final_status = "completed"
        if judge_result["FinalDecisionType"] == "MANUAL_REVIEW":
            next_final_status = "manual_review"
        elif judge_result["FinalDecisionType"] == "NO_MATCH":
            next_final_status = "completed"

        con.execute(f"""
            UPDATE {TASKS_TABLE}
            SET
              JudgeStatus = 'done',
              FinalStatus = ?,
              UpdatedAt = ?
            WHERE TaskId = ?
        """, [next_final_status, ts, task["TaskId"]])

        con.execute("COMMIT;")
    except Exception:
        con.execute("ROLLBACK;")
        raise


def mark_judge_error(
    con: duckdb.DuckDBPyConnection,
    task_id: str,
    batch_id: Optional[str] = None,
) -> None:
    """Set JudgeStatus to 'error' or 'error_{batch_id}' and FinalStatus to 'error'."""
    ts = now_ts_naive_utc()
    status = f"error_{batch_id}" if batch_id else "error"
    con.execute(f"""
        UPDATE {TASKS_TABLE}
        SET
          JudgeStatus = ?,
          FinalStatus = 'error',
          UpdatedAt = ?
        WHERE TaskId = ?
    """, [status, ts, task_id])


def _judge_batch_request_body(model: str, user_prompt: str) -> Dict[str, Any]:
    """Body for one /v1/responses request in the judge batch JSONL."""
    return {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "mdr_reconciliation_judge",
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


def run_batch_submit(
    con: duckdb.DuckDBPyConnection,
    tasks: List[Dict[str, Any]],
    model: str,
) -> str:
    """Build JSONL, upload file, create batch, mark tasks JudgeStatus=submitted_{batch_id}. Returns batch id."""
    lines: List[str] = []
    submitted_task_ids: List[str] = []
    for task in tasks:
        candidates = fetch_candidates_for_task(
            con=con,
            document_title=task["Document_title"],
            prompt_version=task["PromptVersion"],
            embedding_model=task["EmbeddingModel"],
        )
        if not candidates:
            continue
        decisions = fetch_agent_decisions(con, task["TaskId"])
        agent1 = decisions.get("gpt5mini")
        agent2 = decisions.get("claude")
        if not agent1 or not agent2:
            continue
        mdr_ctx = fetch_mdr_context(con, task["Document_title"])
        user_prompt = build_user_prompt(mdr_ctx, candidates, agent1, agent2)
        body = _judge_batch_request_body(model, user_prompt)
        lines.append(json.dumps({
            "custom_id": task["TaskId"],
            "method": "POST",
            "url": BATCH_ENDPOINT,
            "body": body,
        }))
        submitted_task_ids.append(task["TaskId"])
    if not lines:
        raise RuntimeError("No valid tasks to submit (missing candidates or agent decisions?).")
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
        batch_id = batch.id
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    ts = now_ts_naive_utc()
    status_submitted = f"submitted_{batch_id}"
    for task_id in submitted_task_ids:
        con.execute(f"""
            UPDATE {TASKS_TABLE}
            SET JudgeStatus = ?,
                FinalStatus = CASE WHEN FinalStatus = 'ready_for_judge' THEN 'in_progress' ELSE FinalStatus END,
                UpdatedAt = ?
            WHERE TaskId = ?
        """, [status_submitted, ts, task_id])
    BATCH_ID_FILE.write_text(batch_id, encoding="utf-8")
    return batch_id


def run_batch_collect(
    con: duckdb.DuckDBPyConnection,
    batch_id: str,
    model: str,
    output_prompt_version: Optional[str] = None,
    poll_interval: int = 60,
) -> None:
    """Poll batch until completed, then download results and write each to DB."""
    while True:
        batch = client.batches.retrieve(batch_id)
        status = getattr(batch, "status", None) or (batch.get("status") if isinstance(batch, dict) else None)
        if status == "completed":
            break
        if status in ("failed", "canceled", "expired"):
            print(f"Batch {batch_id} ended with status={status}. Cannot collect results.")
            return
        print(f"Batch {batch_id} status={status}, waiting {poll_interval}s...")
        time.sleep(poll_interval)
    status_submitted = f"submitted_{batch_id}"
    status_in_batch = f"in_batch_{batch_id}"
    ts = now_ts_naive_utc()
    con.execute(f"""
        UPDATE {TASKS_TABLE}
        SET JudgeStatus = ?, UpdatedAt = ?
        WHERE JudgeStatus = ?
    """, [status_in_batch, ts, status_submitted])
    saved = 0
    errors = 0
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
            output_text = _extract_output_text_from_batch_response_body(body) if body else None
            if not output_text:
                mark_judge_error(con, task_id, batch_id=batch_id)
                errors += 1
                print(f"  {task_id}: no output text in response")
                continue
            try:
                raw_result = json.loads(output_text)
            except json.JSONDecodeError as e:
                mark_judge_error(con, task_id, batch_id=batch_id)
                errors += 1
                print(f"  {task_id}: JSON parse error {e}")
                continue
            db_row = con.execute(f"""
                SELECT TaskId, Document_title, PromptVersion, EmbeddingModel
                FROM {TASKS_TABLE}
                WHERE TaskId = ?
            """, [task_id]).fetchone()
            if not db_row:
                errors += 1
                continue
            task = {"TaskId": db_row[0], "Document_title": db_row[1], "PromptVersion": db_row[2], "EmbeddingModel": db_row[3]}
            candidates = fetch_candidates_for_task(con, task["Document_title"], task["PromptVersion"], task["EmbeddingModel"])
            if not candidates:
                mark_judge_error(con, task_id, batch_id=batch_id)
                errors += 1
                continue
            try:
                validated = validate_judge_output(raw_result, candidates)
                save_judge_result(
                    con=con,
                    task=task,
                    model=model,
                    judge_result=validated,
                    output_prompt_version=output_prompt_version,
                )
                saved += 1
                print(f"  {task_id} -> {validated['FinalDecisionType']} (saved)")
            except Exception as e:
                mark_judge_error(con, task_id, batch_id=batch_id)
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
                mark_judge_error(con, str(custom_id), batch_id=batch_id)
                errors += 1
                print(f"  {custom_id}: batch error (see error file)")
    print(f"Batch collect done: {saved} saved, {errors} errors.")


def process_one_judge_task(
    con: duckdb.DuckDBPyConnection,
    task: Dict[str, Any],
    model: str,
    output_prompt_version: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Run judge pipeline for one task (caller must have claimed it). Returns validated_result on success, None on skip/error."""
    candidates = fetch_candidates_for_task(
        con=con,
        document_title=task["Document_title"],
        prompt_version=task["PromptVersion"],
        embedding_model=task["EmbeddingModel"]
    )
    if not candidates:
        mark_judge_error(con, task["TaskId"])
        return None

    decisions = fetch_agent_decisions(con, task["TaskId"])
    agent1 = decisions.get("gpt5mini")
    agent2 = decisions.get("claude")
    if not agent1 or not agent2:
        mark_judge_error(con, task["TaskId"])
        return None

    mdr_ctx = fetch_mdr_context(con, task["Document_title"])
    raw_result = call_judge(
        model=model,
        mdr_ctx=mdr_ctx,
        candidates=candidates,
        agent1=agent1,
        agent2=agent2
    )
    validated_result = validate_judge_output(raw_result, candidates)
    save_judge_result(con=con, task=task, model=model, judge_result=validated_result, output_prompt_version=output_prompt_version)
    return validated_result


def _worker(
    task_queue: queue.Queue,
    model: str,
    print_lock: threading.Lock,
    total_tasks: int,
    completed_count: List[int],
    output_prompt_version: Optional[str] = None,
) -> None:
    con = connect_motherduck()
    try:
        while True:
            task = task_queue.get()
            processed = False
            try:
                if task is None:
                    return
                claimed = claim_task_judge(con, task["TaskId"])
                if not claimed:
                    continue
                processed = True
                with print_lock:
                    print(f"[{threading.current_thread().name}] Processing: {task['Document_title']}")
                result = process_one_judge_task(con, task, model, output_prompt_version=output_prompt_version)
                if result:
                    with print_lock:
                        print(
                            f"  Saved -> final_decision={result['FinalDecisionType']} "
                            f"titlekey={result['FinalTitleKey']} "
                            f"confidence={result['FinalConfidence']:.3f} "
                            f"mode={result['ResolutionMode']}"
                        )
            except Exception as e:
                mark_judge_error(con, task["TaskId"])
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


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-version", default=None, help="PromptVersion (default: from config.txt PROMPT_VERSION).")
    ap.add_argument("--embedding-model", default=None, help="EmbeddingModel (default: from config or text-embedding-3-small).")
    ap.add_argument("--limit", type=int, default=None, help="Max number of tasks to process (default: no limit, process all ready).")
    ap.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4).")
    ap.add_argument("--output-prompt-version", default=None, help="PromptVersion to write in output (default: same as --prompt-version). Set in config as JUDGE_OUTPUT_PROMPT_VERSION.")
    ap.add_argument("--model", default=None, help="OpenAI model for judge (default: gpt-5-mini).")
    ap.add_argument("--batch", action="store_true", help="Use Batch API: submit ready tasks and exit (collect later with --batch-collect).")
    ap.add_argument("--batch-collect", action="store_true", help="Poll batch until completed and write results to DB. Use --batch-id or .judge_last_batch_id.")
    ap.add_argument("--batch-id", default=None, help="Batch ID for --batch-collect (default: read from .judge_last_batch_id).")
    args = ap.parse_args()

    args.prompt_version = args.prompt_version or _cfg("PROMPT_VERSION", "v1")
    args.embedding_model = args.embedding_model or "text-embedding-3-small"
    args.output_prompt_version = (args.output_prompt_version or _cfg("JUDGE_OUTPUT_PROMPT_VERSION") or "").strip() or None
    args.model = args.model or DEFAULT_MODEL

    con = connect_motherduck()
    ensure_final_results_table(con)

    if args.batch_collect:
        batch_id = (args.batch_id or "").strip() or (BATCH_ID_FILE.read_text(encoding="utf-8").strip() if BATCH_ID_FILE.exists() else "")
        if not batch_id:
            print("Error: no --batch-id and no .judge_last_batch_id file.")
            con.close()
            return
        print(f"Collecting results for batch: {batch_id}")
        run_batch_collect(
            con=con,
            batch_id=batch_id,
            model=args.model,
            output_prompt_version=args.output_prompt_version,
        )
        con.close()
        return

    if args.batch:
        tasks = fetch_ready_tasks(
            con=con,
            prompt_version=args.prompt_version,
            embedding_model=args.embedding_model,
            limit=args.limit,
        )
        if not tasks:
            print("No ready tasks for batch.")
            con.close()
            return
        batch_id = run_batch_submit(con=con, tasks=tasks, model=args.model)
        con.close()
        print(f"Batch submitted: {batch_id}")
        print("Run with --batch-collect later to write results to DB (or use --batch-id if you save it).")
        return

    tasks = fetch_ready_tasks(
        con=con,
        prompt_version=args.prompt_version,
        embedding_model=args.embedding_model,
        limit=args.limit
    )
    con.close()

    if args.output_prompt_version:
        print(f"Ready tasks fetched for judge: {len(tasks)} (workers={args.workers}, output PromptVersion={args.output_prompt_version})")
    else:
        print(f"Ready tasks fetched for judge: {len(tasks)} (workers={args.workers})")

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
            args=(task_queue, args.model, print_lock, total_tasks, completed_count, args.output_prompt_version),
            name=f"judge-{i}",
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
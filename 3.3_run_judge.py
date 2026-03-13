#!/usr/bin/env python3
"""
Judge per riconciliazione MDR: legge le decisioni dei due agenti, emette il giudizio finale
e scrive in MdrReconciliationAgentDecisions e MdrReconciliationResults.

Config: config.txt (MOTHERDUCK_*, PROMPT_VERSION, VERTEX_*). Vedi config.example.txt.

Uso in tempo reale (default)
  Elabora i task ready_for_judge con N worker paralleli (Vertex AI / Gemini).
  python 3.3_run_judge.py [--prompt-version v1] [--embedding-model ...] [--limit N] [--workers 4] [--output-prompt-version ...] [--model ...]

Uso con Batch Vertex AI (Gemini batch inference, ~50% costo, GCS obbligatorio)
  Config: VERTEX_BATCH_GCS_BUCKET (e opzionale VERTEX_BATCH_GCS_PREFIX).
  1) Submit: carica il JSONL su GCS e crea il job di batch (elaborazione asincrona, entro 24h).
     python 3.3_run_judge.py --batch [--limit N]
     Job name in .judge_last_batch_id; dettagli in .judge_last_batch_info.json.
  2) Collect: quando il job è completato, scarica l'output da GCS e scrive in DB.
     python 3.3_run_judge.py --batch-collect [--batch-id <job_name>] [--output-prompt-version ...]
     Usa .judge_last_batch_info.json per output_prefix e task_ids (necessario per collect).

Stati JudgeStatus (batch): pending | submitted_{batch_id} | in_batch_{batch_id} | done | error_{batch_id}
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
import os

import duckdb
from google import genai
from google.genai.types import CreateBatchJobConfig, JobState

try:
    from google.cloud import storage as gcs_storage
except ImportError:
    gcs_storage = None  # pip install google-cloud-storage

# -----------------------------
# Config da file (stesso formato degli altri script)
# -----------------------------
CONFIG_PATH = Path(__file__).resolve().parent / "config.txt"
BATCH_ID_FILE = Path(__file__).resolve().parent / ".judge_last_batch_id"
BATCH_INFO_FILE = Path(__file__).resolve().parent / ".judge_last_batch_info.json"


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
DEFAULT_MODEL = "gemini-2.5-pro"

DB_SCHEMA = "my_db.mdr_reconciliation"
TASKS_TABLE = f"{DB_SCHEMA}.MdrReconciliationTasks"
AGENT_DECISIONS_TABLE = f"{DB_SCHEMA}.MdrReconciliationAgentDecisions"
FINAL_RESULTS_TABLE = f"{DB_SCHEMA}.MdrReconciliationResults"
AGENT_INPUT_VIEW = f"{DB_SCHEMA}.v_MdrReconciliationAgentInput"
MDR_VIEW = "my_db.historical_mdr_normalization.v_MdrPreviousRecords_Normalized_All"

# Vertex AI / Gemini (google-genai SDK); credentials da config.txt
_creds_rel = _cfg("VERTEX_CREDENTIALS_PATH")
if not _creds_rel:
    raise RuntimeError("Manca VERTEX_CREDENTIALS_PATH nel file di configurazione (config.txt)")
_creds_path = (Path(__file__).resolve().parent / _creds_rel).as_posix()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _creds_path

_genai_client = genai.Client(
    vertexai=True,
    project=_cfg("VERTEX_PROJECT_ID"),
    location=_cfg("VERTEX_LOCATION", "europe-west1"),
)

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
# Judge LLM call (Vertex AI / Gemini)
# --------------------------------------------------
def _extract_json_payload(raw_text: str) -> str:
    """
    Pulisce il testo restituendo solo il JSON (gestisce eventuali code fence o testo extra).
    """
    text = raw_text.strip()

    # Caso 1: fenced JSON ```json ... ```
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Caso 2: testo extra prima/dopo il JSON
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return text


def call_judge(
    model: str,
    mdr_ctx: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    agent1: Dict[str, Any],
    agent2: Dict[str, Any]
) -> Dict[str, Any]:
    user_prompt = build_user_prompt(mdr_ctx, candidates, agent1, agent2)

    # Combina SYSTEM_PROMPT e user_prompt in un unico prompt testuale
    full_prompt = f"{SYSTEM_PROMPT.strip()}\n\nUSER INPUT:\n{user_prompt}"

    response = _genai_client.models.generate_content(
        model=model,
        contents=full_prompt,
    )

    raw_text = getattr(response, "text", None) or ""
    cleaned = _extract_json_payload(raw_text)
    return json.loads(cleaned)


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


def _vertex_batch_request_line(full_prompt: str) -> Dict[str, Any]:
    """One line (request dict) for Vertex batch JSONL: contents + generationConfig."""
    return {
        "request": {
            "contents": [{"role": "user", "parts": [{"text": full_prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 2048},
        }
    }


def _extract_text_from_vertex_batch_response(line_obj: Dict[str, Any]) -> Optional[str]:
    """Extract model output text from Vertex batch output line (response.candidates[0].content.parts[0].text)."""
    if not isinstance(line_obj, dict):
        return None
    resp = line_obj.get("response") or {}
    candidates = resp.get("candidates") or []
    if not candidates:
        return None
    content = (candidates[0] or {}).get("content") or {}
    parts = content.get("parts") or []
    if not parts:
        return None
    return (parts[0] or {}).get("text")


def _gcs_upload_jsonl(bucket_name: str, blob_path: str, lines: List[str], project_id: str) -> None:
    """Upload JSONL lines to GCS (requires google-cloud-storage)."""
    if gcs_storage is None:
        raise RuntimeError("Batch submit richiede google-cloud-storage: pip install google-cloud-storage")
    client = gcs_storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    content = "\n".join(lines) + "\n" if lines else ""
    blob.upload_from_string(content, content_type="application/jsonl")


def _gcs_download_jsonl_lines(bucket_name: str, prefix: str, project_id: str) -> List[Dict[str, Any]]:
    """List blobs under prefix, download and parse as JSONL; return list of parsed lines in order."""
    if gcs_storage is None:
        raise RuntimeError("Batch collect richiede google-cloud-storage: pip install google-cloud-storage")
    client = gcs_storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    blobs.sort(key=lambda b: b.name)
    lines: List[Dict[str, Any]] = []
    for blob in blobs:
        content = blob.download_as_text(encoding="utf-8")
        for line in content.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return lines


def _gcs_delete_batch_artifacts(
    bucket_name: str,
    input_blob: Optional[str],
    output_prefix: str,
    project_id: str,
) -> None:
    """Delete input JSONL blob and all blobs under output prefix (cleanup after collect)."""
    if gcs_storage is None:
        return
    client = gcs_storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    deleted = 0
    if input_blob:
        try:
            bucket.blob(input_blob).delete()
            deleted += 1
        except Exception as e:
            print(f"  (cleanup) input blob delete skip: {e}")
    for blob in bucket.list_blobs(prefix=output_prefix):
        try:
            blob.delete()
            deleted += 1
        except Exception as e:
            print(f"  (cleanup) output blob delete skip: {e}")
    if deleted:
        print(f"  Bucket cleanup: {deleted} object(s) removed.")


def run_batch_submit(
    con: duckdb.DuckDBPyConnection,
    tasks: List[Dict[str, Any]],
    model: str,
) -> str:
    """Build Vertex-format JSONL, upload to GCS, create Vertex batch job, mark tasks submitted_{batch_id}. Returns job name."""
    bucket = _cfg("VERTEX_BATCH_GCS_BUCKET")
    if not bucket:
        raise RuntimeError("Per il batch Vertex serve VERTEX_BATCH_GCS_BUCKET in config.txt")
    prefix = (_cfg("VERTEX_BATCH_GCS_PREFIX") or "").strip().rstrip("/")
    if prefix:
        prefix = prefix + "/"
    project_id = _cfg("VERTEX_PROJECT_ID")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

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
        full_prompt = f"{SYSTEM_PROMPT.strip()}\n\nUSER INPUT:\n{user_prompt}"
        line_obj = _vertex_batch_request_line(full_prompt)
        lines.append(json.dumps(line_obj))
        submitted_task_ids.append(task["TaskId"])
    if not lines:
        raise RuntimeError("No valid tasks to submit (missing candidates or agent decisions?).")

    input_blob = f"{prefix}in/judge_{run_id}.jsonl"
    gcs_input_uri = f"gs://{bucket}/{input_blob}"
    output_prefix = f"{prefix}out/run_{run_id}/"
    gcs_output_uri = f"gs://{bucket}/{output_prefix}"

    _gcs_upload_jsonl(bucket, input_blob, lines, project_id)

    job = _genai_client.batches.create(
        model=model,
        src=gcs_input_uri,
        config=CreateBatchJobConfig(dest=gcs_output_uri),
    )
    job_name = getattr(job, "name", None) or (job.get("name") if isinstance(job, dict) else "")
    batch_id_short = job_name.split("/")[-1] if job_name and "/" in job_name else (job_name or run_id)

    ts = now_ts_naive_utc()
    status_submitted = f"submitted_{batch_id_short}"
    for task_id in submitted_task_ids:
        con.execute(f"""
            UPDATE {TASKS_TABLE}
            SET JudgeStatus = ?,
                FinalStatus = CASE WHEN FinalStatus = 'ready_for_judge' THEN 'in_progress' ELSE FinalStatus END,
                UpdatedAt = ?
            WHERE TaskId = ?
        """, [status_submitted, ts, task_id])

    BATCH_ID_FILE.write_text(job_name, encoding="utf-8")
    BATCH_INFO_FILE.write_text(
        json.dumps({
            "job_name": job_name,
            "output_prefix": gcs_output_uri,
            "input_blob": input_blob,
            "task_ids": submitted_task_ids,
        }, indent=2),
        encoding="utf-8",
    )
    return job_name


def _parse_gcs_uri(gcs_uri: str) -> tuple:
    """Return (bucket_name, prefix) from gs://bucket/prefix/..."""
    if not gcs_uri.startswith("gs://"):
        return "", ""
    path = gcs_uri[5:].strip("/")  # drop gs:// and trailing slash
    if "/" not in path:
        return path, ""
    bucket_name, _, prefix = path.partition("/")
    return bucket_name, (prefix + "/" if prefix else "")


def run_batch_collect(
    con: duckdb.DuckDBPyConnection,
    batch_id: str,
    model: str,
    output_prompt_version: Optional[str] = None,
    poll_interval: int = 60,
) -> None:
    """Poll Vertex batch job until completed, download output from GCS, write results to DB."""
    if not BATCH_INFO_FILE.exists():
        print("Error: .judge_last_batch_info.json non trovato (necessario per output_prefix e task_ids).")
        return
    info = json.loads(BATCH_INFO_FILE.read_text(encoding="utf-8"))
    job_name = (batch_id or "").strip() or info.get("job_name") or ""
    output_prefix_uri = info.get("output_prefix") or ""
    input_blob = info.get("input_blob")  # optional, for cleanup (older runs may not have it)
    task_ids = info.get("task_ids") or []
    if not job_name:
        print("Error: job_name non disponibile (--batch-id o .judge_last_batch_info.json).")
        return
    if not output_prefix_uri or not task_ids:
        print("Error: output_prefix o task_ids mancanti in .judge_last_batch_info.json.")
        return

    batch_id_short = job_name.split("/")[-1] if "/" in job_name else job_name

    _succeeded = getattr(JobState, "JOB_STATE_SUCCEEDED", "JOB_STATE_SUCCEEDED")
    _terminal_fail = (
        getattr(JobState, "JOB_STATE_FAILED", "JOB_STATE_FAILED"),
        getattr(JobState, "JOB_STATE_CANCELLED", "JOB_STATE_CANCELLED"),
        getattr(JobState, "JOB_STATE_PAUSED", "JOB_STATE_PAUSED"),
    )
    while True:
        job = _genai_client.batches.get(name=job_name)
        state = getattr(job, "state", None) or (job.get("state") if isinstance(job, dict) else None)
        if state == _succeeded or (isinstance(state, str) and state == "JOB_STATE_SUCCEEDED"):
            break
        if state in _terminal_fail or (isinstance(state, str) and state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_PAUSED")):
            print(f"Batch {job_name} ended with state={state}. Cannot collect results.")
            return
        print(f"Batch {job_name} state={state}, waiting {poll_interval}s...")
        time.sleep(poll_interval)

    status_submitted = f"submitted_{batch_id_short}"
    status_in_batch = f"in_batch_{batch_id_short}"
    ts = now_ts_naive_utc()
    con.execute(f"""
        UPDATE {TASKS_TABLE}
        SET JudgeStatus = ?, UpdatedAt = ?
        WHERE JudgeStatus = ?
    """, [status_in_batch, ts, status_submitted])

    bucket_name, prefix = _parse_gcs_uri(output_prefix_uri)
    if not bucket_name:
        print("Error: output_prefix non valido (gs://bucket/prefix).")
        return
    project_id = _cfg("VERTEX_PROJECT_ID")
    output_lines = _gcs_download_jsonl_lines(bucket_name, prefix, project_id)

    saved = 0
    errors = 0
    for i, line_obj in enumerate(output_lines):
        task_id = task_ids[i] if i < len(task_ids) else None
        if not task_id:
            errors += 1
            continue
        if line_obj.get("status"):
            mark_judge_error(con, task_id, batch_id=batch_id_short)
            errors += 1
            print(f"  {task_id}: batch row error (status)")
            continue
        output_text = _extract_text_from_vertex_batch_response(line_obj)
        if not output_text:
            mark_judge_error(con, task_id, batch_id=batch_id_short)
            errors += 1
            print(f"  {task_id}: no output text in response")
            continue
        cleaned = _extract_json_payload(output_text)
        try:
            raw_result = json.loads(cleaned)
        except json.JSONDecodeError as e:
            mark_judge_error(con, task_id, batch_id=batch_id_short)
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
            mark_judge_error(con, task_id, batch_id=batch_id_short)
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
            mark_judge_error(con, task_id, batch_id=batch_id_short)
            errors += 1
            print(f"  {task_id}: validation/save error {e}")

    print(f"Batch collect done: {saved} saved, {errors} errors.")

    # Pulizia bucket: rimuovi input JSONL e output di questo run
    try:
        _gcs_delete_batch_artifacts(bucket_name, input_blob, prefix, project_id)
    except Exception as e:
        print(f"  Bucket cleanup warning: {e}")


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
    ap.add_argument("--model", default=None, help="Vertex AI / Gemini model for judge (default: gemini-2.5-pro).")
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
        job_name = (args.batch_id or "").strip() or (BATCH_ID_FILE.read_text(encoding="utf-8").strip() if BATCH_ID_FILE.exists() else "")
        if not job_name:
            print("Error: no --batch-id and no .judge_last_batch_id file.")
            con.close()
            return
        print(f"Collecting results for batch: {job_name}")
        run_batch_collect(
            con=con,
            batch_id=job_name,
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
        job_name = run_batch_submit(con=con, tasks=tasks, model=args.model)
        con.close()
        print(f"Batch submitted: {job_name}")
        print("Run with --batch-collect later to write results to DB (usa .judge_last_batch_info.json).")
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
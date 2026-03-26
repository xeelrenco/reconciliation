#!/usr/bin/env python3
"""
Judge per riconciliazione MDR: unico scrittore di MdrReconciliationResults.

Logica:
- Legge decisioni e top3 da MdrReconciliationAgentDecisions e MdrReconciliationAgentTopCandidates.
- Legge top50 retrieval dalla view arricchita (MdrToRaciCandidates).
- CONSENSO (stesso MATCH o entrambi NO_MATCH): risolve deterministicamente, NON chiama Gemini.
- CONFLITTO (MATCH vs MATCH diverso, o MATCH vs NO_MATCH): chiama Gemini e scrive risultato.
- Scrive sempre il risultato in MdrReconciliationResults (ResolvedBy='judge_script').
- Scrive in MdrReconciliationAgentDecisions solo quando Gemini viene chiamato (AgentName='judge_gemini').

Config: config.txt (MOTHERDUCK_*, PROMPT_VERSION, VERTEX_*). Vedi config.example.txt.

Uso in tempo reale: python 3.3_run_judge.py [--prompt-version v1] [--embedding-model ...] [--limit N] [--workers 4] [--model ...]
Uso batch (solo task in conflitto): python 3.3_run_judge.py --batch [--limit N] poi --batch-collect
"""

import json
import argparse
import queue
import threading
import tempfile
import time
import random
import re
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
TEST_TASK_FILE = Path(__file__).resolve().parent / ".recon_test_tasks.json"


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


# --------------------------------------------------
# Config
# --------------------------------------------------
DEFAULT_MODEL = "gemini-2.5-pro"

DB_SCHEMA = "my_db.mdr_reconciliation"
TASKS_TABLE = f"{DB_SCHEMA}.MdrReconciliationTasks"
AGENT_DECISIONS_TABLE = f"{DB_SCHEMA}.MdrReconciliationAgentDecisions"
AGENT_TOP_CANDIDATES_TABLE = f"{DB_SCHEMA}.MdrReconciliationAgentTopCandidates"
FINAL_RESULTS_TABLE = f"{DB_SCHEMA}.MdrReconciliationResults"
AGENT_INPUT_VIEW = f"{DB_SCHEMA}.v_MdrReconciliationAgentInput"
CANDIDATES_TABLE = f"{DB_SCHEMA}.MdrToRaciCandidates"
MDR_VIEW = "my_db.historical_mdr_normalization.v_MdrPreviousRecords_Normalized_All"

AGENT1_NAME = "gpt5mini"
AGENT2_NAME = "claude"
JUDGE_AGENT_NAME_GEMINI = "judge_gemini"

# Resolution modes (consensus = no Gemini; conflict = Gemini used)
RESOLUTION_CONSENSUS_MATCH = "judge_script_consensus_match"
RESOLUTION_CONSENSUS_NO_MATCH = "judge_script_consensus_no_match"
RESOLUTION_LLM_MATCH_MATCH = "judge_llm_match_match_conflict"
RESOLUTION_LLM_MATCH_NO_MATCH = "judge_llm_match_no_match_conflict"

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
# Bootstrap tables
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
      FinalDecisionType    VARCHAR NOT NULL,
      FinalConfidence      DOUBLE,
      ResolutionMode       VARCHAR NOT NULL,
      FinalReason          VARCHAR NOT NULL,
      ResolvedBy           VARCHAR NOT NULL DEFAULT 'judge_script',
      JudgeUsedFlag        BOOLEAN NOT NULL DEFAULT FALSE,
      JudgeModel           VARCHAR,
      CreatedAt            TIMESTAMP NOT NULL,
      UpdatedAt            TIMESTAMP NOT NULL,
      PRIMARY KEY (TaskId, PromptVersion, EmbeddingModel)
    );
    """)
    _add_column_if_missing(con, FINAL_RESULTS_TABLE, "ResolvedBy", "VARCHAR")
    _add_column_if_missing(con, FINAL_RESULTS_TABLE, "JudgeUsedFlag", "BOOLEAN")
    _add_column_if_missing(con, FINAL_RESULTS_TABLE, "JudgeModel", "VARCHAR")


def _add_column_if_missing(con: duckdb.DuckDBPyConnection, table: str, column: str, col_type: str) -> None:
    try:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
    except Exception:
        pass  # column already exists


def ensure_agent_top_candidates_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"""
    CREATE TABLE IF NOT EXISTS {AGENT_TOP_CANDIDATES_TABLE} (
      TaskId                    VARCHAR NOT NULL,
      AgentName                 VARCHAR NOT NULL,
      PromptVersion             VARCHAR NOT NULL,
      ModelName                 VARCHAR,
      CandidateRankWithinAgent   INTEGER NOT NULL,
      TitleKey                  VARCHAR NOT NULL,
      RaciTitle                 VARCHAR,
      CandidateConfidence       DOUBLE,
      WhyPlausible              VARCHAR,
      CreatedAt                 TIMESTAMP NOT NULL,
      PRIMARY KEY (TaskId, AgentName, CandidateRankWithinAgent)
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
    embedding_model: str,
) -> List[Dict[str, Any]]:
    """Top50 retrieval candidates (enriched). Delegates to load_retrieval_candidates."""
    return load_retrieval_candidates(con, document_title, prompt_version, embedding_model, limit=50)


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
    """Return both agent decisions keyed by agent name (gpt5mini, claude)."""
    a1 = load_agent_decision(con, task_id, AGENT1_NAME)
    a2 = load_agent_decision(con, task_id, AGENT2_NAME)
    out = {}
    if a1:
        out[AGENT1_NAME] = a1
    if a2:
        out[AGENT2_NAME] = a2
    return out


def load_agent_decision(
    con: duckdb.DuckDBPyConnection,
    task_id: str,
    agent_name: str,
) -> Optional[Dict[str, Any]]:
    """Load one agent decision from MdrReconciliationAgentDecisions. Returns None if missing."""
    row = con.execute(f"""
        SELECT
          AgentName,
          AgentModel,
          SelectedTitleKey,
          SelectedRaciTitle,
          DecisionType,
          Confidence,
          ReasoningSummary
        FROM {AGENT_DECISIONS_TABLE}
        WHERE TaskId = ? AND AgentName = ?
    """, [task_id, agent_name]).fetchone()
    if not row:
        return None
    return {
        "AgentName": row[0],
        "AgentModel": row[1],
        "SelectedTitleKey": row[2],
        "SelectedRaciTitle": row[3],
        "DecisionType": row[4],
        "Confidence": float(row[5]) if row[5] is not None else None,
        "ReasoningSummary": row[6],
    }


def load_agent_top_candidates(
    con: duckdb.DuckDBPyConnection,
    task_id: str,
    agent_name: str,
) -> List[Dict[str, Any]]:
    """Load top 3 candidates for one agent from MdrReconciliationAgentTopCandidates. Returns [] if missing."""
    try:
        rows = con.execute(f"""
            SELECT
              TaskId,
              AgentName,
              PromptVersion,
              ModelName,
              CandidateRankWithinAgent,
              TitleKey,
              RaciTitle,
              CandidateConfidence,
              WhyPlausible,
              CreatedAt
            FROM {AGENT_TOP_CANDIDATES_TABLE}
            WHERE TaskId = ? AND AgentName = ?
            ORDER BY CandidateRankWithinAgent
            LIMIT 3
        """, [task_id, agent_name]).fetchall()
    except Exception:
        return []
    cols = [
        "TaskId", "AgentName", "PromptVersion", "ModelName",
        "CandidateRankWithinAgent", "TitleKey", "RaciTitle",
        "CandidateConfidence", "WhyPlausible", "CreatedAt"
    ]
    return [dict(zip(cols, r)) for r in rows]


def load_retrieval_candidates(
    con: duckdb.DuckDBPyConnection,
    document_title: str,
    prompt_version: str,
    embedding_model: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Load top50 retrieval candidates (enriched) from view. Same as fetch_candidates_for_task with limit."""
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
        LIMIT ?
    """, [document_title, prompt_version, embedding_model, limit]).fetchall()
    cols = [
        "Rank", "Similarity", "TitleKey", "RaciTitle", "EffectiveDescription",
        "DisciplineName", "TypeName", "CategoryDescription", "ChapterName"
    ]
    return [dict(zip(cols, r)) for r in rows]


# --------------------------------------------------
# Resolution case classification and consensus
# --------------------------------------------------
CASE_CONSENSUS_MATCH = "consensus_match"
CASE_CONSENSUS_NO_MATCH = "consensus_no_match"
CASE_MATCH_MATCH_CONFLICT = "match_match_conflict"
CASE_MATCH_NO_MATCH_CONFLICT = "match_no_match_conflict"
CASE_MISSING_DATA = "missing_data"


def classify_resolution_case(
    agent1: Optional[Dict[str, Any]],
    agent2: Optional[Dict[str, Any]],
) -> str:
    """
    Classify task into one of four resolution cases or missing_data.
    Returns: consensus_match | consensus_no_match | match_match_conflict | match_no_match_conflict | missing_data
    """
    if not agent1 or not agent2:
        return CASE_MISSING_DATA
    dt1 = (agent1.get("DecisionType") or "").strip().upper()
    dt2 = (agent2.get("DecisionType") or "").strip().upper()
    if dt1 not in ("MATCH", "NO_MATCH") or dt2 not in ("MATCH", "NO_MATCH"):
        return CASE_MISSING_DATA
    if dt1 == "NO_MATCH" and dt2 == "NO_MATCH":
        return CASE_CONSENSUS_NO_MATCH
    if dt1 == "MATCH" and dt2 == "MATCH":
        key1 = norm(agent1.get("SelectedTitleKey"))
        key2 = norm(agent2.get("SelectedTitleKey"))
        if key1 and key2 and key1 == key2:
            return CASE_CONSENSUS_MATCH
        return CASE_MATCH_MATCH_CONFLICT
    return CASE_MATCH_NO_MATCH_CONFLICT


def resolve_consensus_case(
    case: str,
    agent1: Dict[str, Any],
    agent2: Dict[str, Any],
    task: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build result dict for consensus (no Gemini). Used by write_final_result.
    case must be CASE_CONSENSUS_MATCH or CASE_CONSENSUS_NO_MATCH.
    """
    if case == CASE_CONSENSUS_NO_MATCH:
        return {
            "FinalDecisionType": "NO_MATCH",
            "FinalTitleKey": None,
            "FinalRaciTitle": None,
            "FinalConfidence": 0.0,
            "FinalReason": "Both agents agreed: no credible match.",
            "ResolutionMode": RESOLUTION_CONSENSUS_NO_MATCH,
            "ResolvedBy": "judge_script",
            "JudgeUsedFlag": False,
            "JudgeModel": None,
        }
    if case == CASE_CONSENSUS_MATCH:
        key = norm(agent1.get("SelectedTitleKey"))
        title = norm(agent1.get("SelectedRaciTitle") or agent2.get("SelectedRaciTitle"))
        c1 = agent1.get("Confidence")
        c2 = agent2.get("Confidence")
        conf = (float(c1) + float(c2)) / 2.0 if c1 is not None and c2 is not None else 0.9
        return {
            "FinalDecisionType": "MATCH",
            "FinalTitleKey": key,
            "FinalRaciTitle": title or "",
            "FinalConfidence": conf,
            "FinalReason": "Both agents agreed on the same match.",
            "ResolutionMode": RESOLUTION_CONSENSUS_MATCH,
            "ResolvedBy": "judge_script",
            "JudgeUsedFlag": False,
            "JudgeModel": None,
        }
    raise ValueError(f"resolve_consensus_case called with case={case}")


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
        "selected_candidate_id": {
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
                "match_match_conflict_resolved",
                "match_no_match_conflict_resolved",
                "no_credible_candidate",
                "ambiguous_candidates"
            ]
        }
    },
    "required": [
        "decision_type",
        "selected_candidate_id",
        "confidence",
        "reasoning_summary",
        "resolution_mode"
    ]
}

SYSTEM_PROMPT = """
You are the final conflict resolver for EPC document title reconciliation. You are only invoked when the two agents (GPT and Claude) disagree.

Disagreement cases:
- match_match_conflict: both agents chose MATCH but selected different candidates (X vs Y).
- match_no_match_conflict: one agent chose MATCH, the other NO_MATCH.

Your task:
Resolve the conflict and decide the final outcome. You do NOT re-evaluate from scratch; you use the agents' decisions and their top-3 candidates as the primary focus, and the full top-50 list only to verify whether a better candidate was missed.

Allowed outcomes:
- MATCH: one candidate is clearly the best; choose only from the provided top-50.
- NO_MATCH: no candidate is sufficiently credible.
- MANUAL_REVIEW: genuine ambiguity between plausible candidates or insufficient evidence to safely accept or reject.

STRICT FORMAT RULES (CRITICAL):

1) Candidate IDs:
- In the FULL TOP50 CANDIDATES section, each candidate is identified by an ID in square brackets, e.g. [T01], [T02], ..., in the same order as they appear.
- When decision_type is MATCH, you MUST select exactly one of these IDs and return it as "selected_candidate_id" in JSON (e.g. "T02").
- When decision_type is NO_MATCH or MANUAL_REVIEW, "selected_candidate_id" MUST be null.

2) General decision rules:
- Primary focus: the two agents' top-3 candidate lists. The FULL TOP50 is the complete verification context.
- You may select ONLY from the provided top-50 candidates (via their IDs). Never invent candidates.
- SimilarityScore is retrieval evidence only; metadata and semantic meaning matter more than lexical overlap.
- Choose MATCH only if one candidate is clearly superior to all others and semantically consistent with the MDR record and its metadata.
- Choose NO_MATCH when the best available candidates are still weak, generic, metadata-incompatible, or not clearly equivalent.
- Choose MANUAL_REVIEW only when there is real ambiguity between plausible candidates or insufficient evidence to safely accept or reject one.

3) Output constraints:
- confidence must be between 0 and 1.
- reasoning_summary must be brief and factual (max 80 words). Avoid using " characters inside the text when possible.
- resolution_mode must be one of:
  - match_match_conflict_resolved
  - match_no_match_conflict_resolved
  - no_credible_candidate
  - ambiguous_candidates

Output JSON ONLY with the following fields:
- decision_type
- selected_candidate_id
- confidence
- resolution_mode
- reasoning_summary

EXAMPLES (VERY IMPORTANT):

Example 1 – MATCH with candidate ID

Suppose in FULL TOP50 CANDIDATES you see:

----
[T02]
Rank: 5
SimilarityScore: 0.9231
TitleKey: vendor dwgs and documents for centrifugal compressor for process service
RaciTitle: VENDOR DWGS AND DOCUMENTS FOR CENTRIFUGAL COMPRESSOR FOR PROCESS SERVICE
EffectiveDescription: ...
DisciplineName: Mechanical
TypeName: Vendor Drawings
...

If you decide that THIS is the best candidate, your JSON MUST contain:

{
  "decision_type": "MATCH",
  "selected_candidate_id": "T02",
  "confidence": 0.85,
  "resolution_mode": "match_match_conflict_resolved",
  "reasoning_summary": "Short, factual explanation (max 80 words, no extra quotes if possible)."
}

Example 2 – NO_MATCH with null candidate

If, after checking the agents' top-3 lists and the FULL TOP50 CANDIDATES, there is no single clearly credible candidate, you MUST output:

{
  "decision_type": "NO_MATCH",
  "selected_candidate_id": null,
  "confidence": 0.0,
  "resolution_mode": "no_credible_candidate",
  "reasoning_summary": "Explain briefly why no candidate is sufficiently credible."
}

Do NOT put any candidate title or description in selected_candidate_id. It must be either one of the IDs [T01]...[T50] (without brackets, e.g. "T02") or null.
"""


def build_user_prompt(
    mdr_ctx: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    agent1: Dict[str, Any],
    agent2: Dict[str, Any],
    top3_agent1: Optional[List[Dict[str, Any]]] = None,
    top3_agent2: Optional[List[Dict[str, Any]]] = None,
    disagreement_type: Optional[str] = None,
) -> str:
    blocks = []

    blocks.append("HISTORICAL MDR RECORD")
    blocks.append(f"Document_title: {norm(mdr_ctx.get('Document_title'))}")
    blocks.append(f"Discipline_Normalized: {norm(mdr_ctx.get('Discipline_Normalized'))}")
    blocks.append(f"Discipline_Status: {norm(mdr_ctx.get('Discipline_Status'))}")
    blocks.append(f"Type_L1: {norm(mdr_ctx.get('Type_L1'))}")
    blocks.append(f"Type_L1_Status: {norm(mdr_ctx.get('Type_L1_Status'))}")
    blocks.append("")

    if disagreement_type:
        blocks.append("DISAGREEMENT TYPE")
        blocks.append(disagreement_type)
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

    blocks.append("AGENT TOP CANDIDATES")
    if top3_agent1:
        blocks.append("Top 3 GPT (gpt5mini):")
        for i, tc in enumerate(top3_agent1[:3], 1):
            blocks.append(f"  {i}. TitleKey={norm(tc.get('TitleKey'))} RaciTitle={norm(tc.get('RaciTitle'))} Confidence={tc.get('CandidateConfidence')} WhyPlausible={norm(tc.get('WhyPlausible'))}")
        blocks.append("")
    if top3_agent2:
        blocks.append("Top 3 Claude:")
        for i, tc in enumerate(top3_agent2[:3], 1):
            blocks.append(f"  {i}. TitleKey={norm(tc.get('TitleKey'))} RaciTitle={norm(tc.get('RaciTitle'))} Confidence={tc.get('CandidateConfidence')} WhyPlausible={norm(tc.get('WhyPlausible'))}")
        blocks.append("")

    blocks.append("FULL TOP50 CANDIDATES")
    blocks.append("(Use for verification only; primary focus is the agents' top-3 lists above.)")
    for idx, c in enumerate(candidates, 1):
        cid = f"T{idx:02d}"
        selected_by = []
        if str(c["TitleKey"]) == str(agent1.get("SelectedTitleKey")):
            selected_by.append("Agent1")
        if str(c["TitleKey"]) == str(agent2.get("SelectedTitleKey")):
            selected_by.append("Agent2")
        blocks.append("----")
        blocks.append(f"[{cid}]")
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
    # Build ID -> candidate map (T01, T02, ...) in the same order as candidates.
    id_map: Dict[str, Dict[str, Any]] = {}
    for idx, c in enumerate(candidates, 1):
        cid = f"T{idx:02d}"
        id_map[cid] = c

    decision_type = result["decision_type"]
    selected_candidate_id = result.get("selected_candidate_id")
    confidence = float(result["confidence"])
    reasoning_summary = norm(result["reasoning_summary"])
    resolution_mode = norm(result["resolution_mode"])

    if confidence < 0:
        confidence = 0.0
    if confidence > 1:
        confidence = 1.0

    valid_resolution_modes = (
        "match_match_conflict_resolved", "match_no_match_conflict_resolved",
        "no_credible_candidate", "ambiguous_candidates"
    )
    if resolution_mode not in valid_resolution_modes:
        raise ValueError(
            f"resolution_mode must be one of {valid_resolution_modes}, got: {resolution_mode!r}. "
            "Invalid judge output: do not fallback; mark task as error."
        )

    if decision_type in ("NO_MATCH", "MANUAL_REVIEW"):
        # In questi casi selected_candidate_id deve essere null
        if selected_candidate_id not in (None, "null"):
            raise ValueError("NO_MATCH/MANUAL_REVIEW require selected_candidate_id = null")
        return {
            "FinalDecisionType": decision_type,
            "FinalTitleKey": None,
            "FinalRaciTitle": None,
            "FinalConfidence": confidence,
            "FinalReason": reasoning_summary or "No final candidate selected.",
            "ResolutionMode": resolution_mode,
        }

    if decision_type != "MATCH":
        raise ValueError(f"Invalid decision_type: {decision_type}")

    if not selected_candidate_id:
        raise ValueError("MATCH requires selected_candidate_id")

    if selected_candidate_id not in id_map:
        raise ValueError(f"selected_candidate_id not in candidates: {selected_candidate_id}")

    candidate = id_map[selected_candidate_id]
    final_titlekey = norm(candidate["TitleKey"])
    final_racititle = norm(candidate["RaciTitle"])

    return {
        "FinalDecisionType": "MATCH",
        "FinalTitleKey": final_titlekey,
        "FinalRaciTitle": final_racititle,
        "FinalConfidence": confidence,
        "FinalReason": reasoning_summary or "Judge selected the best supported candidate.",
        "ResolutionMode": resolution_mode,
    }


def resolve_conflict_with_gemini(
    con: duckdb.DuckDBPyConnection,
    model: str,
    mdr_ctx: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    agent1: Dict[str, Any],
    agent2: Dict[str, Any],
    top3_agent1: List[Dict[str, Any]],
    top3_agent2: List[Dict[str, Any]],
    task: Dict[str, Any],
    resolution_mode: str,
) -> Dict[str, Any]:
    """
    Call Gemini for conflict resolution; return result dict for write_final_result.
    resolution_mode must be RESOLUTION_LLM_MATCH_MATCH or RESOLUTION_LLM_MATCH_NO_MATCH.
    """
    disagreement_type = "match_match_conflict" if resolution_mode == RESOLUTION_LLM_MATCH_MATCH else "match_no_match_conflict"
    user_prompt = build_user_prompt(
        mdr_ctx, candidates, agent1, agent2,
        top3_agent1=top3_agent1 or None,
        top3_agent2=top3_agent2 or None,
        disagreement_type=disagreement_type,
    )
    full_prompt = f"{SYSTEM_PROMPT.strip()}\n\nUSER INPUT:\n{user_prompt}"
    response = _genai_client.models.generate_content(model=model, contents=full_prompt)
    raw_text = getattr(response, "text", None) or ""
    cleaned = _extract_json_payload(raw_text)
    raw_result = json.loads(cleaned)
    validated = validate_judge_output(raw_result, candidates)
    validated["ResolvedBy"] = "judge_script"
    validated["JudgeUsedFlag"] = True
    validated["JudgeModel"] = "gemini"
    return validated


# --------------------------------------------------
# Save outputs: judge_gemini row (only when Gemini used) + final result
# --------------------------------------------------
def save_judge_agent_decision(
    con: duckdb.DuckDBPyConnection,
    task: Dict[str, Any],
    model: str,
    judge_result: Dict[str, Any],
    output_prompt_version: Optional[str] = None,
) -> None:
    """Insert judge_gemini row into MdrReconciliationAgentDecisions (only when Gemini was called)."""
    ts = now_ts_naive_utc()
    pv = (output_prompt_version and output_prompt_version.strip()) or task["PromptVersion"]
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
        JUDGE_AGENT_NAME_GEMINI,
        model,
        task["Document_title"],
        pv,
        task["EmbeddingModel"],
        judge_result["FinalTitleKey"],
        judge_result["FinalRaciTitle"],
        judge_result["FinalDecisionType"],
        judge_result["FinalConfidence"],
        judge_result["FinalReason"],
        ts,
    ])


def write_final_result(
    con: duckdb.DuckDBPyConnection,
    task: Dict[str, Any],
    result_dict: Dict[str, Any],
    output_prompt_version: Optional[str] = None,
) -> None:
    """
    Write final result to MdrReconciliationResults (only writer). Update task status.
    result_dict must contain: FinalDecisionType, FinalTitleKey, FinalRaciTitle, FinalConfidence,
    FinalReason, ResolutionMode, ResolvedBy, JudgeUsedFlag, JudgeModel.
    """
    ts = now_ts_naive_utc()
    pv = (output_prompt_version and output_prompt_version.strip()) or task["PromptVersion"]
    next_final_status = "completed"
    if result_dict.get("FinalDecisionType") == "MANUAL_REVIEW":
        next_final_status = "manual_review"

    con.execute("BEGIN;")
    try:
        con.execute(f"""
            INSERT INTO {FINAL_RESULTS_TABLE}
              (TaskId, Document_title, PromptVersion, EmbeddingModel,
               FinalTitleKey, FinalRaciTitle, FinalDecisionType, FinalConfidence,
               ResolutionMode, FinalReason, ResolvedBy, JudgeUsedFlag, JudgeModel, CreatedAt, UpdatedAt)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (TaskId, PromptVersion, EmbeddingModel) DO UPDATE SET
              Document_title = excluded.Document_title,
              FinalTitleKey = excluded.FinalTitleKey,
              FinalRaciTitle = excluded.FinalRaciTitle,
              FinalDecisionType = excluded.FinalDecisionType,
              FinalConfidence = excluded.FinalConfidence,
              ResolutionMode = excluded.ResolutionMode,
              FinalReason = excluded.FinalReason,
              ResolvedBy = excluded.ResolvedBy,
              JudgeUsedFlag = excluded.JudgeUsedFlag,
              JudgeModel = excluded.JudgeModel,
              UpdatedAt = excluded.UpdatedAt
        """, [
            task["TaskId"],
            task["Document_title"],
            pv,
            task["EmbeddingModel"],
            result_dict["FinalTitleKey"],
            result_dict["FinalRaciTitle"],
            result_dict["FinalDecisionType"],
            result_dict["FinalConfidence"],
            result_dict["ResolutionMode"],
            result_dict["FinalReason"],
            result_dict.get("ResolvedBy", "judge_script"),
            result_dict.get("JudgeUsedFlag", False),
            result_dict.get("JudgeModel"),
            ts,
            ts,
        ])
        con.execute(f"""
            UPDATE {TASKS_TABLE}
            SET JudgeStatus = 'done', FinalStatus = ?, UpdatedAt = ?
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
            # Aumentiamo il budget di token per permettere a Gemini
            # di restituire JSON completi (decisione + reasoning_summary).
            "generationConfig": {"temperature": 0, "maxOutputTokens": 4096},
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


def _extract_task_id_from_vertex_batch_line(line_obj: Dict[str, Any]) -> Optional[str]:
    """
    Extract TaskId correlation ID from the echoed request text in a Vertex batch output line.
    We embed a line like: 'TASK_ID: <32-hex>' into the prompt at submit time, then recover it here.
    """
    if not isinstance(line_obj, dict):
        return None
    req = line_obj.get("request") or {}
    contents = req.get("contents") or []
    if not contents:
        return None
    parts = (contents[0] or {}).get("parts") or []
    if not parts:
        return None
    text = (parts[0] or {}).get("text") or ""
    if not isinstance(text, str) or not text:
        return None
    m = re.search(r"\bTASK_ID:\s*([0-9a-f]{32})\b", text, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).lower()


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
    output_prompt_version: Optional[str] = None,
) -> str:
    """
    Classify each task: consensus -> write_final_result immediately; conflict -> add to batch.
    Returns job name if batch was created; empty string if all consensus (no batch).
    """
    bucket = _cfg("VERTEX_BATCH_GCS_BUCKET")
    if not bucket:
        raise RuntimeError("Per il batch Vertex serve VERTEX_BATCH_GCS_BUCKET in config.txt")
    prefix = (_cfg("VERTEX_BATCH_GCS_PREFIX") or "").strip().rstrip("/")
    if prefix:
        prefix = prefix + "/"
    project_id = _cfg("VERTEX_PROJECT_ID")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    consensus_count = 0
    lines: List[str] = []
    conflict_tasks: List[Dict[str, str]] = []  # [{"task_id": ..., "resolution_mode": ...}, ...]

    for task in tasks:
        agent1 = load_agent_decision(con, task["TaskId"], AGENT1_NAME)
        agent2 = load_agent_decision(con, task["TaskId"], AGENT2_NAME)
        if not agent1 or not agent2:
            continue
        case = classify_resolution_case(agent1, agent2)
        if case == CASE_MISSING_DATA:
            continue
        if case == CASE_CONSENSUS_MATCH or case == CASE_CONSENSUS_NO_MATCH:
            result_dict = resolve_consensus_case(case, agent1, agent2, task)
            write_final_result(con, task, result_dict, output_prompt_version)
            consensus_count += 1
            continue
        if case == CASE_MATCH_MATCH_CONFLICT:
            resolution_mode = RESOLUTION_LLM_MATCH_MATCH
        else:
            resolution_mode = RESOLUTION_LLM_MATCH_NO_MATCH

        candidates = load_retrieval_candidates(
            con, task["Document_title"], task["PromptVersion"], task["EmbeddingModel"], limit=50
        )
        if not candidates:
            continue
        top3_a1 = load_agent_top_candidates(con, task["TaskId"], AGENT1_NAME)
        top3_a2 = load_agent_top_candidates(con, task["TaskId"], AGENT2_NAME)
        mdr_ctx = fetch_mdr_context(con, task["Document_title"])
        disagreement_type = "match_match_conflict" if resolution_mode == RESOLUTION_LLM_MATCH_MATCH else "match_no_match_conflict"
        user_prompt = build_user_prompt(
            mdr_ctx, candidates, agent1, agent2,
            top3_agent1=top3_a1 or None,
            top3_agent2=top3_a2 or None,
            disagreement_type=disagreement_type,
        )
        # Correlation ID: embed TaskId into the request text so batch outputs can be mapped
        # deterministically without relying on output row order.
        user_prompt_with_id = f"TASK_ID: {task['TaskId']}\n{user_prompt}"
        full_prompt = f"{SYSTEM_PROMPT.strip()}\n\nUSER INPUT:\n{user_prompt_with_id}"
        line_obj = _vertex_batch_request_line(full_prompt)
        lines.append(json.dumps(line_obj))
        conflict_tasks.append({"task_id": task["TaskId"], "resolution_mode": resolution_mode})

    if consensus_count:
        print(f"Resolved by consensus (no Gemini): {consensus_count} task(s).")
    if not lines:
        if consensus_count == len(tasks):
            print("All tasks resolved by consensus; no batch created.")
        else:
            print("No conflict tasks to submit (or missing data).")
        return ""

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

    submitted_task_ids = [ct["task_id"] for ct in conflict_tasks]
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
            "conflict_tasks": conflict_tasks,
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
    expected_ids = set(str(x).strip().lower() for x in (task_ids or []) if str(x).strip())
    for i, line_obj in enumerate(output_lines):
        task_id = _extract_task_id_from_vertex_batch_line(line_obj)
        if not task_id:
            errors += 1
            print(f"  row {i}: missing TASK_ID in echoed request; cannot map result to task")
            continue
        if expected_ids and task_id.lower() not in expected_ids:
            errors += 1
            print(f"  {task_id}: TASK_ID not in expected task_ids list; skipping row")
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
            validated["ResolvedBy"] = "judge_script"
            validated["JudgeUsedFlag"] = True
            validated["JudgeModel"] = "gemini"
            save_judge_agent_decision(con, task, model, validated, output_prompt_version)
            write_final_result(con, task, validated, output_prompt_version)
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
    """
    Run judge pipeline for one task (caller must have claimed it).
    Load data -> classify -> consensus: write_final_result; conflict: Gemini -> save_judge_agent_decision -> write_final_result.
    Returns result dict on success, None on skip/error.
    """
    agent1 = load_agent_decision(con, task["TaskId"], AGENT1_NAME)
    agent2 = load_agent_decision(con, task["TaskId"], AGENT2_NAME)
    if not agent1 or not agent2:
        mark_judge_error(con, task["TaskId"])
        return None

    top3_a1 = load_agent_top_candidates(con, task["TaskId"], AGENT1_NAME)
    top3_a2 = load_agent_top_candidates(con, task["TaskId"], AGENT2_NAME)
    candidates = load_retrieval_candidates(
        con, task["Document_title"], task["PromptVersion"], task["EmbeddingModel"], limit=50
    )
    if not candidates:
        mark_judge_error(con, task["TaskId"])
        return None

    case = classify_resolution_case(agent1, agent2)
    if case == CASE_MISSING_DATA:
        mark_judge_error(con, task["TaskId"])
        return None

    if case == CASE_CONSENSUS_MATCH or case == CASE_CONSENSUS_NO_MATCH:
        result_dict = resolve_consensus_case(case, agent1, agent2, task)
        write_final_result(con, task, result_dict, output_prompt_version)
        return result_dict

    if case == CASE_MATCH_MATCH_CONFLICT:
        resolution_mode = RESOLUTION_LLM_MATCH_MATCH
    else:
        resolution_mode = RESOLUTION_LLM_MATCH_NO_MATCH

    mdr_ctx = fetch_mdr_context(con, task["Document_title"])
    result_dict = resolve_conflict_with_gemini(
        con, model, mdr_ctx, candidates, agent1, agent2,
        top3_a1, top3_a2, task, resolution_mode,
    )
    save_judge_agent_decision(con, task, model, result_dict, output_prompt_version)
    write_final_result(con, task, result_dict, output_prompt_version)
    return result_dict


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
    ap.add_argument("--test-fixed-tasks", action="store_true", help="Test mode: use a shared fixed list of TaskId stored in .recon_test_tasks.json.")
    ap.add_argument("--test-task-count", type=int, default=150, help="Number of random TaskId to pick when creating the shared test list (default: 150).")
    args = ap.parse_args()

    args.prompt_version = args.prompt_version or _cfg("PROMPT_VERSION", "v1")
    args.embedding_model = args.embedding_model or "text-embedding-3-small"
    args.output_prompt_version = (args.output_prompt_version or _cfg("JUDGE_OUTPUT_PROMPT_VERSION") or "").strip() or None
    args.model = args.model or DEFAULT_MODEL

    con = connect_motherduck()
    ensure_final_results_table(con)
    ensure_agent_top_candidates_table(con)

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
        if args.test_fixed_tasks:
            tasks = load_or_create_test_tasks(tasks, args.test_task_count)
        job_name = run_batch_submit(
            con=con, tasks=tasks, model=args.model, output_prompt_version=args.output_prompt_version
        )
        con.close()
        if job_name:
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

    if args.test_fixed_tasks:
        tasks = load_or_create_test_tasks(tasks, args.test_task_count)

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


# --------------------------------------------------
# Proposed test cases (no test file exists yet)
# --------------------------------------------------
# 1. consensus_match: agent1=MATCH(X), agent2=MATCH(X) -> no Gemini, write_final_result with
#    ResolutionMode=judge_script_consensus_match, JudgeUsedFlag=false, no row in AgentDecisions for judge_gemini.
# 2. consensus_no_match: agent1=NO_MATCH, agent2=NO_MATCH -> no Gemini, write_final_result with
#    FinalDecisionType=NO_MATCH, ResolutionMode=judge_script_consensus_no_match, JudgeUsedFlag=false.
# 3. match_match_conflict: agent1=MATCH(X), agent2=MATCH(Y), X!=Y -> call Gemini, save_judge_agent_decision,
#    write_final_result with ResolutionMode=judge_llm_match_match_conflict, JudgeUsedFlag=true, JudgeModel=gemini.
# 4. match_no_match_conflict: agent1=MATCH(X), agent2=NO_MATCH (or vice versa) -> call Gemini,
#    write_final_result with ResolutionMode=judge_llm_match_no_match_conflict, JudgeUsedFlag=true.

if __name__ == "__main__":
    main()
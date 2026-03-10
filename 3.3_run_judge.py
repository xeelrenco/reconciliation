#!/usr/bin/env python3

import json
import argparse
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
from openai import OpenAI

# -----------------------------
# Config da file (stesso formato degli altri script)
# -----------------------------
CONFIG_PATH = Path(__file__).resolve().parent / "config.txt"


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
# Bootstrap final results table
# --------------------------------------------------
def ensure_final_results_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"""
    CREATE TABLE IF NOT EXISTS {FINAL_RESULTS_TABLE} (
      TaskId               VARCHAR PRIMARY KEY,
      Document_title       VARCHAR NOT NULL,
      PromptVersion        VARCHAR NOT NULL,
      EmbeddingModel       VARCHAR NOT NULL,
      FinalTitleKey        VARCHAR,
      FinalRaciTitle       VARCHAR,
      FinalDecisionType    VARCHAR NOT NULL,   -- MATCH | NO_MATCH | MANUAL_REVIEW
      FinalConfidence      DOUBLE,
      ResolutionMode       VARCHAR NOT NULL,   -- UNANIMOUS | JUDGE_WITH_AGENT | ALL_DIFFERENT | NO_MATCH_CONFIRMED | MANUAL_REVIEW
      FinalReason          VARCHAR NOT NULL,
      CreatedAt            TIMESTAMP NOT NULL,
      UpdatedAt            TIMESTAMP NOT NULL
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
- a list of 50 candidate RACI documents
- the decisions of two independent agents (gpt5mini and claude)

Your task:
Determine the final reconciliation outcome.

Allowed outcomes:
- MATCH
- NO_MATCH
- MANUAL_REVIEW

Definitions:

MATCH
Choose MATCH only if there is a single candidate that is clearly the best semantic match to the MDR document.

NO_MATCH
Choose NO_MATCH if none of the candidates is a credible match.

MANUAL_REVIEW
Choose MANUAL_REVIEW if at least one candidate appears plausible but the evidence is ambiguous, conflicting, or insufficient for a safe automatic match.

Core principles:

1. Be conservative and precise.
2. Do not invent information.
3. Do not rely on external knowledge.
4. Only select candidates from the provided list.
5. Similarity score is only a retrieval signal and must not be treated as proof of equivalence.
6. Prefer semantic equivalence between the MDR title and the candidate description.
7. Metadata compatibility (discipline, type, category, chapter) is important supporting evidence.
8. Strong metadata incompatibility is a negative signal.
9. Generic lexical overlap alone is not sufficient to justify MATCH.

Use prior agent decisions as evidence, not authority.

Decision logic:

1. If both agents selected the same candidate AND the candidate is semantically consistent → MATCH is strongly supported.

2. If the agents disagree:
   - You may still choose MATCH if one candidate is clearly superior based on semantic meaning and metadata consistency.

3. If none of the candidates is credible → choose NO_MATCH.

4. If multiple candidates remain plausible and cannot be safely distinguished → choose MANUAL_REVIEW.

Output requirements:

Return JSON only with the following fields:
- decision_type
- selected_titlekey
- selected_raci_title
- confidence
- resolution_mode
- reasoning_summary

resolution_mode (required): choose exactly one of:
- agent_consensus: both agents selected the same candidate and you confirm it.
- judge_override: you select one of the two agents' choice or a different candidate from the list.
- no_credible_candidate: no candidate is plausible (use with NO_MATCH).
- ambiguous_candidates: plausible candidates exist but are not clearly distinguishable (use with MANUAL_REVIEW).

Rules:
- confidence must be between 0 and 1
- reasoning_summary must be concise and factual (maximum 100 words)
- selected_titlekey and selected_raci_title must be null if decision_type is NO_MATCH or MANUAL_REVIEW
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
    blocks.append("- Choose MATCH only if a single candidate is clearly the best")
    blocks.append("- Choose NO_MATCH if none of the candidates is credible")
    blocks.append("- Choose MANUAL_REVIEW if multiple candidates remain plausible or evidence conflicts")
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
    judge_result: Dict[str, Any]
) -> None:
    ts = now_ts_naive_utc()

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
            task["PromptVersion"],
            task["EmbeddingModel"],
            judge_result["FinalTitleKey"],
            judge_result["FinalRaciTitle"],
            judge_result["FinalDecisionType"],
            judge_result["FinalConfidence"],
            judge_result["FinalReason"],
            ts
        ])

        # Save final result
        con.execute(f"""
            INSERT INTO {FINAL_RESULTS_TABLE}
              (TaskId, Document_title, PromptVersion, EmbeddingModel,
               FinalTitleKey, FinalRaciTitle, FinalDecisionType, FinalConfidence,
               ResolutionMode, FinalReason, CreatedAt, UpdatedAt)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (TaskId) DO UPDATE SET
              Document_title = excluded.Document_title,
              PromptVersion = excluded.PromptVersion,
              EmbeddingModel = excluded.EmbeddingModel,
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
            task["PromptVersion"],
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


def mark_judge_error(con: duckdb.DuckDBPyConnection, task_id: str) -> None:
    ts = now_ts_naive_utc()
    con.execute(f"""
        UPDATE {TASKS_TABLE}
        SET
          JudgeStatus = 'error',
          FinalStatus = 'error',
          UpdatedAt = ?
        WHERE TaskId = ?
    """, [ts, task_id])


def process_one_judge_task(
    con: duckdb.DuckDBPyConnection,
    task: Dict[str, Any],
    model: str
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
    save_judge_result(con=con, task=task, model=model, judge_result=validated_result)
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
                claimed = claim_task_judge(con, task["TaskId"])
                if not claimed:
                    continue
                processed = True
                with print_lock:
                    print(f"[{threading.current_thread().name}] Processing: {task['Document_title']}")
                result = process_one_judge_task(con, task, model)
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
    ap.add_argument("--model", default=None, help="OpenAI model for judge (default: gpt-5-mini).")
    args = ap.parse_args()

    args.prompt_version = args.prompt_version or _cfg("PROMPT_VERSION", "v1")
    args.embedding_model = args.embedding_model or "text-embedding-3-small"
    args.model = args.model or DEFAULT_MODEL

    con = connect_motherduck()
    ensure_final_results_table(con)
    tasks = fetch_ready_tasks(
        con=con,
        prompt_version=args.prompt_version,
        embedding_model=args.embedding_model,
        limit=args.limit
    )
    con.close()

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
            args=(task_queue, args.model, print_lock, total_tasks, completed_count),
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
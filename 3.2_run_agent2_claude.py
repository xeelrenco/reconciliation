#!/usr/bin/env python3

import json
import argparse
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import anthropic

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


AGENT_NAME = "claude"
DEFAULT_MODEL = "claude-sonnet-4-6"

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


SYSTEM_PROMPT = """
You are a specialist agent for document title reconciliation in EPC project documentation.

You will receive:
- one historical MDR title with normalized metadata
- a list of 50 candidate standard RACI documents

Your task:
Determine whether one candidate is a sufficiently credible semantic match.

Allowed outcomes:
- MATCH
- NO_MATCH

Definitions:

MATCH
Choose MATCH only if one candidate is clearly the best semantic match to the MDR record.

NO_MATCH
Choose NO_MATCH if no candidate is sufficiently credible based on semantic meaning and metadata consistency.

Core principles:
- Only select from the provided candidates.
- Do not invent missing information.
- Do not use external knowledge.
- Similarity score and rank are retrieval hints, not proof of equivalence.
- Prefer semantic equivalence over lexical overlap.
- Use MDR metadata (discipline, document type) as context.
- Use candidate title, description, discipline, type, category, and chapter as supporting evidence.
- Strong metadata incompatibility is negative evidence.
- Generic wording overlap alone is not sufficient for MATCH.

Decision rules:
- Return MATCH only when one candidate is clearly stronger than the others.
- If multiple candidates appear plausible but none is clearly superior, return NO_MATCH.
- If the best candidate is still weak, generic, or not clearly equivalent, return NO_MATCH.

Output format:
Return JSON only with:
- decision_type
- selected_titlekey
- selected_raci_title
- confidence
- reasoning_summary

Rules:
- confidence must be between 0 and 1
- selected_titlekey and selected_raci_title must be null when decision_type is NO_MATCH
- reasoning_summary must be concise and factual (maximum 80 words)
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
                max_tokens=600,
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
    return json.loads(cleaned_json)


def validate_agent_output(result: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    candidate_map = {norm(c["TitleKey"]): c for c in candidates}

    decision_type = result["decision_type"]
    selected_titlekey = result["selected_titlekey"]
    selected_raci_title = result["selected_raci_title"]
    confidence = float(result["confidence"])
    reasoning_summary = norm(result["reasoning_summary"])

    if confidence < 0:
        confidence = 0.0
    if confidence > 1:
        confidence = 1.0

    if decision_type == "NO_MATCH":
        return {
            "DecisionType": "NO_MATCH",
            "SelectedTitleKey": None,
            "SelectedRaciTitle": None,
            "Confidence": confidence,
            "ReasoningSummary": reasoning_summary or "No credible match among the provided candidates."
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
        "DecisionType": "MATCH",
        "SelectedTitleKey": selected_titlekey,
        "SelectedRaciTitle": norm(selected_raci_title),
        "Confidence": confidence,
        "ReasoningSummary": reasoning_summary or "Selected best semantic match among provided candidates."
    }


def save_agent2_evaluation(
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

        con.execute("COMMIT;")
    except Exception:
        con.execute("ROLLBACK;")
        raise

def mark_agent2_error(con: duckdb.DuckDBPyConnection, task_id: str) -> None:
    ts = now_ts_naive_utc()
    con.execute("""
        UPDATE my_db.mdr_reconciliation.MdrReconciliationTasks
        SET
          Agent2Status = 'error',
          FinalStatus = 'error',
          UpdatedAt = ?
        WHERE TaskId = ?
    """, [ts, task_id])


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
    args = ap.parse_args()

    args.prompt_version = args.prompt_version or _cfg("PROMPT_VERSION", "v1")
    args.embedding_model = args.embedding_model or "text-embedding-3-small"
    args.model = args.model or DEFAULT_MODEL

    con = connect_motherduck()
    ensure_agent_eval_table(con)
    tasks = fetch_pending_tasks(
        con=con,
        prompt_version=args.prompt_version,
        embedding_model=args.embedding_model,
        limit=args.limit
    )
    con.close()

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
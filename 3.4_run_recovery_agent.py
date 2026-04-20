#!/usr/bin/env python3
"""
Recovery agent post-reconciliation.

Purpose:
- Revisit tasks already processed by the standard 3-agent pipeline
- Do NOT overwrite MdrReconciliationResults
- Save a separate recovery proposal in MdrReconciliationRecoveryResults

Modes:
- manual-review: re-evaluate tasks whose final result is MANUAL_REVIEW
- no-match: re-evaluate tasks whose final result is NO_MATCH
- both: process both categories

Candidate pool:
- Uses only the union of GPT/Claude top-3 candidates already saved in
  MdrReconciliationAgentTopCandidates
- Does not re-run retrieval and does not use full top50 candidates

Config:
- config.txt (MOTHERDUCK_*, OPENAI_API_KEY, PROMPT_VERSION)

Usage:
  python 3.4_run_recovery_agent.py --mode manual-review
  python 3.4_run_recovery_agent.py --mode no-match
  python 3.4_run_recovery_agent.py --mode both --limit 50 --workers 4
"""

import argparse
import json
import queue
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
from openai import OpenAI


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


DEFAULT_MODEL = _cfg("LLM_MODEL", "gpt-5-mini")
RECOVERY_AGENT_NAME = "recovery_gpt"

DB_SCHEMA = "my_db.mdr_reconciliation"
TASKS_TABLE = f"{DB_SCHEMA}.MdrReconciliationTasks"
RESULTS_TABLE = f"{DB_SCHEMA}.MdrReconciliationResults"
AGENT_DECISIONS_TABLE = f"{DB_SCHEMA}.MdrReconciliationAgentDecisions"
AGENT_TOP_CANDIDATES_TABLE = f"{DB_SCHEMA}.MdrReconciliationAgentTopCandidates"
RECOVERY_RESULTS_TABLE = f"{DB_SCHEMA}.MdrReconciliationRecoveryResults"
MDR_VIEW = "my_db.historical_mdr_normalization.v_MdrPreviousRecords_Normalized_All"

AGENT1_NAME = "gpt5mini"
AGENT2_NAME = "claude"
JUDGE_AGENT_NAME = "judge_gemini"

STAGE_MANUAL_REVIEW = "manual_review_resolver"
STAGE_NO_MATCH = "no_match_recovery"
RECOVERY_CANDIDATE_POOL_TYPE = "agent_top3_union_only"

MODE_TO_SOURCE = {
    "manual-review": "MANUAL_REVIEW",
    "no-match": "NO_MATCH",
}

SOURCE_TO_STAGE = {
    "MANUAL_REVIEW": STAGE_MANUAL_REVIEW,
    "NO_MATCH": STAGE_NO_MATCH,
}


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


def ensure_recovery_results_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"""
    CREATE TABLE IF NOT EXISTS {RECOVERY_RESULTS_TABLE} (
      TaskId                  VARCHAR NOT NULL,
      Document_title          VARCHAR NOT NULL,
      PromptVersion           VARCHAR NOT NULL,
      EmbeddingModel          VARCHAR NOT NULL,
      SourceFinalDecisionType VARCHAR NOT NULL,
      RecoveryStage           VARCHAR NOT NULL,
      RecoveryAgentName       VARCHAR NOT NULL,
      RecoveryModel           VARCHAR,
      RecoveryDecisionType    VARCHAR NOT NULL,
      RecoveryTitleKey        VARCHAR,
      RecoveryRaciTitle       VARCHAR,
      RecoveryConfidence      DOUBLE,
      RecoveryReason          VARCHAR NOT NULL,
      RecoveryMode            VARCHAR NOT NULL,
      CandidatePoolType       VARCHAR NOT NULL,
      CandidatePoolSize       INTEGER,
      CreatedAt               TIMESTAMP NOT NULL,
      UpdatedAt               TIMESTAMP NOT NULL,
      PRIMARY KEY (TaskId, PromptVersion, EmbeddingModel, RecoveryStage)
    );
    """)


def recovery_stage_for_final_decision(final_decision_type: str) -> str:
    try:
        return SOURCE_TO_STAGE[str(final_decision_type).strip().upper()]
    except KeyError as e:
        raise ValueError(f"Unsupported final decision type for recovery: {final_decision_type!r}") from e


def fetch_tasks(
    con: duckdb.DuckDBPyConnection,
    mode: str,
    prompt_version: str,
    embedding_model: Optional[str] = None,
    limit: Optional[int] = None,
    rerun_existing: bool = False,
) -> List[Dict[str, Any]]:
    source_decisions: List[str]
    if mode == "both":
        source_decisions = ["MANUAL_REVIEW", "NO_MATCH"]
    else:
        source_decisions = [MODE_TO_SOURCE[mode]]

    sql = f"""
        SELECT
          t.TaskId,
          t.Document_title,
          t.PromptVersion,
          t.EmbeddingModel,
          r.FinalDecisionType,
          r.FinalTitleKey,
          r.FinalRaciTitle,
          r.FinalConfidence,
          r.ResolutionMode,
          r.FinalReason,
          rr.RecoveryStage AS ExistingRecoveryStage
        FROM {RESULTS_TABLE} r
        JOIN {TASKS_TABLE} t
          ON t.TaskId = r.TaskId
         AND t.PromptVersion = r.PromptVersion
         AND t.EmbeddingModel = r.EmbeddingModel
        LEFT JOIN {RECOVERY_RESULTS_TABLE} rr
          ON rr.TaskId = r.TaskId
         AND rr.PromptVersion = r.PromptVersion
         AND rr.EmbeddingModel = r.EmbeddingModel
         AND rr.RecoveryStage = CASE
              WHEN r.FinalDecisionType = 'MANUAL_REVIEW' THEN '{STAGE_MANUAL_REVIEW}'
              WHEN r.FinalDecisionType = 'NO_MATCH' THEN '{STAGE_NO_MATCH}'
              ELSE '__unsupported__'
             END
        WHERE r.PromptVersion = ?
          AND r.FinalDecisionType IN ({",".join(["?"] * len(source_decisions))})
    """
    params: List[Any] = [prompt_version, *source_decisions]
    if embedding_model:
        sql += " AND r.EmbeddingModel = ?"
        params.append(embedding_model)
    if not rerun_existing:
        sql += " AND rr.TaskId IS NULL"
    sql += " ORDER BY r.FinalDecisionType, t.Document_title"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    rows = con.execute(sql, params).fetchall()
    cols = [
        "TaskId",
        "Document_title",
        "PromptVersion",
        "EmbeddingModel",
        "FinalDecisionType",
        "FinalTitleKey",
        "FinalRaciTitle",
        "FinalConfidence",
        "ResolutionMode",
        "FinalReason",
        "ExistingRecoveryStage",
    ]
    return [dict(zip(cols, r)) for r in rows]


def fetch_mdr_context(con: duckdb.DuckDBPyConnection, document_title: str) -> Dict[str, Any]:
    row = con.execute(
        f"""
        SELECT
          Document_title,
          Discipline_Normalized,
          Discipline_Status,
          Type_L1,
          Type_L1_Status
        FROM {MDR_VIEW}
        WHERE Document_title = ?
        LIMIT 1
        """,
        [document_title],
    ).fetchone()
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


def load_agent_decision(
    con: duckdb.DuckDBPyConnection,
    task_id: str,
    agent_name: str,
) -> Optional[Dict[str, Any]]:
    row = con.execute(
        f"""
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
        LIMIT 1
        """,
        [task_id, agent_name],
    ).fetchone()
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
    rows = con.execute(
        f"""
        SELECT
          CandidateRankWithinAgent,
          TitleKey,
          RaciTitle,
          CandidateConfidence,
          WhyPlausible
        FROM {AGENT_TOP_CANDIDATES_TABLE}
        WHERE TaskId = ? AND AgentName = ?
        ORDER BY CandidateRankWithinAgent
        """,
        [task_id, agent_name],
    ).fetchall()
    cols = [
        "CandidateRankWithinAgent",
        "TitleKey",
        "RaciTitle",
        "CandidateConfidence",
        "WhyPlausible",
    ]
    return [dict(zip(cols, r)) for r in rows]


def build_agent_pool(
    top3_gpt: List[Dict[str, Any]],
    top3_claude: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    pool: Dict[str, Dict[str, Any]] = {}

    def _add(agent_label: str, items: List[Dict[str, Any]]) -> None:
        for item in items:
            key = norm(item.get("TitleKey"))
            if not key:
                continue
            if key not in pool:
                pool[key] = {
                    "TitleKey": key,
                    "RaciTitle": norm(item.get("RaciTitle")),
                    "Sources": [],
                }
            pool[key]["Sources"].append(
                {
                    "Agent": agent_label,
                    "Rank": item.get("CandidateRankWithinAgent"),
                    "Confidence": item.get("CandidateConfidence"),
                    "WhyPlausible": norm(item.get("WhyPlausible")),
                }
            )

    _add("GPT", top3_gpt or [])
    _add("Claude", top3_claude or [])

    merged = list(pool.values())
    merged.sort(
        key=lambda c: (
            min(int(s.get("Rank") or 999) for s in c["Sources"]),
            c["TitleKey"],
        )
    )
    for idx, item in enumerate(merged, 1):
        item["CandidateId"] = f"C{idx:02d}"
    return merged


RECOVERY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision_type": {
            "type": "string",
            "enum": ["MATCH", "NO_MATCH"],
        },
        "selected_candidate_id": {
            "type": ["string", "null"],
        },
        "confidence": {
            "type": "number",
        },
        "reasoning_summary": {
            "type": "string",
        },
        "recovery_mode": {
            "type": "string",
            "enum": [
                "manual_review_forced_match",
                "manual_review_forced_no_match",
                "no_match_recovered_to_match",
                "no_match_confirmed",
            ],
        },
    },
    "required": [
        "decision_type",
        "selected_candidate_id",
        "confidence",
        "reasoning_summary",
        "recovery_mode",
    ],
}


SYSTEM_PROMPT = """
You are a recovery agent for MDR-to-RACI reconciliation.

You are invoked only after the main reconciliation pipeline has already run.
Your task is to re-evaluate difficult cases using ONLY the candidate pool already
proposed by the two primary agents.

You will receive:
- one historical MDR title with normalized metadata
- the current final outcome from the baseline pipeline
- GPT and Claude decisions/reasoning
- GPT and Claude top candidates
- a merged candidate pool built ONLY from the union of GPT/Claude top candidates

Important:
- Do NOT invent new candidates.
- Do NOT use any candidates outside the provided AGENT POOL.
- You MUST return only MATCH or NO_MATCH.
- If you choose MATCH, the selected candidate must be one of the candidate IDs in the AGENT POOL.
- If you choose NO_MATCH, selected_candidate_id must be null.

Decision policy (strict):
- Prefer MATCH when one candidate is the best available semantic fit, even if not perfect.
- Use NO_MATCH only when no candidate is credibly compatible with the MDR title intent.
- Do not reject a candidate only because it is broader/narrower if it still preserves the main document intent.
- If the ideal title is missing from the pool, select the closest available candidate and explain the limitation.
- Prioritize, in order:
  1) core semantic intent of MDR title,
  2) document type compatibility (layout/specification/drawing/etc.),
  3) discipline/metadata consistency.
- For civil/structural cases, prefer object+document-type alignment over restrictive qualifiers.
- For layout cases, prefer direct layout-title alignment when available.
- For electrical/ICT panel cases, prefer equipment form-factor alignment (panel/switchboard/layout) over generic system wording when semantically close.

Mode-specific meanings:
- manual_review_resolver:
  - manual_review_forced_match
  - manual_review_forced_no_match
- no_match_recovery:
  - no_match_recovered_to_match
  - no_match_confirmed

Output JSON only with:
- decision_type
- selected_candidate_id
- confidence
- reasoning_summary
- recovery_mode
"""


def build_user_prompt(
    stage: str,
    task: Dict[str, Any],
    mdr_ctx: Dict[str, Any],
    gpt_decision: Optional[Dict[str, Any]],
    claude_decision: Optional[Dict[str, Any]],
    judge_decision: Optional[Dict[str, Any]],
    pool: List[Dict[str, Any]],
) -> str:
    blocks: List[str] = []

    blocks.append("RECOVERY STAGE")
    blocks.append(stage)
    blocks.append("")

    blocks.append("CURRENT FINAL RESULT")
    blocks.append(f"FinalDecisionType: {norm(task.get('FinalDecisionType'))}")
    blocks.append(f"FinalTitleKey: {norm(task.get('FinalTitleKey'))}")
    blocks.append(f"FinalRaciTitle: {norm(task.get('FinalRaciTitle'))}")
    blocks.append(f"ResolutionMode: {norm(task.get('ResolutionMode'))}")
    blocks.append(f"FinalReason: {norm(task.get('FinalReason'))}")
    blocks.append("")

    blocks.append("HISTORICAL MDR RECORD")
    blocks.append(f"Document_title: {norm(mdr_ctx.get('Document_title'))}")
    blocks.append(f"Discipline_Normalized: {norm(mdr_ctx.get('Discipline_Normalized'))}")
    blocks.append(f"Discipline_Status: {norm(mdr_ctx.get('Discipline_Status'))}")
    blocks.append(f"Type_L1: {norm(mdr_ctx.get('Type_L1'))}")
    blocks.append(f"Type_L1_Status: {norm(mdr_ctx.get('Type_L1_Status'))}")
    blocks.append("")

    def _append_decision(label: str, dec: Optional[Dict[str, Any]]) -> None:
        if not dec:
            blocks.append(f"{label}: <missing>")
            blocks.append("")
            return
        blocks.append(label)
        blocks.append(f"DecisionType: {norm(dec.get('DecisionType'))}")
        blocks.append(f"SelectedTitleKey: {norm(dec.get('SelectedTitleKey'))}")
        blocks.append(f"SelectedRaciTitle: {norm(dec.get('SelectedRaciTitle'))}")
        blocks.append(f"Confidence: {dec.get('Confidence')}")
        blocks.append(f"ReasoningSummary: {norm(dec.get('ReasoningSummary'))}")
        blocks.append("")

    blocks.append("AGENT DECISIONS")
    _append_decision("GPT", gpt_decision)
    _append_decision("Claude", claude_decision)
    _append_decision("Judge", judge_decision)

    blocks.append("AGENT POOL")
    if not pool:
        blocks.append("<empty>")
    for item in pool:
        blocks.append("----")
        blocks.append(f"[{item['CandidateId']}]")
        blocks.append(f"TitleKey: {norm(item.get('TitleKey'))}")
        blocks.append(f"RaciTitle: {norm(item.get('RaciTitle'))}")
        for src in item.get("Sources") or []:
            blocks.append(
                f"Source={src.get('Agent')} Rank={src.get('Rank')} "
                f"Confidence={src.get('Confidence')} Why={norm(src.get('WhyPlausible'))}"
            )
    blocks.append("")
    blocks.append("DECISION REMINDER")
    blocks.append("- Prefer MATCH on the best available pool candidate when semantically credible.")
    blocks.append("- Use NO_MATCH only if all pool candidates are not credibly compatible.")
    blocks.append("- If forced to choose among imperfect options, pick closest available and explain trade-offs.")
    blocks.append("")
    blocks.append("Return JSON only.")
    return "\n".join(blocks)


def call_recovery_agent(
    model: str,
    stage: str,
    task: Dict[str, Any],
    mdr_ctx: Dict[str, Any],
    gpt_decision: Optional[Dict[str, Any]],
    claude_decision: Optional[Dict[str, Any]],
    judge_decision: Optional[Dict[str, Any]],
    pool: List[Dict[str, Any]],
) -> Dict[str, Any]:
    user_prompt = build_user_prompt(
        stage=stage,
        task=task,
        mdr_ctx=mdr_ctx,
        gpt_decision=gpt_decision,
        claude_decision=claude_decision,
        judge_decision=judge_decision,
        pool=pool,
    )
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "recovery_agent_evaluation",
                "schema": RECOVERY_SCHEMA,
                "strict": True,
            }
        },
    )
    return json.loads(resp.output_text)


def _expand_candidate_ids_in_reasoning(text: str, id_map: Dict[str, Dict[str, Any]]) -> str:
    if not text:
        return text

    def _label_for(cid: str) -> str:
        c = id_map.get(cid.upper()) or {}
        label = norm(c.get("RaciTitle")) or norm(c.get("TitleKey")) or cid
        return f"[{label}]"

    text = re.sub(r"\[(C\d{2})\]", lambda m: _label_for(m.group(1)), text, flags=re.IGNORECASE)
    text = re.sub(r"\b(C\d{2})\b", lambda m: _label_for(m.group(1)), text, flags=re.IGNORECASE)
    return text


def validate_recovery_output(
    stage: str,
    result: Dict[str, Any],
    pool: List[Dict[str, Any]],
) -> Dict[str, Any]:
    id_map = {str(c["CandidateId"]).upper(): c for c in pool}

    decision_type = str(result["decision_type"]).strip().upper()
    selected_candidate_id = result.get("selected_candidate_id")
    confidence = float(result["confidence"])
    reasoning_summary = norm(result.get("reasoning_summary") or "")
    recovery_mode = norm(result.get("recovery_mode") or "")

    reasoning_summary = _expand_candidate_ids_in_reasoning(reasoning_summary, id_map)

    if confidence < 0:
        confidence = 0.0
    if confidence > 1:
        confidence = 1.0

    expected_modes = {
        STAGE_MANUAL_REVIEW: {
            "MATCH": "manual_review_forced_match",
            "NO_MATCH": "manual_review_forced_no_match",
        },
        STAGE_NO_MATCH: {
            "MATCH": "no_match_recovered_to_match",
            "NO_MATCH": "no_match_confirmed",
        },
    }

    if decision_type not in ("MATCH", "NO_MATCH"):
        raise ValueError(f"Invalid decision_type: {decision_type}")

    expected_mode = expected_modes[stage][decision_type]
    if recovery_mode != expected_mode:
        raise ValueError(
            f"Invalid recovery_mode for stage={stage}, decision_type={decision_type}: "
            f"expected {expected_mode}, got {recovery_mode!r}"
        )

    if decision_type == "NO_MATCH":
        if selected_candidate_id not in (None, "null"):
            raise ValueError("NO_MATCH requires selected_candidate_id = null")
        return {
            "RecoveryDecisionType": "NO_MATCH",
            "RecoveryTitleKey": None,
            "RecoveryRaciTitle": None,
            "RecoveryConfidence": confidence,
            "RecoveryReason": reasoning_summary or "No candidate in the agent pool was sufficiently credible.",
            "RecoveryMode": recovery_mode,
        }

    if not selected_candidate_id:
        raise ValueError("MATCH requires selected_candidate_id")
    selected_candidate_id = str(selected_candidate_id).strip().upper()
    if selected_candidate_id not in id_map:
        raise ValueError(f"selected_candidate_id not in agent pool: {selected_candidate_id}")

    candidate = id_map[selected_candidate_id]
    return {
        "RecoveryDecisionType": "MATCH",
        "RecoveryTitleKey": norm(candidate["TitleKey"]),
        "RecoveryRaciTitle": norm(candidate["RaciTitle"]),
        "RecoveryConfidence": confidence,
        "RecoveryReason": reasoning_summary or "Selected the best supported candidate from the agent pool.",
        "RecoveryMode": recovery_mode,
    }


def save_recovery_result(
    con: duckdb.DuckDBPyConnection,
    task: Dict[str, Any],
    stage: str,
    model: str,
    candidate_pool_type: str,
    candidate_pool_size: int,
    result: Dict[str, Any],
) -> None:
    ts = now_ts_naive_utc()
    con.execute(
        f"""
        INSERT INTO {RECOVERY_RESULTS_TABLE}
          (TaskId, Document_title, PromptVersion, EmbeddingModel,
           SourceFinalDecisionType, RecoveryStage, RecoveryAgentName, RecoveryModel,
           RecoveryDecisionType, RecoveryTitleKey, RecoveryRaciTitle,
           RecoveryConfidence, RecoveryReason, RecoveryMode,
           CandidatePoolType, CandidatePoolSize, CreatedAt, UpdatedAt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (TaskId, PromptVersion, EmbeddingModel, RecoveryStage) DO UPDATE SET
          Document_title = excluded.Document_title,
          SourceFinalDecisionType = excluded.SourceFinalDecisionType,
          RecoveryAgentName = excluded.RecoveryAgentName,
          RecoveryModel = excluded.RecoveryModel,
          RecoveryDecisionType = excluded.RecoveryDecisionType,
          RecoveryTitleKey = excluded.RecoveryTitleKey,
          RecoveryRaciTitle = excluded.RecoveryRaciTitle,
          RecoveryConfidence = excluded.RecoveryConfidence,
          RecoveryReason = excluded.RecoveryReason,
          RecoveryMode = excluded.RecoveryMode,
          CandidatePoolType = excluded.CandidatePoolType,
          CandidatePoolSize = excluded.CandidatePoolSize,
          UpdatedAt = excluded.UpdatedAt
        """,
        [
            task["TaskId"],
            task["Document_title"],
            task["PromptVersion"],
            task["EmbeddingModel"],
            task["FinalDecisionType"],
            stage,
            RECOVERY_AGENT_NAME,
            model,
            result["RecoveryDecisionType"],
            result["RecoveryTitleKey"],
            result["RecoveryRaciTitle"],
            result["RecoveryConfidence"],
            result["RecoveryReason"],
            result["RecoveryMode"],
            candidate_pool_type,
            candidate_pool_size,
            ts,
            ts,
        ],
    )


def process_one_task(
    con: duckdb.DuckDBPyConnection,
    task: Dict[str, Any],
    model: str,
) -> Optional[Dict[str, Any]]:
    stage = recovery_stage_for_final_decision(task["FinalDecisionType"])

    mdr_ctx = fetch_mdr_context(con, task["Document_title"])
    gpt_decision = load_agent_decision(con, task["TaskId"], AGENT1_NAME)
    claude_decision = load_agent_decision(con, task["TaskId"], AGENT2_NAME)
    judge_decision = load_agent_decision(con, task["TaskId"], JUDGE_AGENT_NAME)

    top3_gpt = load_agent_top_candidates(con, task["TaskId"], AGENT1_NAME)
    top3_claude = load_agent_top_candidates(con, task["TaskId"], AGENT2_NAME)
    pool = build_agent_pool(top3_gpt, top3_claude)

    if not pool:
        result = {
            "RecoveryDecisionType": "NO_MATCH",
            "RecoveryTitleKey": None,
            "RecoveryRaciTitle": None,
            "RecoveryConfidence": 0.0,
            "RecoveryReason": "No candidate available in the saved agent pool.",
            "RecoveryMode": (
                "manual_review_forced_no_match"
                if stage == STAGE_MANUAL_REVIEW
                else "no_match_confirmed"
            ),
        }
        save_recovery_result(
            con=con,
            task=task,
            stage=stage,
            model=model,
            candidate_pool_type=RECOVERY_CANDIDATE_POOL_TYPE,
            candidate_pool_size=0,
            result=result,
        )
        return result

    raw_result = call_recovery_agent(
        model=model,
        stage=stage,
        task=task,
        mdr_ctx=mdr_ctx,
        gpt_decision=gpt_decision,
        claude_decision=claude_decision,
        judge_decision=judge_decision,
        pool=pool,
    )
    validated = validate_recovery_output(stage=stage, result=raw_result, pool=pool)
    save_recovery_result(
        con=con,
        task=task,
        stage=stage,
        model=model,
        candidate_pool_type=RECOVERY_CANDIDATE_POOL_TYPE,
        candidate_pool_size=len(pool),
        result=validated,
    )
    return validated


def _increment_stats(
    stats: Dict[str, Any],
    stage: Optional[str] = None,
    decision_type: Optional[str] = None,
    error: bool = False,
) -> None:
    stats["processed"] = int(stats.get("processed", 0)) + 1
    if error:
        stats["errors"] = int(stats.get("errors", 0)) + 1
        return
    if stage:
        per_stage = stats.setdefault("per_stage", {})
        stage_map = per_stage.setdefault(stage, {"MATCH": 0, "NO_MATCH": 0})
        if decision_type in ("MATCH", "NO_MATCH"):
            stage_map[decision_type] = int(stage_map.get(decision_type, 0)) + 1


def _worker(
    task_queue: queue.Queue,
    model: str,
    print_lock: threading.Lock,
    total_tasks: int,
    completed_count: List[int],
    stats: Dict[str, Any],
) -> None:
    con = connect_motherduck()
    try:
        ensure_recovery_results_table(con)
        while True:
            task = task_queue.get()
            try:
                if task is None:
                    return
                stage = recovery_stage_for_final_decision(task["FinalDecisionType"])
                with print_lock:
                    print(
                        f"[{threading.current_thread().name}] "
                        f"{stage} ({task['FinalDecisionType']}) -> {task['Document_title']}"
                    )
                result = process_one_task(con, task, model)
                with print_lock:
                    if result:
                        _increment_stats(
                            stats,
                            stage=stage,
                            decision_type=result["RecoveryDecisionType"],
                        )
                        print(
                            f"  Saved -> {result['RecoveryDecisionType']} "
                            f"titlekey={result.get('RecoveryTitleKey')} "
                            f"mode={result['RecoveryMode']}"
                        )
            except Exception as e:
                with print_lock:
                    _increment_stats(stats, error=True)
                    print(f"  ERROR on task {task.get('TaskId')}: {e}")
            finally:
                task_queue.task_done()
                if task is not None:
                    with print_lock:
                        completed_count[0] += 1
                        n = completed_count[0]
                        print(f"Progress: {n}/{total_tasks} (remaining: {total_tasks - n})")
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["manual-review", "no-match", "both"],
        default="manual-review",
        help="Which final-result category to recover.",
    )
    ap.add_argument(
        "--prompt-version",
        default=None,
        help="PromptVersion to read from results (default: PROMPT_VERSION from config.txt).",
    )
    ap.add_argument(
        "--embedding-model",
        default=None,
        help="Optional EmbeddingModel filter.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help=f"OpenAI model for the recovery agent (default: {DEFAULT_MODEL}).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of tasks to process.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers.",
    )
    ap.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Reprocess tasks even if a recovery result already exists for the same stage.",
    )
    args = ap.parse_args()

    prompt_version = args.prompt_version or _cfg("PROMPT_VERSION")
    if not prompt_version:
        raise RuntimeError("Specificare --prompt-version o impostare PROMPT_VERSION in config.txt")
    model = args.model or DEFAULT_MODEL

    con = connect_motherduck()
    try:
        ensure_recovery_results_table(con)
        tasks = fetch_tasks(
            con=con,
            mode=args.mode,
            prompt_version=prompt_version,
            embedding_model=args.embedding_model,
            limit=args.limit,
            rerun_existing=args.rerun_existing,
        )
    finally:
        con.close()

    if not tasks:
        print("No tasks found for recovery.")
        return

    print(f"Tasks to process: {len(tasks)}")
    task_queue: queue.Queue = queue.Queue()
    for task in tasks:
        task_queue.put(task)

    print_lock = threading.Lock()
    completed_count: List[int] = [0]
    stats: Dict[str, Any] = {"processed": 0, "errors": 0, "per_stage": {}}
    workers: List[threading.Thread] = []
    n_workers = max(1, int(args.workers or 1))
    for i in range(n_workers):
        t = threading.Thread(
            target=_worker,
            args=(task_queue, model, print_lock, len(tasks), completed_count, stats),
            name=f"recovery-{i+1}",
            daemon=True,
        )
        workers.append(t)
        t.start()

    for _ in workers:
        task_queue.put(None)

    try:
        task_queue.join()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return

    print("\nRecovery summary")
    print(f"- processed: {stats['processed']}")
    print(f"- errors: {stats['errors']}")
    for stage in (STAGE_MANUAL_REVIEW, STAGE_NO_MATCH):
        stage_stats = stats["per_stage"].get(stage)
        if not stage_stats:
            continue
        print(
            f"- {stage}: MATCH={stage_stats.get('MATCH', 0)} "
            f"NO_MATCH={stage_stats.get('NO_MATCH', 0)}"
        )


if __name__ == "__main__":
    main()

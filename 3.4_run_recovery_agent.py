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
- Uses the union of GPT/Claude top-3 candidates already saved in
  MdrReconciliationAgentTopCandidates
- Also merges the top-N retrieval candidates from v_MdrReconciliationAgentInput
  into the same single-pass pool
- The model should prefer AGENT_TOP3 candidates by default and use RAG_FALLBACK
  candidates only when they are clearly better

Config:
- config.txt (MOTHERDUCK_*, OPENAI_API_KEY, PROMPT_VERSION)

Usage:
  python 3.4_run_recovery_agent.py --mode manual-review
  python 3.4_run_recovery_agent.py --mode no-match
  python 3.4_run_recovery_agent.py --mode both --limit 50 --workers 4
  python 3.4_run_recovery_agent.py --mode both --batch --limit 200
  python 3.4_run_recovery_agent.py --batch-collect
  --batch-collect reads all ids from .recovery_last_batch_metas.json
"""

import argparse
import json
import queue
import re
import threading
import tempfile
import time
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
from openai import OpenAI


CONFIG_PATH = Path(__file__).resolve().parent / "config.txt"
BATCH_META_FILE = Path(__file__).resolve().parent / ".recovery_last_batch_meta.json"
BATCH_METAS_FILE = Path(__file__).resolve().parent / ".recovery_last_batch_metas.json"
BATCH_ENDPOINT = "/v1/responses"
OPENAI_BATCH_INPUT_FILE_HARD_LIMIT_BYTES = 209_715_200
DEFAULT_BATCH_TARGET_BYTES = 180_000_000


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


DEFAULT_MODEL = "gpt-5"
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
DEFAULT_FALLBACK_TOP_N = 10
RAG_FALLBACK_POOL_PREFIX = "agent_top3_plus_rag_top"
MDR_AGENT_INPUT_VIEW = "my_db.mdr_reconciliation.v_MdrReconciliationAgentInput"

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


def make_batch_custom_id(task: Dict[str, Any]) -> str:
    return "||".join(
        [
            norm(task.get("TaskId")),
            norm(task.get("PromptVersion")),
            norm(task.get("EmbeddingModel")),
        ]
    )


def parse_batch_custom_id(custom_id: str) -> Dict[str, str]:
    parts = str(custom_id or "").split("||")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"Invalid batch custom_id: {custom_id!r}")
    return {
        "TaskId": parts[0],
        "PromptVersion": parts[1],
        "EmbeddingModel": parts[2],
    }


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


def fetch_task_by_identity(
    con: duckdb.DuckDBPyConnection,
    task_id: str,
    prompt_version: str,
    embedding_model: str,
) -> Optional[Dict[str, Any]]:
    row = con.execute(
        f"""
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
          r.FinalReason
        FROM {RESULTS_TABLE} r
        JOIN {TASKS_TABLE} t
          ON t.TaskId = r.TaskId
         AND t.PromptVersion = r.PromptVersion
         AND t.EmbeddingModel = r.EmbeddingModel
        WHERE t.TaskId = ?
          AND t.PromptVersion = ?
          AND t.EmbeddingModel = ?
        LIMIT 1
        """,
        [task_id, prompt_version, embedding_model],
    ).fetchone()
    if not row:
        return None
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
    ]
    return dict(zip(cols, row))


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


def load_rag_top_candidates(
    con: duckdb.DuckDBPyConnection,
    document_title: str,
    prompt_version: str,
    embedding_model: str,
    top_n: int,
) -> List[Dict[str, Any]]:
    rows = con.execute(
        f"""
        SELECT
          Rank,
          Similarity,
          TitleKey,
          RaciTitle,
          DisciplineName,
          TypeName,
          ChapterName
        FROM {MDR_AGENT_INPUT_VIEW}
        WHERE Document_title = ?
          AND PromptVersion = ?
          AND EmbeddingModel = ?
          AND Rank <= ?
        ORDER BY Rank
        """,
        [document_title, prompt_version, embedding_model, int(top_n)],
    ).fetchall()
    cols = [
        "Rank",
        "Similarity",
        "TitleKey",
        "RaciTitle",
        "DisciplineName",
        "TypeName",
        "ChapterName",
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
        item["InAgentTop3"] = True
    return merged


def build_expanded_pool(
    top3_gpt: List[Dict[str, Any]],
    top3_claude: List[Dict[str, Any]],
    rag_top_candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    base_pool = build_agent_pool(top3_gpt, top3_claude)
    seen = {norm(item.get("TitleKey")).lower() for item in base_pool if norm(item.get("TitleKey"))}
    expanded = []
    for item in base_pool:
        cloned = dict(item)
        cloned["Sources"] = list(item.get("Sources") or [])
        cloned["InAgentTop3"] = True
        expanded.append(cloned)

    for item in rag_top_candidates or []:
        key = norm(item.get("TitleKey"))
        if not key or key.lower() in seen:
            continue
        seen.add(key.lower())
        expanded.append(
            {
                "TitleKey": key,
                "RaciTitle": norm(item.get("RaciTitle")),
                "Sources": [
                    {
                        "Agent": "RAG_FALLBACK",
                        "Rank": item.get("Rank"),
                        "Confidence": item.get("Similarity"),
                        "WhyPlausible": "",
                        "DisciplineName": norm(item.get("DisciplineName")),
                        "TypeName": norm(item.get("TypeName")),
                        "ChapterName": norm(item.get("ChapterName")),
                    }
                ],
                "InAgentTop3": False,
            }
        )

    expanded.sort(
        key=lambda c: (
            0 if c.get("InAgentTop3") else 1,
            min(int(s.get("Rank") or 999) for s in c["Sources"]),
            c["TitleKey"],
        )
    )
    for idx, item in enumerate(expanded, 1):
        item["CandidateId"] = f"C{idx:02d}"
    return expanded


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


def _responses_batch_request_body(model: str, stage: str, user_prompt: str) -> Dict[str, Any]:
    system_prompt = build_system_prompt(stage)
    return {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "recovery_agent_evaluation",
                "schema": RECOVERY_SCHEMA,
                "strict": True,
            }
        },
    }


def _extract_output_text_from_batch_response_body(body: Dict[str, Any]) -> Optional[str]:
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


BASE_SYSTEM_PROMPT = """
You are a recovery agent for MDR-to-RACI reconciliation.

You are invoked only after the main reconciliation pipeline has already run.
Your task is to re-evaluate difficult cases using ONLY the candidate pool already
proposed by the two primary agents.

You will receive:
- one historical MDR title with normalized metadata
- the current final outcome from the baseline pipeline
- GPT and Claude decisions/reasoning
- GPT and Claude top candidates
- a single merged candidate pool that includes:
  * candidates already present in the GPT/Claude top-3 union, tagged as [AGENT_TOP3]
  * additional retrieval candidates up to top-N, tagged as [RAG_FALLBACK]

Important:
- Do NOT invent new candidates.
- Do NOT use any candidates outside the provided AGENT POOL.
- You MUST return only MATCH or NO_MATCH.
- If you choose MATCH, the selected candidate must be one of the candidate IDs in the AGENT POOL.
- If you choose NO_MATCH, selected_candidate_id must be null.
- Prefer [AGENT_TOP3] candidates by default.
- Select a [RAG_FALLBACK] candidate only when it is clearly a better semantic fit than every plausible [AGENT_TOP3] candidate.

Decision process:
1. Start from the PRIMARY SUBJECT / ASSET / SYSTEM / ACTIVITY named by the MDR title.
2. If no candidate in the pool is really about that same primary subject, return NO_MATCH.
3. If one or more candidates preserve that subject, prefer MATCH even with secondary mismatches.
4. Then apply the strict exceptions below before using tie-breakers.

Core policy:
- Prefer MATCH whenever a candidate preserves the MDR title's core subject / functional asset, even with secondary mismatches in:
  * document sub-type,
  * construction method,
  * scope breadth,
  * discipline tag.
- Use NO_MATCH when every candidate fails on the primary subject/asset itself.
- By default, rank candidates in this order:
  1) core semantic intent / functional asset,
  2) package or system anchor when explicitly named,
  3) discipline/chapter coherence,
  4) document type compatibility.
- Do NOT reject a candidate only because:
  * it is broader or narrower in scope,
  * it is a specification instead of a drawing / data sheet / report (or vice versa),
  * it is a "typical" or "standard" version of a site-specific item,
  * it is a collection/vendor-docs bundle instead of a single document,
  * the construction method differs.

Strict exceptions (these override the general bias toward MATCH):

A) Documentary-function strictness.
- Some titles are primarily about the DOCUMENTARY FUNCTION itself, not just the asset.
- Strong documentary families include:
  * register / certificates / certificate / certified record,
  * vendor list / document list / register / index / schedule / progress report / monthly report / plan,
  * dossier / data book / quality book / final book / vendor book / turnover book,
  * material certificates / EN10204 3.1 / inspection certificates / compliance certificates.
- For these families:
  * preserve BOTH the asset AND the documentary role whenever possible,
  * do NOT degrade too easily to generic data sheets, specs, drawings, architecture documents, or equipment lists,
  * if the pool contains only same-asset technical content but not the documentary function, be much more willing to return NO_MATCH.
- Important carve-outs:
  * ITP exception: an Inspection & Test Plan may still MATCH to an inspection data sheet or to a specification/procedure with explicit inspection and acceptance content when the same asset/system is preserved.
  * QDB / dossier exception: a Quality Data Book / dossier / turnover compilation may still MATCH to a vendor-documents bundle or other compiled documentation package when both the asset and the bundle/compilation nature are preserved.
  * IT / ICT plan exception: for system-level IT / ICT / communication planning documents, an architecture document remains an acceptable proxy only when it clearly represents the same system-level blueprint/structure.

B) Execution-document strictness.
- Be stricter for execution-grade families where the missing function materially changes the deliverable.
- This includes:
  * field test report / inspection report / test report / proof-test record / test execution record,
  * method statement / execution methodology / commissioning procedure / operating procedure / energization procedure.
- For these families:
  * do NOT degrade too easily to inspection data sheets/checklists, generic specifications, manuals, or as-builts,
  * preserve BOTH the asset AND the execution/report/procedure function,
  * do NOT extend the ITP exception automatically to field test reports, completed inspection reports, or method statements.

C) Drawing-role preservation.
- When the MDR explicitly asks for a drawing-family deliverable such as layout, general layout, arrangement, overview, routing, wiring diagram, section, detail, elevation, supporting-structure drawing, or area/general-arrangement piping drawing, preserve that functional drawing role more strictly.
- Prefer candidates that preserve BOTH the asset AND the requested drawing role.
- Be especially cautious with:
  * hook-up drawings,
  * rack-only single lines,
  * equipment data sheets,
  * manuals,
  * generic equipment layouts,
  * non-graphical documents.
- Building-specific and system-level layouts/diagrams should keep both the system/building context and the layout/diagram family whenever possible.
- A layout-only proxy can still be acceptable when it clearly preserves the same asset and drawing family better than all alternatives.
- But when the MDR explicitly asks for sections, do NOT degrade too easily to layouts/arrangements/plans unless the pool clearly lacks any closer graphical section/elevation/detail family and the asset match is otherwise very strong.
- Likewise, when the MDR explicitly asks for an asset-specific drawing/detail/layout, do NOT degrade too easily to as-built sets, vendor-document bundles, or manuals unless the title itself is already an as-built/vendor-document/manual family.
- For area/general-arrangement layouts, be cautious with interface-only or partial-scope proxies:
  * tie-ins layouts are usually narrower than a full area arrangement,
  * rack-only single lines are usually narrower than a full piping layout,
  * area layouts may be weaker than building-specific layouts when the building anchor is explicit.
- When the MDR explicitly asks for an area piping arrangement/layout/general arrangement, do NOT treat tie-ins layouts or rack-only single lines as acceptable proxies unless the title itself is explicitly limited to tie-ins or to a rack-only scope.

D) Revision / update strictness.
- When the MDR title is about a revision / update / modification of an existing controlled document, preserve that specific revision-controlled document family and scope.
- Do NOT force MATCH to a merely related neighboring specification unless it credibly preserves the same base-document identity and revision scope.

E) Generic-umbrella caution.
- Do NOT force MATCH when the candidate only preserves a broad engineering class but loses the requested working role or deliverable scope.
- Typical weak umbrella substitutions:
  * buffer/accumulator tank for actuated valves -> generic pressure vessels data sheet,
  * full power supply & distribution calculations -> simple load summary,
  * equipment-specific reliability report -> plant-level RAM report,
  * system/database setup -> coding-system design criteria,
  * combined support data sheet (spring + PTFE/Teflon) -> generic supports system spec,
  * control philosophy -> procurement/supply specification,
  * study specification / terms of reference -> completed study report,
  * equipment/package study -> generic vendor-docs bundle.
- Also be cautious with scope-loss substitutions such as:
  * general communication / infrastructure / system-level deliverable -> narrow telecom procurement specification,
  * general communication / infrastructure / system-level deliverable -> structured-cabling / LAN bid evaluation or similarly narrow subsystem procurement document,
  * full calculation package -> subsystem-only study or summary,
  * project-wide or temporary-works system document -> permanent building/subsystem artifact,
  * general study/specification title -> very narrow subsystem study in another discipline just because it shares the word "study",
  * umbrella multi-study specification / ToR -> single-study ToR unless the title itself clearly centers that one study.

Tie-breakers (use only after the rules above):

T1 - Package anchor dominance.
     When the MDR explicitly names a package / skid / unit / train / major equipment family, treat that anchor as a primary semantic constraint.
     Prefer a package-level vendor-docs/manuals bundle tied to that same named package over:
       - a sub-component vendor-docs bundle,
       - an inspection data sheet of a sub-component,
       - a generic same-discipline document that loses the named package.
     But this package-anchor preference does NOT override execution-document strictness: for lifting / installation / commissioning procedures or plans, do not accept vendor-doc bundles or manuals when they materially lose the requested procedure/plan role.
     Examples:
      * compressor skid vendor documentation set -> VENDOR DWGS AND DOCUMENTS FOR PACKAGE SYSTEMS,
       * gas turbine UPS commissioning procedure -> VENDOR DWGS AND DOCUMENTS FOR GAS TURBINES,
       * HRSG cable list -> VENDOR DWGS AND DOCUMENTS FOR HRSG AND WHRU,
       * seals gas recovery skid compressor study -> PACKAGE SYSTEMS vendor docs over reciprocating-compressor docs.

T2 - Discipline coherence.
     When two candidates preserve the same core asset and one is in the SAME discipline as the MDR while another is not, prefer the discipline-coherent candidate even if it is more generic.
     Example: Process pumps data sheet -> PROCESS DATA SHEET FOR ROTATING EQUIPMENT over a more specific Mechanical pump sheet.

T3 - System-level architecture over procurement spec.
     For general infrastructure / communication infrastructure / IT-ICT system-level planning titles, prefer an architecture document over a technical supply specification when both are plausible.

T4 - Combined-scope calculation reports.
     When the MDR is a CALCULATION REPORT / TECHNICAL REPORT combining two systems, prefer a generic calculation/report candidate that can plausibly cover both systems over:
       - a design criteria document,
       - a single-system calculation report.
     Example: sanitary sewage + rainwater drainage -> PLUMBING CALCULATION REPORT.
     But do NOT degrade a clearly broad calculation package to a simple load summary or to a calculation that covers only one narrow subsystem unless nothing broader exists at all.

T5 - Micro-domain cautions.
     Apply these only when directly relevant:
       - canopy / shelter / enclosure are NOT interchangeable unless the functional asset and structural role are genuinely aligned,
       - near-duplicate façade precast panel titles should be treated consistently when the available pool quality is materially the same,
       - for façade precast panel drawings, if the pool lacks any genuine façade / precast / architectural-panel drawing, generic reinforced-concrete building/foundation drawings are usually too weak and NO_MATCH is generally preferred,
       - apply that façade-panel policy consistently across sibling façade panel titles under materially equivalent pools,
       - a noise study specification can acceptably map to a noise-control technical specification,
       - a general communication/system-infrastructure title should not collapse too quickly to a narrow telecom supply spec or to a structured-cabling/LAN bid-evaluation style document when the broader system scope is materially lost,
       - control philosophy should prefer philosophy / narratives / architecture style documents over supply specifications when both are plausible,
       - study specifications / terms of reference should prefer specification-like documents over completed reports when both are plausible,
       - for study/specification titles, do not prefer a much narrower or cross-discipline study merely because it preserves the word "study" if it materially loses the main subject scope,
       - for umbrella study/specification titles, do not narrow to a single-study ToR/report unless the MDR clearly signals that one study as the dominant scope,
       - for spare-parts / commissioning-start-up list titles, preserve the spare-parts list role and lifecycle phase; generic equipment lists, vendor-doc bundles, or neighboring-discipline spare-parts lists are usually too weak,
       - for lifting / installation procedure-plan titles, preserve the site-execution procedure/plan role; manuals or generic installation documents are usually too weak even when the equipment family is related,
       - when the MDR title explicitly names two co-equal assets/components in the same deliverable, do not degrade to a candidate that clearly covers only one of them unless the other is plainly accessory or implied by the same document family.

Only confirm NO_MATCH when the pool lacks a credible candidate after applying the strict exceptions and tie-breakers above.

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

MANUAL_REVIEW_SYSTEM_ADDON = """

Manual-review resolver preference:
- In manual_review_resolver mode, when more than one candidate remains plausible, prefer the least-risk title family that a document controller would most likely have intended, rather than an over-specialized or over-detailed variant.
- In practice, this means:
  * prefer a semantically close standard/general family over a narrower special/critical-item family unless the MDR explicitly signals that special/critical scope,
  * prefer the same requested view/working level (layout, plan, general arrangement, bulk material list, reinforcement details, foundation plan) over a more detailed or differently purposed neighboring document,
  * when a direct family-level match exists, do not over-penalize it just because another candidate is more specific but also more interpretive or lower-level,
  * if multiple upstream agents converge on the same plausible candidate, treat that as positive evidence unless a clearly better same-subject title exists,
  * for building/civil/structural titles, prefer ordinary building-level RC/steel families over special/critical-item families unless the MDR explicitly names a special/critical item,
  * for building/civil/structural titles, if no exact same-building title exists but an ordinary reinforcement/foundation/steel drawing family remains a plausible building-level proxy, prefer that least-risk family-level match over NO_MATCH,
  * for panel/switchboard titles, prefer the candidate preserving the panel/switchboard family and documentary role over one that preserves only a technology qualifier,
  * for underground/network/layout titles, prefer the direct network/layout family over neighboring plot-plan or subsystem-detail families,
  * when the MDR names a combined or neighboring pair of systems/components in one deliverable and no candidate cleanly covers both, you may still choose the least-risk family-level candidate that preserves the requested document role and the dominant shared context instead of forcing NO_MATCH,
  * in manual review, strictness rules are still important, but they should not eliminate an otherwise credible best-available candidate when the remaining mismatch is only partial scope loss rather than a true subject change.
"""


NO_MATCH_SYSTEM_ADDON = """

No-match recovery preference:
- In no_match_recovery mode, use a more conservative threshold before overturning an existing NO_MATCH.
- Prefer MATCH only when one candidate is clearly credible on the same primary subject after applying the strict exceptions above.
- If the pool offers only partial, neighboring, or role-shifted proxies, confirm NO_MATCH rather than forcing a recovery.
"""


def build_system_prompt(stage: str) -> str:
    if stage == STAGE_MANUAL_REVIEW:
        return BASE_SYSTEM_PROMPT + MANUAL_REVIEW_SYSTEM_ADDON
    if stage == STAGE_NO_MATCH:
        return BASE_SYSTEM_PROMPT + NO_MATCH_SYSTEM_ADDON
    return BASE_SYSTEM_PROMPT


def build_user_prompt(
    stage: str,
    task: Dict[str, Any],
    mdr_ctx: Dict[str, Any],
    gpt_decision: Optional[Dict[str, Any]],
    claude_decision: Optional[Dict[str, Any]],
    judge_decision: Optional[Dict[str, Any]],
    pool: List[Dict[str, Any]],
    fallback_top_n: int = DEFAULT_FALLBACK_TOP_N,
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
        origin_tag = "[AGENT_TOP3]" if item.get("InAgentTop3", True) else "[RAG_FALLBACK]"
        blocks.append(f"[{item['CandidateId']}] {origin_tag}")
        blocks.append(f"TitleKey: {norm(item.get('TitleKey'))}")
        blocks.append(f"RaciTitle: {norm(item.get('RaciTitle'))}")
        for src in item.get("Sources") or []:
            extra = ""
            if src.get("DisciplineName") or src.get("TypeName") or src.get("ChapterName"):
                extra = (
                    f" Discipline={norm(src.get('DisciplineName'))}"
                    f" Type={norm(src.get('TypeName'))}"
                    f" Chapter={norm(src.get('ChapterName'))}"
                )
            blocks.append(
                f"Source={src.get('Agent')} Rank={src.get('Rank')} "
                f"Confidence={src.get('Confidence')} Why={norm(src.get('WhyPlausible'))}{extra}"
            )
    blocks.append("")
    blocks.append("DECISION REMINDER")
    blocks.append("- First decide whether any candidate truly preserves the MDR's main asset/system/activity. If not, return NO_MATCH.")
    blocks.append("- Keep a strong bias toward MATCH only after a same-subject candidate exists; then accept secondary mismatches in subtype, scope, or construction method.")
    blocks.append("- Before stretching, check whether the title belongs to a strict family: documentary-function, execution-grade, revision/update, drawing-role, broad system/package scope, or a role-sensitive study/philosophy/specification title. Those families require closer preservation of the requested function and scope.")
    blocks.append("- If the title explicitly names a package/skid/unit, prefer candidates that preserve that same package anchor over generic same-discipline documents.")
    blocks.append("- Be cautious with narrow proxies for broad asks: telecom/structured-cabling procurement docs for general infrastructure, load summaries for full calculation packages, and tie-ins/rack-only drawings for full area arrangements are often too weak.")
    blocks.append("- Be cautious with role drift: control philosophy -> supply spec, study specification -> report, single-study ToR for an umbrella study spec, asset-specific drawings -> as-built bundles, section drawings -> layouts, and façade panel titles with equivalent pools should usually be decided consistently.")
    blocks.append("- If the title explicitly contains two co-equal assets/components, prefer candidates that plausibly cover both; a single-asset candidate is usually too weak unless the other item is clearly accessory.")
    blocks.append("- For study/specification titles, do not prefer a much narrower or cross-discipline study only because it contains the word 'study'.")
    blocks.append("- For façade precast panel drawings without any true façade/precast proxy in pool, generic RC building/foundation drawings are usually too weak, so prefer NO_MATCH.")
    blocks.append("- For spare-parts lists and lifting/installation procedure-plans, preserve that operational role directly; generic manuals, vendor bundles, or generic lists are usually too weak.")
    if stage == STAGE_MANUAL_REVIEW:
        blocks.append("- In manual review, when several candidates are plausible, lean toward the least-risk standard/general family that preserves the intended document level instead of a more specialized or over-detailed variant.")
        blocks.append("- In manual review, if no candidate cleanly covers every combined sub-scope, do not force NO_MATCH too early: a best-available family-level layout/drawing/list/spec can still be acceptable when it preserves the main shared context and document role better than the rest of the pool.")
        blocks.append("- In manual review building/civil/structural cases, if the pool lacks the exact building title but contains a plausible ordinary reinforcement/foundation/steel family, prefer that ordinary building-level proxy over NO_MATCH or over a special/critical-item family.")
    elif stage == STAGE_NO_MATCH:
        blocks.append("- In no-match recovery, overturn the baseline NO_MATCH only when a candidate is clearly credible on the same primary subject; partial or role-shifted proxies should usually remain NO_MATCH.")
    blocks.append("- Prefer AGENT_TOP3 candidates by default; use a RAG_FALLBACK candidate only when it is clearly better on the same subject.")
    blocks.append("- Use tie-breakers only after the rules above, and explain the chosen trade-off briefly in reasoning_summary.")
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
    fallback_top_n: int = DEFAULT_FALLBACK_TOP_N,
) -> Dict[str, Any]:
    system_prompt = build_system_prompt(stage)
    user_prompt = build_user_prompt(
        stage=stage,
        task=task,
        mdr_ctx=mdr_ctx,
        gpt_decision=gpt_decision,
        claude_decision=claude_decision,
        judge_decision=judge_decision,
        pool=pool,
        fallback_top_n=fallback_top_n,
    )
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
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


def build_empty_pool_recovery_result(stage: str) -> Dict[str, Any]:
    return {
        "RecoveryDecisionType": "NO_MATCH",
        "RecoveryTitleKey": None,
        "RecoveryRaciTitle": None,
        "RecoveryConfidence": 0.0,
        "RecoveryReason": "No candidate available in the merged recovery pool.",
        "RecoveryMode": (
            "manual_review_forced_no_match"
            if stage == STAGE_MANUAL_REVIEW
            else "no_match_confirmed"
        ),
    }


def prepare_recovery_inputs(
    con: duckdb.DuckDBPyConnection,
    task: Dict[str, Any],
    fallback_top_n: int,
) -> Dict[str, Any]:
    stage = recovery_stage_for_final_decision(task["FinalDecisionType"])
    mdr_ctx = fetch_mdr_context(con, task["Document_title"])
    gpt_decision = load_agent_decision(con, task["TaskId"], AGENT1_NAME)
    claude_decision = load_agent_decision(con, task["TaskId"], AGENT2_NAME)
    judge_decision = load_agent_decision(con, task["TaskId"], JUDGE_AGENT_NAME)
    top3_gpt = load_agent_top_candidates(con, task["TaskId"], AGENT1_NAME)
    top3_claude = load_agent_top_candidates(con, task["TaskId"], AGENT2_NAME)
    rag_top_candidates = load_rag_top_candidates(
        con=con,
        document_title=task["Document_title"],
        prompt_version=task["PromptVersion"],
        embedding_model=task["EmbeddingModel"],
        top_n=fallback_top_n,
    )
    pool = build_expanded_pool(top3_gpt, top3_claude, rag_top_candidates)
    return {
        "stage": stage,
        "mdr_ctx": mdr_ctx,
        "gpt_decision": gpt_decision,
        "claude_decision": claude_decision,
        "judge_decision": judge_decision,
        "pool": pool,
        "candidate_pool_type": f"{RAG_FALLBACK_POOL_PREFIX}{fallback_top_n}",
    }


def save_recovery_result(
    con: duckdb.DuckDBPyConnection,
    task: Dict[str, Any],
    stage: str,
    model: str,
    candidate_pool_type: str,
    candidate_pool_size: int,
    result: Dict[str, Any],
    manage_transaction: bool = True,
) -> None:
    ts = now_ts_naive_utc()
    if manage_transaction:
        con.execute("BEGIN;")
    try:
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
        if manage_transaction:
            con.execute("COMMIT;")
    except Exception:
        if manage_transaction:
            con.execute("ROLLBACK;")
        raise


def process_one_task(
    con: duckdb.DuckDBPyConnection,
    task: Dict[str, Any],
    model: str,
    fallback_top_n: int,
) -> Optional[Dict[str, Any]]:
    prepared = prepare_recovery_inputs(con, task, fallback_top_n)
    stage = prepared["stage"]
    mdr_ctx = prepared["mdr_ctx"]
    gpt_decision = prepared["gpt_decision"]
    claude_decision = prepared["claude_decision"]
    judge_decision = prepared["judge_decision"]
    pool = prepared["pool"]
    candidate_pool_type = prepared["candidate_pool_type"]

    if not pool:
        result = build_empty_pool_recovery_result(stage)
        save_recovery_result(
            con=con,
            task=task,
            stage=stage,
            model=model,
            candidate_pool_type=candidate_pool_type,
            candidate_pool_size=0,
            result=result,
        )
        result["UsedCandidatePoolType"] = candidate_pool_type
        result["UsedCandidatePoolSize"] = 0
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
        fallback_top_n=fallback_top_n,
    )
    validated = validate_recovery_output(stage=stage, result=raw_result, pool=pool)
    candidate_pool_size = len(pool)

    save_recovery_result(
        con=con,
        task=task,
        stage=stage,
        model=model,
        candidate_pool_type=candidate_pool_type,
        candidate_pool_size=candidate_pool_size,
        result=validated,
    )
    validated["UsedCandidatePoolType"] = candidate_pool_type
    validated["UsedCandidatePoolSize"] = candidate_pool_size
    return validated


def write_batch_meta(meta: Dict[str, Any]) -> None:
    BATCH_META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    BATCH_METAS_FILE.write_text(json.dumps([meta], indent=2), encoding="utf-8")


def load_batch_metas() -> List[Dict[str, Any]]:
    if not BATCH_METAS_FILE.exists():
        return []
    try:
        metas = json.loads(BATCH_METAS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(metas, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in metas:
        if isinstance(item, dict) and str(item.get("batch_id") or "").strip():
            out.append(item)
    return out


def run_batch_submit(
    con: duckdb.DuckDBPyConnection,
    tasks: List[Dict[str, Any]],
    model: str,
    fallback_top_n: int,
    mode: str,
    prompt_version: str,
    embedding_model: Optional[str],
    rerun_existing: bool,
) -> Dict[str, Any]:
    return run_batch_submit_chunked(
        con=con,
        tasks=tasks,
        model=model,
        fallback_top_n=fallback_top_n,
        mode=mode,
        prompt_version=prompt_version,
        embedding_model=embedding_model,
        rerun_existing=rerun_existing,
        target_max_bytes=DEFAULT_BATCH_TARGET_BYTES,
    )


def run_batch_submit_chunked(
    con: duckdb.DuckDBPyConnection,
    tasks: List[Dict[str, Any]],
    model: str,
    fallback_top_n: int,
    mode: str,
    prompt_version: str,
    embedding_model: Optional[str],
    rerun_existing: bool,
    target_max_bytes: int,
) -> Dict[str, Any]:
    if target_max_bytes <= 0:
        raise ValueError("target_max_bytes must be > 0")
    if target_max_bytes > OPENAI_BATCH_INPUT_FILE_HARD_LIMIT_BYTES:
        raise ValueError(
            f"target_max_bytes cannot exceed hard limit {OPENAI_BATCH_INPUT_FILE_HARD_LIMIT_BYTES}"
        )

    current_lines: List[str] = []
    current_custom_ids: List[str] = []
    current_bytes = 0
    batch_metas: List[Dict[str, Any]] = []
    saved_without_batch = 0
    skipped_too_large = 0
    total_submitted = 0
    total_bytes = 0

    def flush_chunk() -> None:
        nonlocal current_lines, current_custom_ids, current_bytes
        if not current_lines:
            return
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            for line in current_lines:
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

        meta = {
            "batch_id": batch_id,
            "model": model,
            "fallback_top_n": int(fallback_top_n),
            "mode": mode,
            "prompt_version": prompt_version,
            "embedding_model": embedding_model,
            "rerun_existing": bool(rerun_existing),
            "submitted_count": len(current_custom_ids),
            "saved_without_batch": 0,
            "submitted_at": now_ts_naive_utc().isoformat(),
            "jsonl_bytes": current_bytes,
        }
        batch_metas.append(meta)
        # Persist after each submitted chunk so partial progress is recoverable on later failures.
        BATCH_META_FILE.write_text(json.dumps(batch_metas[-1], indent=2), encoding="utf-8")
        BATCH_METAS_FILE.write_text(json.dumps(batch_metas, indent=2), encoding="utf-8")
        print(
            f"Submitted chunk {len(batch_metas)}: batch_id={batch_id}, "
            f"requests={len(current_custom_ids)}, jsonl_bytes={current_bytes}"
        )
        current_lines = []
        current_custom_ids = []
        current_bytes = 0

    for task in tasks:
        prepared = prepare_recovery_inputs(con, task, fallback_top_n)
        stage = prepared["stage"]
        pool = prepared["pool"]
        candidate_pool_type = prepared["candidate_pool_type"]

        if not pool:
            result = build_empty_pool_recovery_result(stage)
            save_recovery_result(
                con=con,
                task=task,
                stage=stage,
                model=model,
                candidate_pool_type=candidate_pool_type,
                candidate_pool_size=0,
                result=result,
            )
            saved_without_batch += 1
            continue

        user_prompt = build_user_prompt(
            stage=stage,
            task=task,
            mdr_ctx=prepared["mdr_ctx"],
            gpt_decision=prepared["gpt_decision"],
            claude_decision=prepared["claude_decision"],
            judge_decision=prepared["judge_decision"],
            pool=pool,
            fallback_top_n=fallback_top_n,
        )
        body = _responses_batch_request_body(model, stage, user_prompt)
        custom_id = make_batch_custom_id(task)
        line = json.dumps(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": BATCH_ENDPOINT,
                "body": body,
            }
        )
        line_bytes = len((line + "\n").encode("utf-8"))
        if line_bytes > target_max_bytes:
            skipped_too_large += 1
            print(
                f"  {task['TaskId']}: skipped (single request {line_bytes} bytes exceeds target {target_max_bytes})"
            )
            continue
        if current_lines and (current_bytes + line_bytes > target_max_bytes):
            flush_chunk()
        current_lines.append(line)
        current_custom_ids.append(custom_id)
        current_bytes += line_bytes
        total_submitted += 1
        total_bytes += line_bytes

    flush_chunk()

    if not batch_metas:
        return {
            "batch_ids": [],
            "submitted_count": 0,
            "saved_without_batch": saved_without_batch,
        }
    avg_bytes = total_bytes / max(total_submitted, 1)
    estimated_per_batch = max(1, int(target_max_bytes // max(avg_bytes, 1)))
    estimated_batches = math.ceil(total_submitted / estimated_per_batch)
    print(
        "Batch sizing summary: "
        f"submitted_requests={total_submitted}, avg_request_bytes={avg_bytes:.0f}, "
        f"target_max_bytes={target_max_bytes}, estimated_requests_per_batch={estimated_per_batch}, "
        f"estimated_batches={estimated_batches}, submitted_batches={len(batch_metas)}, "
        f"skipped_too_large={skipped_too_large}"
    )
    BATCH_META_FILE.write_text(json.dumps(batch_metas[-1], indent=2), encoding="utf-8")
    BATCH_METAS_FILE.write_text(json.dumps(batch_metas, indent=2), encoding="utf-8")
    return {
        "batch_ids": [m["batch_id"] for m in batch_metas],
        "submitted_count": total_submitted,
        "saved_without_batch": saved_without_batch,
    }


def run_batch_collect(
    con: duckdb.DuckDBPyConnection,
    batch_id: str,
    model: str,
    fallback_top_n: int,
    poll_interval: int = 60,
) -> None:
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

    saved = 0
    errors = 0
    output_file_id = getattr(batch, "output_file_id", None) or (batch.get("output_file_id") if isinstance(batch, dict) else None)
    error_file_id = getattr(batch, "error_file_id", None) or (batch.get("error_file_id") if isinstance(batch, dict) else None)

    con.execute("BEGIN;")
    try:
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
                try:
                    identity = parse_batch_custom_id(str(custom_id))
                except Exception as e:
                    errors += 1
                    print(f"  {custom_id}: invalid custom_id ({e})")
                    continue
                output_text = _extract_output_text_from_batch_response_body(body) if body else None
                if not output_text:
                    errors += 1
                    print(f"  {identity['TaskId']}: no output text in response")
                    continue
                try:
                    raw_result = json.loads(output_text)
                except json.JSONDecodeError as e:
                    errors += 1
                    print(f"  {identity['TaskId']}: JSON parse error {e}")
                    continue

                task = fetch_task_by_identity(
                    con=con,
                    task_id=identity["TaskId"],
                    prompt_version=identity["PromptVersion"],
                    embedding_model=identity["EmbeddingModel"],
                )
                if not task:
                    errors += 1
                    print(f"  {identity['TaskId']}: task not found in DB")
                    continue

                prepared = prepare_recovery_inputs(con, task, fallback_top_n)
                pool = prepared["pool"]
                stage = prepared["stage"]
                candidate_pool_type = prepared["candidate_pool_type"]
                if not pool:
                    result = build_empty_pool_recovery_result(stage)
                    save_recovery_result(
                        con=con,
                        task=task,
                        stage=stage,
                        model=model,
                        candidate_pool_type=candidate_pool_type,
                        candidate_pool_size=0,
                        result=result,
                        manage_transaction=False,
                    )
                    saved += 1
                    print(f"  {identity['TaskId']} -> NO_MATCH (saved from empty pool)")
                    continue

                try:
                    validated = validate_recovery_output(stage=stage, result=raw_result, pool=pool)
                    save_recovery_result(
                        con=con,
                        task=task,
                        stage=stage,
                        model=model,
                        candidate_pool_type=candidate_pool_type,
                        candidate_pool_size=len(pool),
                        result=validated,
                        manage_transaction=False,
                    )
                    saved += 1
                    print(f"  {identity['TaskId']} -> {validated['RecoveryDecisionType']} (saved)")
                except Exception as e:
                    errors += 1
                    print(f"  {identity['TaskId']}: validation/save error {e}")

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
                    try:
                        identity = parse_batch_custom_id(str(custom_id))
                        print(f"  {identity['TaskId']}: batch error (see error file)")
                    except Exception:
                        print(f"  {custom_id}: batch error (see error file)")
                    errors += 1
        con.execute("COMMIT;")
    except Exception:
        con.execute("ROLLBACK;")
        raise

    print(f"Batch collect done: {saved} saved, {errors} errors.")


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
    fallback_top_n: int,
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
                result = process_one_task(con, task, model, fallback_top_n)
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
                            f"mode={result['RecoveryMode']} "
                            f"pool={result.get('UsedCandidatePoolType')}"
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
    ap.add_argument(
        "--fallback-top-n",
        type=int,
        default=DEFAULT_FALLBACK_TOP_N,
        help=(
            "Always merge top-N RAG candidates into the single-pass recovery pool, "
            "tagged as RAG_FALLBACK, while keeping AGENT_TOP3 candidates as the preferred "
            f"source by policy (default: {DEFAULT_FALLBACK_TOP_N})."
        ),
    )
    ap.add_argument(
        "--batch",
        action="store_true",
        help="Use OpenAI Batch API: submit recovery tasks and exit.",
    )
    ap.add_argument(
        "--batch-collect",
        action="store_true",
        help="Poll and collect all batch ids saved by latest submit in .recovery_last_batch_metas.json.",
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
        "--poll-interval",
        type=int,
        default=60,
        help="Polling interval in seconds for --batch-collect.",
    )
    args = ap.parse_args()

    if args.batch and args.batch_collect:
        raise RuntimeError("Use either --batch or --batch-collect, not both.")

    prompt_version = args.prompt_version or _cfg("PROMPT_VERSION")
    if not prompt_version:
        raise RuntimeError("Specificare --prompt-version o impostare PROMPT_VERSION in config.txt")
    model = args.model or DEFAULT_MODEL

    if args.batch_collect:
        metas = load_batch_metas()
        if not metas:
            raise RuntimeError("Error: no .recovery_last_batch_metas.json file or no valid batch ids.")
        print(f"Collecting {len(metas)} recovery batch(es) from {BATCH_METAS_FILE.name}...")
        con = connect_motherduck()
        try:
            ensure_recovery_results_table(con)
            for i, meta in enumerate(metas, start=1):
                batch_id = str(meta.get("batch_id") or "").strip()
                if not batch_id:
                    continue
                collect_model = str(meta.get("model") or model)
                collect_fallback_top_n = int(meta.get("fallback_top_n") or args.fallback_top_n)
                print(f"[{i}/{len(metas)}] Collecting recovery batch: {batch_id}")
                run_batch_collect(
                    con=con,
                    batch_id=batch_id,
                    model=collect_model,
                    fallback_top_n=collect_fallback_top_n,
                    poll_interval=max(1, int(args.poll_interval or 60)),
                )
        finally:
            con.close()
        return

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

    if args.batch:
        print(f"Tasks to submit in batch: {len(tasks)}")
        con = connect_motherduck()
        try:
            ensure_recovery_results_table(con)
            batch_info = run_batch_submit_chunked(
                con=con,
                tasks=tasks,
                model=model,
                fallback_top_n=args.fallback_top_n,
                mode=args.mode,
                prompt_version=prompt_version,
                embedding_model=args.embedding_model,
                rerun_existing=args.rerun_existing,
                target_max_bytes=args.batch_max_bytes,
            )
        finally:
            con.close()
        if batch_info["saved_without_batch"]:
            print(
                f"Saved immediately without batch: {batch_info['saved_without_batch']} "
                f"(empty merged pool)."
            )
        if batch_info["batch_ids"]:
            if len(batch_info["batch_ids"]) == 1:
                print(f"Batch submitted: {batch_info['batch_ids'][0]}")
            else:
                print(
                    f"Batches submitted: {len(batch_info['batch_ids'])} "
                    f"(last={batch_info['batch_ids'][-1]})."
                )
            print(f"Submitted requests: {batch_info['submitted_count']}")
            print("Run with --batch-collect later to write batch results to DB.")
        else:
            print("No LLM batch submitted; only deterministic empty-pool tasks were saved.")
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
            args=(
                task_queue,
                model,
                args.fallback_top_n,
                print_lock,
                len(tasks),
                completed_count,
                stats,
            ),
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

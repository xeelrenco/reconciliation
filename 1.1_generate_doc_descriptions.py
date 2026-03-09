#!/usr/bin/env python3
"""
Generate standardized descriptions for document titles (from mdr_reconciliation.v_DocumentsEnriched)
and store them into mdr_reconciliation.DocumentTitleDescriptions (MotherDuck / DuckDB).

Parallel version: fetches pending rows, processes in batches via AsyncOpenAI with concurrency,
upserts results in bulk.

Config: tutte le variabili in config.txt (stessa cartella dello script). Vedi config.example.txt.
"""

import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
from openai import AsyncOpenAI

# -----------------------------
# Config da file di testo
# -----------------------------
CONFIG_PATH = Path(__file__).resolve().parent / "config.txt"


def load_config(path: Optional[Path] = None) -> Dict[str, str]:
    """
    Legge un file di testo con righe chiave=valore.
    Ignora righe vuote e righe che iniziano con #.
    """
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
    """Carica la config una sola volta (cached nel modulo)."""
    if not hasattr(get_config, "_cache"):
        get_config._cache = load_config()  # type: ignore[attr-defined]
    return get_config._cache  # type: ignore[attr-defined]


def _cfg(key: str, default: Optional[str] = None) -> str:
    val = get_config().get(key, default or "")
    return val.strip()


PROMPT_VERSION = _cfg("PROMPT_VERSION", "v1")
MODEL_NAME = _cfg("LLM_MODEL", "gpt-4o-mini")

BATCH_SIZE = int(_cfg("BATCH_SIZE", "25"))
CONCURRENCY = int(_cfg("CONCURRENCY", "6"))
SLEEP_S = float(_cfg("SLEEP_S", "0.0"))
MAX_RETRIES = int(_cfg("MAX_RETRIES", "3"))

client = AsyncOpenAI(api_key=_cfg("OPENAI_API_KEY"))

# Schema: il modello deve ritornare una lista di risultati, uno per TitleKey.
BATCH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title_key": {"type": "string"},
                    "description": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
                    "scope": {"type": "string"},
                    "exclusions": {"type": "string"},
                },
                "required": ["title_key", "description", "keywords", "scope", "exclusions"],
            },
        }
    },
    "required": ["items"],
}

SYSTEM_MSG = """You are a senior document control specialist working for Renco, an EPC contractor active in industrial and energy projects.

Renco operates in Engineering, Procurement and Construction (EPC) environments.
The documents analyzed belong to a structured RACI matrix used to define responsibilities and expected deliverables across engineering disciplines.

Each title represents a canonical document type that is part of an engineering project lifecycle (design, procurement, construction, commissioning, as-built, etc.).

Your task is to generate a standardized, discipline-aware description that:

- Clearly states the purpose of the document.
- Explains what type of technical content it typically contains.
- Is suitable for semantic matching against real MDR (Master Document Register) historical titles.
- Is neutral, formal, and aligned with EPC project documentation standards.
- Does NOT invent project-specific details.
- Does NOT refer to RACI, AI, or internal processes.
- Uses only the provided human-readable discipline/type/category labels (never codes).

The output will later be used by other AI agents to:
- Match historical MDR titles to standard RACI titles.
- Evaluate semantic similarity.
- Support document reconciliation activities.

Therefore:
- Be precise but not overly detailed.
- Avoid speculation.
- Avoid marketing language.
- Avoid assumptions not supported by the input."""

PROMPT_TEMPLATE_BATCH = """Generate standardized descriptions for the following canonical document titles used in Renco EPC projects.

Follow strictly the system instructions.

Each item must be described independently.

Return JSON according to the provided schema.

RULES (apply to every item):
- Output MUST be valid JSON and ONLY JSON (no markdown, no commentary).
- Each "description" must be 1–2 sentences, 20–40 words total.
- Do NOT invent details not implied by the inputs.
- Mention purpose and typical contents, aligned with Discipline name, Type name, Chapter name and Category description if provided.
- Avoid buzzwords and avoid mentioning "RACI", "matrix", "agent", "AI".
- Do NOT output any codes; use only the provided human-readable labels.

Return JSON with the following schema:
{{
  "items": [
    {{
      "title_key": string,
      "description": string,
      "keywords": [string, ... up to 8],
      "scope": string,          // max 12 words
      "exclusions": string      // max 12 words
    }}, ...
  ]
}}

EXAMPLES (style and level of detail reference):

{examples_block}

INPUT ITEMS:
{items_block}
"""

EXAMPLES_BLOCK = """{
  "items": [
    {
      "title_key": "EXAMPLE_1",
      "description": "As-built record for rotary compressors in process service, documenting final mechanical configuration, piping connections, equipment tags, and references to inspections and performance tests executed during construction and commissioning.",
      "keywords": ["as-built", "rotary compressors", "final configuration", "piping connections", "inspections", "commissioning"],
      "scope": "Final configuration and executed test references",
      "exclusions": "Does not include preliminary design criteria"
    },
    {
      "title_key": "EXAMPLE_2",
      "description": "Technical specification for the supply and installation of low voltage switchboards, defining performance requirements, applicable standards, materials, factory acceptance tests, and vendor documentation deliverables.",
      "keywords": ["technical specification", "low voltage switchboards", "performance requirements", "factory acceptance test", "vendor documentation"],
      "scope": "Technical requirements and supply documentation",
      "exclusions": "Does not include detailed construction drawings"
    },
    {
      "title_key": "EXAMPLE_3",
      "description": "Operational procedure for performing pressure testing on process piping, outlining preparation steps, acceptance criteria, recording of results, and responsibilities during the construction phase.",
      "keywords": ["procedure", "pressure testing", "process piping", "acceptance criteria", "construction phase"],
      "scope": "Execution method and acceptance criteria",
      "exclusions": "Does not include pipeline design calculations"
    }
  ]
}"""

# -----------------------------
# MotherDuck / DuckDB
# -----------------------------
def connect_motherduck() -> duckdb.DuckDBPyConnection:
    token = _cfg("MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError("Manca MOTHERDUCK_TOKEN nel file di configurazione (config.txt)")
    dbname = _cfg("MOTHERDUCK_DB", "renco")
    return duckdb.connect(f"md:{dbname}?token={token}")


def now_ts_naive_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS mdr_reconciliation.DocumentTitleDescriptions (
          TitleKey      VARCHAR NOT NULL,
          PromptVersion VARCHAR NOT NULL,
          Model         VARCHAR,
          Description   VARCHAR NOT NULL,
          KeywordsJson  VARCHAR,
          Scope         VARCHAR,
          Exclusions    VARCHAR,
          Status        VARCHAR NOT NULL,
          Error         VARCHAR,
          CreatedAt     TIMESTAMP NOT NULL,
          UpdatedAt     TIMESTAMP NOT NULL,
          PRIMARY KEY (TitleKey, PromptVersion)
        );
        """
    )


def fetch_pending(con: duckdb.DuckDBPyConnection, limit: int) -> List[Dict[str, Any]]:
    """
    Pull records that do not yet have generated/reviewed/approved for this version.
    Reads from the enriched view (human-friendly values).
    """
    q = f"""
    SELECT
      e.TitleKey,
      e.Title,
      e.DisciplineName,
      e.TypeName,
      e.CategoryDescription,
      e.ChapterName,
      e.Scalable
    FROM mdr_reconciliation.v_DocumentsEnriched e
    LEFT JOIN mdr_reconciliation.DocumentTitleDescriptions t
      ON t.TitleKey = e.TitleKey
     AND t.PromptVersion = ?
     AND t.Status IN ('generated','reviewed','approved')
    WHERE t.TitleKey IS NULL
    ORDER BY e.TitleKey
    LIMIT {limit}
    """
    rows = con.execute(q, [PROMPT_VERSION]).fetchall()
    cols = ["TitleKey", "Title", "DisciplineName", "TypeName", "CategoryDescription", "ChapterName", "Scalable"]
    return [dict(zip(cols, r)) for r in rows]


def upsert_many(con: duckdb.DuckDBPyConnection, results: List[Dict[str, Any]]) -> None:
    """Bulk UPSERT in a single transaction using executemany."""
    ts = now_ts_naive_utc()
    con.execute("BEGIN;")
    try:
        con.executemany(
            """
            INSERT INTO mdr_reconciliation.DocumentTitleDescriptions
              (TitleKey, PromptVersion, Model, Description, KeywordsJson, Scope, Exclusions,
               Status, Error, CreatedAt, UpdatedAt)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (TitleKey, PromptVersion) DO UPDATE SET
              Model       = excluded.Model,
              Description = excluded.Description,
              KeywordsJson= excluded.KeywordsJson,
              Scope       = excluded.Scope,
              Exclusions  = excluded.Exclusions,
              Status      = excluded.Status,
              Error       = excluded.Error,
              UpdatedAt   = excluded.UpdatedAt
            """,
            [
                (
                    r["TitleKey"], PROMPT_VERSION, MODEL_NAME,
                    r.get("Description", ""),
                    r.get("KeywordsJson", ""),
                    r.get("Scope", ""),
                    r.get("Exclusions", ""),
                    r.get("Status", "generated"),
                    r.get("Error"),
                    ts, ts
                )
                for r in results
            ],
        )
        con.execute("COMMIT;")
    except Exception:
        con.execute("ROLLBACK;")
        raise


# -----------------------------
# Prompt / Validation
# -----------------------------
def items_block(rows: List[Dict[str, Any]]) -> str:
    lines = []
    for r in rows:
        scalable = "" if r.get("Scalable") is None else ("true" if bool(r["Scalable"]) else "false")
        lines.append(
            "----\n"
            f"title_key: {r['TitleKey']}\n"
            f"Title: {r.get('Title','')}\n"
            f"Discipline name: {r.get('DisciplineName','')}\n"
            f"Type name: {r.get('TypeName','')}\n"
            f"Category description: {r.get('CategoryDescription','')}\n"
            f"Chapter name: {r.get('ChapterName','')}\n"
            f"Scalable: {scalable}\n"
        )
    return "\n".join(lines)


def soft_trim_words(text: str, max_words: int) -> str:
    parts = str(text).split()
    return " ".join(parts[:max_words]) if len(parts) > max_words else str(text)


def normalize_result(item: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    warn = None
    desc = (item.get("description") or "").strip()
    if not desc:
        raise ValueError("Empty description")

    kws = item.get("keywords") or []
    kws = [str(k).strip() for k in kws if str(k).strip()]
    kws = kws[:8]

    scope = soft_trim_words((item.get("scope") or "").strip(), 12)
    excl = soft_trim_words((item.get("exclusions") or "").strip(), 12)

    wcount = len(desc.split())
    if wcount < 15 or wcount > 55:
        warn = f"WARN: description word count {wcount} outside target"

    out = {
        "TitleKey": str(item["title_key"]),
        "Description": desc,
        "KeywordsJson": json.dumps(kws, ensure_ascii=False),
        "Scope": scope,
        "Exclusions": excl,
        "Status": "generated",
        "Error": warn,
    }
    return out, warn


# -----------------------------
# OpenAI batch call (async)
# -----------------------------
async def call_llm_batch(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt = PROMPT_TEMPLATE_BATCH.format(examples_block=EXAMPLES_BLOCK, items_block=items_block(rows))
    resp = await client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "document_title_description_batch",
                "schema": BATCH_SCHEMA,
                "strict": True,
            }
        },
    )
    return json.loads(resp.output_text)


async def process_one_batch(con: duckdb.DuckDBPyConnection, rows: List[Dict[str, Any]]) -> int:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            payload = await call_llm_batch(rows)
            items = payload.get("items", [])
            by_key = {str(it["title_key"]): it for it in items}

            results_to_upsert: List[Dict[str, Any]] = []

            for r in rows:
                k = str(r["TitleKey"])
                if k not in by_key:
                    results_to_upsert.append({
                        "TitleKey": k,
                        "Status": "rejected",
                        "Error": "Missing item in model output for title_key",
                        "Description": "",
                        "KeywordsJson": "",
                        "Scope": "",
                        "Exclusions": "",
                    })
                    continue

                try:
                    normalized, _ = normalize_result(by_key[k])
                    results_to_upsert.append(normalized)
                except Exception as e:
                    results_to_upsert.append({
                        "TitleKey": k,
                        "Status": "rejected",
                        "Error": f"NormalizationError: {type(e).__name__}: {e}",
                        "Description": "",
                        "KeywordsJson": "",
                        "Scope": "",
                        "Exclusions": "",
                    })

            upsert_many(con, results_to_upsert)
            return len(rows)

        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < MAX_RETRIES:
                await asyncio.sleep(0.8 * attempt)
            else:
                rej = []
                for r in rows:
                    rej.append({
                        "TitleKey": str(r["TitleKey"]),
                        "Status": "rejected",
                        "Error": f"BatchError: {last_err}",
                        "Description": "",
                        "KeywordsJson": "",
                        "Scope": "",
                        "Exclusions": "",
                    })
                upsert_many(con, rej)
                return len(rows)
    return 0


# -----------------------------
# Main
# -----------------------------
async def main():
    con = connect_motherduck()
    ensure_table(con)

    total_done = 0

    while True:
        pending = fetch_pending(con, limit=BATCH_SIZE * CONCURRENCY)
        if not pending:
            break

        batches = [pending[i:i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]

        sem = asyncio.Semaphore(CONCURRENCY)

        async def run_with_sem(batch_rows):
            async with sem:
                n = await process_one_batch(con, batch_rows)
                if SLEEP_S:
                    await asyncio.sleep(SLEEP_S)
                return n

        results = await asyncio.gather(*(run_with_sem(b) for b in batches))
        total_done += sum(results)

    con.close()


if __name__ == "__main__":
    asyncio.run(main())

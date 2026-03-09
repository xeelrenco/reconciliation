#!/usr/bin/env python3
"""
Semantic matching: EffectiveDescription (canonical docs) vs historical MDR Document_title using OpenAI embeddings + cosine.

Prereqs:
  pip install duckdb openai numpy

Config: usa config.txt (stessa cartella). Vedi config.example.txt.
  MOTHERDUCK_TOKEN, MOTHERDUCK_DB, OPENAI_API_KEY; PROMPT_VERSION (default per --prompt-version).

Usage:
  python 2.1_semantic_match_mdr.py [--prompt-version v1] --embedding-model text-embedding-3-small
"""

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
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


client = OpenAI(api_key=_cfg("OPENAI_API_KEY"))


# ---------- helpers ----------
def now_ts_naive_utc():
    return datetime.now(timezone.utc).replace(tzinfo=None)

def norm(s: str) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().split())

def text_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def pack_f32(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes(order="C")

def connect_motherduck() -> duckdb.DuckDBPyConnection:
    token = _cfg("MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError("Manca MOTHERDUCK_TOKEN nel file di configurazione (config.txt)")
    dbname = _cfg("MOTHERDUCK_DB", "my_db")
    return duckdb.connect(f"md:{dbname}?token={token}")


# ---------- OpenAI embeddings (batched) ----------
def embed_texts(texts: List[str], model: str, batch_size: int = 256) -> List[np.ndarray]:
    """
    Returns list of float vectors (np.ndarray) in same order as texts.
    """
    out: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=chunk)
        # resp.data preserves input order
        for item in resp.data:
            out.append(np.array(item.embedding, dtype=np.float32))
    return out


# ---------- DB: pull data ----------
def fetch_canonical_descriptions(con: duckdb.DuckDBPyConnection, prompt_version: str) -> List[Tuple[str, str, str]]:
    """
    Returns [(TitleKey, PromptVersion, EffectiveDescription)]
    """
    rows = con.execute(
        """
        SELECT TitleKey, PromptVersion, EffectiveDescription
        FROM my_db.mdr_reconciliation.v_DocumentTitleDescriptions_Final
        WHERE PromptVersion = ?
          AND EffectiveDescription IS NOT NULL
          AND TRIM(EffectiveDescription) <> ''
        ORDER BY TitleKey
        """,
        [prompt_version],
    ).fetchall()
    return [(r[0], r[1], norm(r[2])) for r in rows]

def fetch_distinct_mdr_titles(con: duckdb.DuckDBPyConnection) -> List[str]:
    rows = con.execute(
        """
        SELECT DISTINCT Document_title
        FROM my_db.historical_mdr_normalization.v_MdrPreviousRecords_Normalized_All
        WHERE Document_title IS NOT NULL
          AND TRIM(Document_title) <> ''
        ORDER BY Document_title
        """
    ).fetchall()
    return [norm(r[0]) for r in rows]


# ---------- DB: incremental embedding refresh ----------
def upsert_doc_embeddings(
    con: duckdb.DuckDBPyConnection,
    items: List[Tuple[str, str, str, str, bytes, int, datetime]],
):
    """
    items: [(TitleKey, PromptVersion, EmbeddingModel, TextHash, EmbeddingBlob, Dim, UpdatedAt)]
    """
    con.execute("BEGIN;")
    try:
        con.executemany(
            """
            INSERT INTO my_db.mdr_reconciliation.DocumentDescriptionEmbeddings
              (TitleKey, PromptVersion, EmbeddingModel, TextHash, Embedding, Dim, UpdatedAt)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (TitleKey, PromptVersion, EmbeddingModel) DO UPDATE SET
              TextHash = excluded.TextHash,
              Embedding = excluded.Embedding,
              Dim = excluded.Dim,
              UpdatedAt = excluded.UpdatedAt
            """,
            items,
        )
        con.execute("COMMIT;")
    except Exception:
        con.execute("ROLLBACK;")
        raise

def upsert_mdr_embeddings(
    con: duckdb.DuckDBPyConnection,
    items: List[Tuple[str, str, str, bytes, int, datetime]],
):
    """
    items: [(Document_title, EmbeddingModel, TextHash, EmbeddingBlob, Dim, UpdatedAt)]
    """
    con.execute("BEGIN;")
    try:
        con.executemany(
            """
            INSERT INTO my_db.mdr_reconciliation.MdrTitleEmbeddings
              (Document_title, EmbeddingModel, TextHash, Embedding, Dim, UpdatedAt)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (Document_title, EmbeddingModel) DO UPDATE SET
              TextHash = excluded.TextHash,
              Embedding = excluded.Embedding,
              Dim = excluded.Dim,
              UpdatedAt = excluded.UpdatedAt
            """,
            items,
        )
        con.execute("COMMIT;")
    except Exception:
        con.execute("ROLLBACK;")
        raise

def get_existing_doc_hashes(con: duckdb.DuckDBPyConnection, prompt_version: str, model: str) -> Dict[str, str]:
    rows = con.execute(
        """
        SELECT TitleKey, TextHash
        FROM my_db.mdr_reconciliation.DocumentDescriptionEmbeddings
        WHERE PromptVersion = ? AND EmbeddingModel = ?
        """,
        [prompt_version, model],
    ).fetchall()
    return {r[0]: r[1] for r in rows}

def get_existing_mdr_hashes(con: duckdb.DuckDBPyConnection, model: str) -> Dict[str, str]:
    rows = con.execute(
        """
        SELECT Document_title, TextHash
        FROM my_db.mdr_reconciliation.MdrTitleEmbeddings
        WHERE EmbeddingModel = ?
        """,
        [model],
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-version", default=None, help="PromptVersion to use (default: from config.txt PROMPT_VERSION).")
    ap.add_argument("--embedding-model", default="text-embedding-3-small", help="OpenAI embedding model.")
    ap.add_argument("--embed-batch-size", type=int, default=256, help="Embedding API batch size.")
    ap.add_argument("--force-refresh", action="store_true", help="Recompute embeddings even if hashes match.")
    args = ap.parse_args()

    prompt_version = args.prompt_version or _cfg("PROMPT_VERSION", "v1")
    args.prompt_version = prompt_version

    con = connect_motherduck()

    # Pull texts
    docs = fetch_canonical_descriptions(con, args.prompt_version)
    mdr_titles = fetch_distinct_mdr_titles(con)

    print(f"Canonical docs (PromptVersion={args.prompt_version}): {len(docs)}")
    print(f"Distinct MDR titles: {len(mdr_titles)}")

    # Existing hashes for incremental update
    doc_hashes = get_existing_doc_hashes(con, args.prompt_version, args.embedding_model)
    mdr_hashes = get_existing_mdr_hashes(con, args.embedding_model)

    # Determine what to embed
    doc_to_embed = []
    for titlekey, pv, text in docs:
        h = text_hash(text)
        if args.force_refresh or doc_hashes.get(titlekey) != h:
            doc_to_embed.append((titlekey, pv, text, h))

    mdr_to_embed = []
    for t in mdr_titles:
        h = text_hash(t)
        if args.force_refresh or mdr_hashes.get(t) != h:
            mdr_to_embed.append((t, h))

    print(f"Doc embeddings to refresh: {len(doc_to_embed)}")
    print(f"MDR embeddings to refresh: {len(mdr_to_embed)}")

    # Embed & upsert docs
    if doc_to_embed:
        texts = [x[2] for x in doc_to_embed]
        vecs = embed_texts(texts, args.embedding_model, batch_size=args.embed_batch_size)
        updated = now_ts_naive_utc()
        dim = int(len(vecs[0])) if vecs else 0

        rows = []
        for (titlekey, pv, _text, h), v in zip(doc_to_embed, vecs):
            v = l2_normalize(v)
            rows.append((titlekey, pv, args.embedding_model, h, pack_f32(v), dim, updated))
        upsert_doc_embeddings(con, rows)

    # Embed & upsert MDR titles
    if mdr_to_embed:
        texts = [x[0] for x in mdr_to_embed]
        vecs = embed_texts(texts, args.embedding_model, batch_size=args.embed_batch_size)
        updated = now_ts_naive_utc()
        dim = int(len(vecs[0])) if vecs else 0

        rows = []
        for (t, h), v in zip(mdr_to_embed, vecs):
            v = l2_normalize(v)
            rows.append((t, args.embedding_model, h, pack_f32(v), dim, updated))
        upsert_mdr_embeddings(con, rows)

    print("Done. Embeddings in DocumentDescriptionEmbeddings and MdrTitleEmbeddings.")
    con.close()


if __name__ == "__main__":
    main()
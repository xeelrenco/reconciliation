#!/usr/bin/env python3
"""
Compute MDR -> RACI Top-K semantic matches for ALL distinct MDR titles.

Uses stored, L2-normalized embeddings:
- my_db.mdr_reconciliation.DocumentDescriptionEmbeddings  (docs / TitleKey)
- my_db.mdr_reconciliation.MdrTitleEmbeddings            (MDR titles)

Computes cosine via dot product:
  scores = B @ A.T
where:
  A: (n_docs, dim)
  B: (n_mdr, dim)

Writes Top-K per MDR title into:
  my_db.mdr_reconciliation.MdrToRaciCandidates

Prereqs:
  pip install duckdb numpy pandas

Config: usa config.txt (stessa cartella). Vedi config.example.txt.
  MOTHERDUCK_TOKEN, MOTHERDUCK_DB; PROMPT_VERSION (default per --prompt-version).

Usage:
  python 2.2_mdr_to_raci_topk.py [--prompt-version v1] --embedding-model text-embedding-3-small --top-k 50
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

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


def now_ts_naive_utc():
    return datetime.now(timezone.utc).replace(tzinfo=None)

def connect_motherduck() -> duckdb.DuckDBPyConnection:
    token = _cfg("MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError("Manca MOTHERDUCK_TOKEN nel file di configurazione (config.txt)")
    dbname = _cfg("MOTHERDUCK_DB", "my_db")
    return duckdb.connect(f"md:{dbname}?token={token}")

def unpack_f32(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim)

def load_doc_embedding_matrix(con: duckdb.DuckDBPyConnection, prompt_version: str, model: str) -> Tuple[List[str], np.ndarray]:
    rows = con.execute(
        """
        SELECT TitleKey, Dim, Embedding
        FROM my_db.mdr_reconciliation.DocumentDescriptionEmbeddings
        WHERE PromptVersion = ? AND EmbeddingModel = ?
        ORDER BY TitleKey
        """,
        [prompt_version, model],
    ).fetchall()
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    dim = int(rows[0][1])
    keys = [r[0] for r in rows]
    mat = np.vstack([unpack_f32(r[2], dim) for r in rows]).astype(np.float32, copy=False)
    return keys, mat

def load_mdr_embedding_matrix(con: duckdb.DuckDBPyConnection, model: str) -> Tuple[List[str], np.ndarray]:
    rows = con.execute(
        """
        SELECT Document_title, Dim, Embedding
        FROM my_db.mdr_reconciliation.MdrTitleEmbeddings
        WHERE EmbeddingModel = ?
        ORDER BY Document_title
        """,
        [model],
    ).fetchall()
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    dim = int(rows[0][1])
    titles = [r[0] for r in rows]
    mat = np.vstack([unpack_f32(r[2], dim) for r in rows]).astype(np.float32, copy=False)
    return titles, mat

def write_topk_mdr_to_raci(
    con: duckdb.DuckDBPyConnection,
    prompt_version: str,
    model: str,
    mdr_titles: List[str],
    titlekeys: List[str],
    scores: np.ndarray,   # shape (n_mdr, n_docs)
    top_k: int
) -> None:
    """
    Build top-k rows, load into DataFrame, register as temp table,
    then single INSERT ... SELECT for fast bulk load (instead of executemany).
    """
    created = now_ts_naive_utc()
    n_mdr = scores.shape[0]

    print("Building top-k rows ...")
    out_rows = []
    for i in range(n_mdr):
        row = scores[i]
        if top_k >= row.shape[0]:
            idx = np.argsort(-row)
        else:
            idx = np.argpartition(-row, top_k)[:top_k]
            idx = idx[np.argsort(-row[idx])]
        for rank, j in enumerate(idx[:top_k], start=1):
            out_rows.append((
                mdr_titles[i],
                prompt_version,
                model,
                titlekeys[int(j)],
                float(row[int(j)]),
                int(rank),
                created
            ))

    print(f"Rows built: {len(out_rows)}. Loading into DataFrame ...")
    df = pd.DataFrame(
        out_rows,
        columns=[
            "Document_title",
            "PromptVersion",
            "EmbeddingModel",
            "TitleKey",
            "Similarity",
            "Rank",
            "CreatedAt",
        ],
    )

    print("Registering temp dataframe ...")
    con.register("tmp_mdr_matches", df)

    print("Writing to MotherDuck with INSERT ... SELECT ...")
    con.execute("BEGIN;")
    try:
        con.execute(
            """
            DELETE FROM my_db.mdr_reconciliation.MdrToRaciCandidates
            WHERE PromptVersion = ? AND EmbeddingModel = ?
            """,
            [prompt_version, model],
        )
        con.execute(
            """
            INSERT INTO my_db.mdr_reconciliation.MdrToRaciCandidates
              (Document_title, PromptVersion, EmbeddingModel, TitleKey, Similarity, Rank, CreatedAt)
            SELECT
              Document_title,
              PromptVersion,
              EmbeddingModel,
              TitleKey,
              Similarity,
              Rank,
              CreatedAt
            FROM tmp_mdr_matches
            """
        )
        con.execute("COMMIT;")
    except Exception:
        con.execute("ROLLBACK;")
        raise
    finally:
        con.unregister("tmp_mdr_matches")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-version", default=None, help="PromptVersion (default: from config.txt PROMPT_VERSION).")
    ap.add_argument("--embedding-model", default="text-embedding-3-small")
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    prompt_version = args.prompt_version or _cfg("PROMPT_VERSION", "v1")
    args.prompt_version = prompt_version

    con = connect_motherduck()

    titlekeys, A = load_doc_embedding_matrix(con, args.prompt_version, args.embedding_model)
    mdr_titles, B = load_mdr_embedding_matrix(con, args.embedding_model)

    if A.size == 0:
        raise RuntimeError("No doc embeddings found. Run the embedding build step first.")
    if B.size == 0:
        raise RuntimeError("No MDR title embeddings found. Run the embedding build step first.")
    if A.shape[1] != B.shape[1]:
        raise RuntimeError(f"Embedding dim mismatch: docs {A.shape[1]} vs mdr {B.shape[1]}")

    print(f"Docs: {A.shape[0]} | MDR titles: {B.shape[0]} | dim: {A.shape[1]}")
    print("Computing scores ...")
    scores = (B @ A.T).astype(np.float32, copy=False)
    print("Scores computed.")

    print(f"Writing Top-{args.top_k} for ALL MDR titles ...")
    write_topk_mdr_to_raci(
        con=con,
        prompt_version=args.prompt_version,
        model=args.embedding_model,
        mdr_titles=mdr_titles,
        titlekeys=titlekeys,
        scores=scores,
        top_k=args.top_k
    )

    print("Done. Results in my_db.mdr_reconciliation.MdrToRaciCandidates")
    con.close()


if __name__ == "__main__":
    main()
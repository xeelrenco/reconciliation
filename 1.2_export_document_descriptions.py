#!/usr/bin/env python3
"""
Export document descriptions: join mdr_reconciliation.DocumentTitleDescriptions with
mdr_reconciliation.v_DocumentsEnriched and save to Excel.

Usa config.txt (stessa cartella) per MOTHERDUCK_TOKEN, MOTHERDUCK_DB, PROMPT_VERSION.
Opzionale: EXPORT_OUTPUT_FILE per il path del file Excel (default: export_raci.xlsx).

Prereq: pip install duckdb pandas openpyxl
"""

from pathlib import Path
from typing import Dict, Optional

import duckdb
import pandas as pd

# -----------------------------
# Config da file (stesso formato di generate_doc_descriptions.py)
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


def connect_motherduck() -> duckdb.DuckDBPyConnection:
    token = _cfg("MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError("Manca MOTHERDUCK_TOKEN nel file di configurazione (config.txt)")
    dbname = _cfg("MOTHERDUCK_DB", "my_db")
    return duckdb.connect(f"md:{dbname}?token={token}")


def main() -> None:
    con = connect_motherduck()
    prompt_version = _cfg("PROMPT_VERSION", "v1")

    df = con.execute(
        """
        SELECT
          DOC.TypeName,
          DOC.ChapterName,
          DOC.CategoryDescription,
          DOC.DisciplineName,
          DOC.Title,
          TITLE.Description,
          TITLE.Scope
        FROM mdr_reconciliation.DocumentTitleDescriptions AS TITLE
        JOIN mdr_reconciliation.v_DocumentsEnriched AS DOC
          ON TITLE.TitleKey = DOC.TitleKey
        WHERE TITLE.PromptVersion = ?
        """,
        [prompt_version],
    ).df()

    con.close()

    out_path = _cfg("EXPORT_OUTPUT_FILE") or "document_descriptions.xlsx"
    # Se path relativo, salva nella cartella dello script
    if not Path(out_path).is_absolute():
        out_path = str(Path(__file__).resolve().parent / out_path)

    df.to_excel(out_path, index=False)
    print(f"Export completato: {len(df)} righe -> {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Apply manual reviewed descriptions from a single Excel file to MotherDuck.

Excel format:
- Column A: Title (must match raci_matrix.Documents.Title exactly; Title is UNIQUE)
- Column B: ManualDescription (reviewed text)

Behavior:
- Resolves Title -> TitleKey via raci_matrix.Documents
- Ensures a row exists in mdr_reconciliation.DocumentTitleDescriptions for (TitleKey, PromptVersion)
- Updates ONLY ManualDescription (AI Description remains untouched)
- Optionally sets Status='manual_written' (default: yes) to distinguish hand-written from AI-generated

Requirements:
  pip install duckdb pandas openpyxl

Config: usa config.txt (stessa cartella). Vedi config.example.txt.
  MOTHERDUCK_TOKEN, MOTHERDUCK_DB; PROMPT_VERSION (default per --prompt-version).

Usage:
  python apply_manual_reviews.py --excel /path/reviews.xlsx [--prompt-version v1]
  python apply_manual_reviews.py --excel /path/reviews.xlsx --no-approve
  python apply_manual_reviews.py --excel /path/reviews.xlsx --dry-run
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import duckdb

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


def norm(s) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().split())


def clean_str(x):
    # robusto: gestisce None, NaN/pd.NA e NBSP
    if x is None or pd.isna(x):
        return ""
    return str(x).replace("\u00A0", " ").strip()


def title_key(s: str) -> str:
    return clean_str(s).lower()


def now_ts_naive_utc():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def connect_motherduck() -> duckdb.DuckDBPyConnection:
    token = _cfg("MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError("Manca MOTHERDUCK_TOKEN nel file di configurazione (config.txt)")
    dbname = _cfg("MOTHERDUCK_DB", "renco")
    return duckdb.connect(f"md:{dbname}?token={token}")


def ensure_manual_column(con: duckdb.DuckDBPyConnection) -> None:
    cols = con.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema='mdr_reconciliation'
          AND table_name='DocumentTitleDescriptions'
    """).fetchall()
    existing = {c[0] for c in cols}
    if "ManualDescription" not in existing:
        con.execute("ALTER TABLE mdr_reconciliation.DocumentTitleDescriptions ADD COLUMN ManualDescription VARCHAR;")


def load_excel(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, header=None, dtype=str)
    if df.shape[1] < 2:
        raise ValueError("Excel must have at least 2 columns: A=Title, B=ManualDescription")

    df = df.iloc[:, :2].copy()
    df.columns = ["Title", "ManualDescription"]
    df["Title"] = df["Title"].apply(clean_str)
    df["ManualDescription"] = df["ManualDescription"].apply(norm)
    df["TitleKey"] = df["Title"].apply(title_key)

    # Drop fully empty rows
    df = df[(df["Title"] != "") | (df["ManualDescription"] != "")]
    return df


def validate_excel(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    invalid = df[(df["Title"] == "") | (df["ManualDescription"] == "")]
    valid = df[(df["Title"] != "") & (df["ManualDescription"] != "")]
    return valid, invalid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="Path to Excel (A=Title, B=ManualDescription).")
    ap.add_argument("--prompt-version", default=None, help="PromptVersion to target (default: from config.txt PROMPT_VERSION).")
    ap.add_argument("--dry-run", action="store_true", help="If set, do not write changes; only report.")
    ap.add_argument("--no-approve", action="store_true", help="If set, do NOT change Status to 'manual_written'.")
    args = ap.parse_args()

    prompt_version = args.prompt_version or _cfg("PROMPT_VERSION", "v1")
    args.prompt_version = prompt_version

    df = load_excel(args.excel)
    valid, invalid = validate_excel(df)

    dup = valid[valid.duplicated(subset=["TitleKey"], keep=False)].sort_values("Title")

    print(f"Rows in Excel (non-empty): {len(df)}")
    print(f"Valid rows (Title+ManualDescription present): {len(valid)}")
    print(f"Invalid rows (missing Title or ManualDescription): {len(invalid)}")
    print(f"Duplicate TitleKeys in Excel: {len(dup)}")

    if len(invalid) > 0:
        print("\nINVALID ROWS (first 20):")
        print(invalid.head(20).to_string(index=False))

    if len(dup) > 0:
        print("\nDUPLICATE TITLEKEYS IN EXCEL (first 50):")
        print(dup.head(50).to_string(index=False))
        raise SystemExit("Fix duplicate TitleKeys in the Excel before applying updates.")

    con = connect_motherduck()
    ensure_manual_column(con)

    # Register valid rows
    con.register("tmp_reviews", valid)

    # Resolve Excel-derived TitleKey -> DB TitleKey
    resolved = con.execute("""
        SELECT r.Title, r.ManualDescription, d.TitleKey
        FROM tmp_reviews r
        LEFT JOIN raci_matrix.Documents d
          ON d.TitleKey = r.TitleKey
    """).fetchdf()

    missing = resolved[resolved["TitleKey"].isna()].copy()
    ok = resolved[~resolved["TitleKey"].isna()].copy()

    print(f"\nTitles not found in raci_matrix.Documents: {len(missing)}")
    if len(missing) > 0:
        print("\nMISSING TITLES (first 50):")
        print(missing.head(50)[["Title"]].to_string(index=False))
        con.close()
        raise SystemExit("Some Titles were not found in Documents. Fix titles in Excel and rerun.")

    con.register("tmp_ok", ok)

    if args.dry_run:
        print("\nDRY RUN enabled: no changes written to DB.")
        con.close()
        return

    ts = now_ts_naive_utc()
    con.execute("BEGIN;")
    try:
        # 1) Ensure row exists for (TitleKey, PromptVersion)
        # Insert placeholder rows where missing.
        con.execute("""
            INSERT INTO mdr_reconciliation.DocumentTitleDescriptions
              (TitleKey, PromptVersion, Model, Description, KeywordsJson, Scope, Exclusions,
               Status, Error, CreatedAt, UpdatedAt, ManualDescription)
            SELECT
              o.TitleKey,
              ? AS PromptVersion,
              NULL AS Model,
              '' AS Description,
              NULL AS KeywordsJson,
              NULL AS Scope,
              NULL AS Exclusions,
              'manual_written' AS Status,
              NULL AS Error,
              ? AS CreatedAt,
              ? AS UpdatedAt,
              o.ManualDescription AS ManualDescription
            FROM tmp_ok o
            LEFT JOIN mdr_reconciliation.DocumentTitleDescriptions t
              ON t.TitleKey = o.TitleKey AND t.PromptVersion = ?
            WHERE t.TitleKey IS NULL
        """, [args.prompt_version, ts, ts, args.prompt_version])

        # 2) Update ManualDescription on existing rows
        if args.no_approve:
            con.execute("""
                UPDATE mdr_reconciliation.DocumentTitleDescriptions t
                SET
                  ManualDescription = o.ManualDescription,
                  UpdatedAt = ?
                FROM tmp_ok o
                WHERE t.TitleKey = o.TitleKey
                  AND t.PromptVersion = ?
            """, [ts, args.prompt_version])
        else:
            con.execute("""
                UPDATE mdr_reconciliation.DocumentTitleDescriptions t
                SET
                  ManualDescription = o.ManualDescription,
                  Status = 'manual_written',
                  UpdatedAt = ?
                FROM tmp_ok o
                WHERE t.TitleKey = o.TitleKey
                  AND t.PromptVersion = ?
            """, [ts, args.prompt_version])

        con.execute("COMMIT;")
    except Exception:
        con.execute("ROLLBACK;")
        con.close()
        raise

    # Stats
    stats = con.execute("""
        SELECT
          COUNT(*) AS total_rows,
          SUM(CASE WHEN ManualDescription IS NOT NULL AND ManualDescription <> '' THEN 1 ELSE 0 END) AS manual_filled
        FROM mdr_reconciliation.DocumentTitleDescriptions
        WHERE PromptVersion = ?
    """, [args.prompt_version]).fetchall()[0]

    print("\nDONE.")
    print(f"PromptVersion={args.prompt_version} -> total rows={stats[0]}, manual descriptions filled={stats[1]}")

    con.close()


if __name__ == "__main__":
    main()
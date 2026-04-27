#!/usr/bin/env python3
"""
backup_motherduck_database.py
=============================
Backup completo di un database MotherDuck in una cartella locale timestampata.

Strategia:
  - Connessione a MotherDuck via DuckDB.
  - EXPORT DATABASE verso filesystem locale (schema + dati).
  - Salvataggio metadata del backup in JSON.

Prerequisiti:
  pip install duckdb

Configurazione:
  - config.txt (stessa cartella script): MOTHERDUCK_TOKEN, MOTHERDUCK_DB
  - oppure variabili ambiente MOTHERDUCK_TOKEN, MOTHERDUCK_DB

Uso base:
  python backup_motherduck_database.py

Esempi:
  python backup_motherduck_database.py --output-dir ./backups
  python backup_motherduck_database.py --format parquet
  python backup_motherduck_database.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import duckdb


CONFIG_PATH = Path(__file__).resolve().parent / "config.txt"
DEFAULT_BACKUP_ROOT = Path(__file__).resolve().parent / "backups"


def load_config(path: Optional[Path] = None) -> Dict[str, str]:
    p = path or CONFIG_PATH
    config: Dict[str, str] = {}
    if not p.exists():
        return config
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                config[key.strip()] = value.strip()
    return config


def _cfg(cfg: Dict[str, str], key: str, default: str = "") -> str:
    return (cfg.get(key) or os.environ.get(key) or default).strip()


def connect_motherduck(cfg: Dict[str, str]) -> tuple[duckdb.DuckDBPyConnection, str]:
    token = _cfg(cfg, "MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError(
            "Manca MOTHERDUCK_TOKEN (impostalo in config.txt o come variabile ambiente)."
        )
    dbname = _cfg(cfg, "MOTHERDUCK_DB", "my_db")
    conn = duckdb.connect(f"md:{dbname}?token={token}")
    return conn, dbname


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backup completo database MotherDuck")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_BACKUP_ROOT),
        help="Cartella radice dei backup (default: ./backups accanto allo script)",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Formato export DuckDB (default: parquet)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra configurazione e destinazione senza eseguire export",
    )
    return parser.parse_args()


def list_user_schemas(con: duckdb.DuckDBPyConnection) -> list[str]:
    rows = con.execute(
        """
        SELECT DISTINCT schema_name
        FROM information_schema.schemata
        WHERE schema_name NOT IN ('information_schema', 'pg_catalog')
        ORDER BY schema_name
        """
    ).fetchall()
    return [r[0] for r in rows]


def _sql_quote(value: str) -> str:
    return value.replace("'", "''")


def collect_table_comments(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str, str]]:
    rows = con.execute(
        """
        SELECT schema_name, table_name, comment
        FROM duckdb_tables()
        WHERE internal = false
          AND comment IS NOT NULL
          AND length(trim(comment)) > 0
        ORDER BY schema_name, table_name
        """
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def collect_column_comments(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str, str, str]]:
    rows = con.execute(
        """
        SELECT schema_name, table_name, column_name, comment
        FROM duckdb_columns()
        WHERE internal = false
          AND comment IS NOT NULL
          AND length(trim(comment)) > 0
        ORDER BY schema_name, table_name, column_index
        """
    ).fetchall()
    return [(r[0], r[1], r[2], r[3]) for r in rows]


def write_comments_sql(
    backup_dir: Path,
    table_comments: list[tuple[str, str, str]],
    column_comments: list[tuple[str, str, str, str]],
) -> None:
    lines = [
        "-- comments.sql",
        "-- Commenti esportati dal DB sorgente.",
        "-- Eseguire dopo IMPORT DATABASE per ripristinare i commenti.",
        "",
    ]
    for schema_name, table_name, comment in table_comments:
        lines.append(
            f"COMMENT ON TABLE \"{schema_name}\".\"{table_name}\" IS '{_sql_quote(comment)}';"
        )
    for schema_name, table_name, column_name, comment in column_comments:
        lines.append(
            "COMMENT ON COLUMN "
            f"\"{schema_name}\".\"{table_name}\".\"{column_name}\" "
            f"IS '{_sql_quote(comment)}';"
        )
    lines.append("")
    (backup_dir / "comments.sql").write_text("\n".join(lines), encoding="utf-8")


def write_metadata(
    backup_dir: Path,
    db_name: str,
    export_format: str,
    schemas: list[str],
    table_comment_count: int,
    column_comment_count: int,
) -> None:
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "database": db_name,
        "export_format": export_format,
        "schemas": schemas,
        "table_comment_count": table_comment_count,
        "column_comment_count": column_comment_count,
        "comments_file": "comments.sql",
        "restore_hint": (
            "Esempio restore in DuckDB locale: "
            "IMPORT DATABASE '<backup_dir_assoluto>'; "
            "poi esegui comments.sql per riapplicare i commenti."
        ),
    }
    (backup_dir / "backup_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def run_backup(
    con: duckdb.DuckDBPyConnection,
    backup_dir: Path,
    export_format: str,
) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    safe_path = str(backup_dir).replace("'", "''")
    sql = f"EXPORT DATABASE '{safe_path}' (FORMAT {export_format.upper()});"
    con.execute(sql)


def main() -> None:
    args = parse_args()
    cfg = load_config()
    con, dbname = connect_motherduck(cfg)
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(args.output_dir).expanduser().resolve()
        backup_dir = root / f"{dbname}_backup_{ts}"
        schemas = list_user_schemas(con)
        table_comments = collect_table_comments(con)
        column_comments = collect_column_comments(con)

        print(f"Database: {dbname}")
        print(f"Schemas trovati: {len(schemas)} -> {', '.join(schemas) if schemas else '(nessuno)'}")
        print(
            f"Commenti trovati: tabelle={len(table_comments)}, colonne={len(column_comments)}"
        )
        print(f"Destinazione backup: {backup_dir}")
        print(f"Formato export: {args.format}")

        if args.dry_run:
            print("[DRY-RUN] Nessun file scritto.")
            return

        print("Avvio backup completo (EXPORT DATABASE)...")
        run_backup(con, backup_dir, args.format)
        write_comments_sql(backup_dir, table_comments, column_comments)
        write_metadata(
            backup_dir,
            dbname,
            args.format,
            schemas,
            table_comment_count=len(table_comments),
            column_comment_count=len(column_comments),
        )

        print("[OK] Backup completato.")
        print(f"[OK] Cartella backup: {backup_dir}")
        print(
            "Restore hint: apri DuckDB e usa "
            f"IMPORT DATABASE '{str(backup_dir).replace(chr(92), '/')}'"
        )
        print("[OK] Commenti esportati in comments.sql")
        print("Dopo l'IMPORT, esegui anche il file comments.sql per ripristinarli.")
    finally:
        con.close()


if __name__ == "__main__":
    main()


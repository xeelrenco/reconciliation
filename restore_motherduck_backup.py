#!/usr/bin/env python3
"""
restore_motherduck_backup.py
============================
Ripristina un backup creato con backup_motherduck_database.py.

Importante:
  - Questo script NON legge mai config.txt.
  - Legge le credenziali SOLO da un file dedicato (default: restore_config.txt).
  - In assenza di --execute, gira in DRY-RUN e non modifica nulla.

Prerequisiti:
  pip install duckdb

Uso:
  python restore_motherduck_backup.py --backup-dir ./backups/my_db_backup_YYYYMMDD_HHMMSS
  python restore_motherduck_backup.py --backup-dir <path> --execute
  python restore_motherduck_backup.py --backup-dir <path> --creds-file ./restore_config.txt --execute
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import duckdb


DEFAULT_CREDS_PATH = Path(__file__).resolve().parent / "restore_config.txt"
HELPER_DB_NAME = "__md_restore_admin_helper"


def load_key_value_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"File credenziali restore non trovato: {path}\n"
            "Crea il file (vedi restore_config.example.txt)."
        )
    cfg: Dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                cfg[key.strip()] = value.strip()
    return cfg


def _cfg(cfg: Dict[str, str], key: str, default: str = "") -> str:
    return (cfg.get(key) or default).strip()


def _sql_quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def get_restore_credentials(creds: Dict[str, str]) -> tuple[str, str]:
    token = _cfg(creds, "RESTORE_MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError(
            "Manca RESTORE_MOTHERDUCK_TOKEN nel file credenziali restore."
        )
    dbname = _cfg(creds, "RESTORE_MOTHERDUCK_DB", "my_db")
    return token, dbname


def connect_target_motherduck(token: str, dbname: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(f"md:{dbname}?token={token}")


def connect_admin_motherduck(token: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(f"md:?token={token}")


def list_databases(con: duckdb.DuckDBPyConnection) -> set[str]:
    rows = con.execute("SHOW DATABASES").fetchall()
    names: set[str] = set()
    for row in rows:
        if row and row[0]:
            names.add(str(row[0]))
    return names


def database_exists(con: duckdb.DuckDBPyConnection, db_name: str) -> bool:
    return db_name in list_databases(con)


def ensure_database_exists(token: str, db_name: str) -> None:
    admin_con = connect_admin_motherduck(token)
    try:
        admin_con.execute(
            f"CREATE DATABASE IF NOT EXISTS {_sql_quote_identifier(db_name)};"
        )
    finally:
        admin_con.close()


def choose_existing_db_action(existing_db: str) -> str:
    print(f"[WARN] Il database target '{existing_db}' esiste gia'.")
    print("Scegli una delle opzioni:")
    print("  1) Sovrascrivere il DB esistente")
    print("  2) Rinominare il DB esistente")
    print("  3) Usare un altro nome per il restore")
    while True:
        choice = input("Seleziona opzione [1/2/3]: ").strip()
        if choice in {"1", "2", "3"}:
            return choice
        print("Valore non valido. Inserisci 1, 2 oppure 3.")


def logical_rename_database(
    admin_con: duckdb.DuckDBPyConnection,
    source_db: str,
    target_db: str,
) -> None:
    # Tentativo 1: zero-copy clone.
    # Su alcuni piani (es. lite) puo' fallire per opzioni retention del sorgente.
    try:
        admin_con.execute(
            f"CREATE DATABASE {_sql_quote_identifier(target_db)} "
            f"FROM {_sql_quote_identifier(source_db)};"
        )
    except duckdb.Error as exc:
        msg = str(exc).lower()
        if "snapshot retention" not in msg:
            raise
        # Fallback: crea DB standard e copia contenuti con overwrite.
        admin_con.execute(f"CREATE DATABASE {_sql_quote_identifier(target_db)};")
        admin_con.execute(
            f"COPY FROM DATABASE {_sql_quote_identifier(source_db)} "
            f"(OVERWRITE) TO {_sql_quote_identifier(target_db)};"
        )
    # Per poter eliminare il DB sorgente non deve essere quello "in uso".
    admin_con.execute(f"USE {_sql_quote_identifier(target_db)};")
    admin_con.execute(f"DROP DATABASE {_sql_quote_identifier(source_db)};")


def drop_database_safe(
    admin_con: duckdb.DuckDBPyConnection,
    db_name: str,
) -> None:
    helper_db = HELPER_DB_NAME
    admin_con.execute(f"CREATE DATABASE IF NOT EXISTS {_sql_quote_identifier(helper_db)};")
    admin_con.execute(f"USE {_sql_quote_identifier(helper_db)};")
    admin_con.execute(f"DROP DATABASE {_sql_quote_identifier(db_name)};")


def cleanup_helper_database(token: str, preferred_db: str) -> None:
    admin_con = connect_admin_motherduck(token)
    try:
        # Prova a spostare il contesto su un DB reale non-helper per poterlo eliminare.
        use_db = preferred_db if preferred_db != HELPER_DB_NAME else "my_db"
        try:
            admin_con.execute(f"USE {_sql_quote_identifier(use_db)};")
        except duckdb.Error:
            pass
        try:
            admin_con.execute(f"DROP DATABASE {_sql_quote_identifier(HELPER_DB_NAME)};")
        except duckdb.Error:
            # Best effort: non bloccare il restore per il cleanup.
            pass
    finally:
        admin_con.close()


def resolve_target_db_name(
    token: str,
    requested_db: str,
    execute: bool,
    on_existing: str,
    rename_existing_to: str,
    restore_db_name: str,
) -> str:
    admin_con = connect_admin_motherduck(token)
    try:
        existing = list_databases(admin_con)
        if requested_db not in existing:
            return requested_db

        if not execute:
            print(
                f"[WARN] Il database '{requested_db}' esiste gia'. "
                "In DRY-RUN nessuna azione viene eseguita."
            )
            print(
                "[INFO] Con --execute verra' applicata la policy scelta "
                "(--on-existing) oppure il prompt interattivo."
            )
            return requested_db

        if on_existing == "overwrite":
            drop_database_safe(admin_con, requested_db)
            print(f"[OK] Database esistente '{requested_db}' eliminato.")
            return requested_db

        if on_existing == "rename-existing":
            new_existing_name = rename_existing_to.strip()
            if not new_existing_name:
                raise RuntimeError(
                    "Manca --rename-existing-to con --on-existing rename-existing."
                )
            if new_existing_name == requested_db:
                raise RuntimeError("--rename-existing-to deve essere diverso dal nome DB corrente.")
            if database_exists(admin_con, new_existing_name):
                raise RuntimeError(
                    f"Il nome '{new_existing_name}' esiste gia'. Scegline uno diverso."
                )
            logical_rename_database(admin_con, requested_db, new_existing_name)
            print(
                f"[OK] Database esistente rinominato: "
                f"'{requested_db}' -> '{new_existing_name}'."
            )
            return requested_db

        if on_existing == "new-name":
            target_name = restore_db_name.strip()
            if not target_name:
                raise RuntimeError("Manca --restore-db-name con --on-existing new-name.")
            if database_exists(admin_con, target_name):
                raise RuntimeError(
                    f"Il nome '{target_name}' esiste gia'. Scegline uno diverso."
                )
            print(f"[OK] Restore impostato su nuovo database: '{target_name}'.")
            return target_name

        if not sys.stdin.isatty():
            raise RuntimeError(
                f"Il database '{requested_db}' esiste gia' ma la sessione non e' interattiva. "
                "Usa --on-existing overwrite|rename-existing|new-name."
            )

        while True:
            action = choose_existing_db_action(requested_db)
            if action == "1":
                drop_database_safe(admin_con, requested_db)
                print(f"[OK] Database esistente '{requested_db}' eliminato.")
                return requested_db

            if action == "2":
                new_existing_name = input(
                    "Nuovo nome per il DB esistente: "
                ).strip()
                if not new_existing_name:
                    print("Il nome non puo' essere vuoto.")
                    continue
                if database_exists(admin_con, new_existing_name):
                    print(f"Il nome '{new_existing_name}' esiste gia'. Scegline un altro.")
                    continue
                logical_rename_database(admin_con, requested_db, new_existing_name)
                print(
                    f"[OK] Database esistente rinominato: "
                    f"'{requested_db}' -> '{new_existing_name}'."
                )
                return requested_db

            restore_name = input("Nome database da usare per il restore: ").strip()
            if not restore_name:
                print("Il nome non puo' essere vuoto.")
                continue
            if database_exists(admin_con, restore_name):
                print(f"Il nome '{restore_name}' esiste gia'. Scegline un altro.")
                continue
            print(f"[OK] Restore impostato su nuovo database: '{restore_name}'.")
            return restore_name
    finally:
        admin_con.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore backup MotherDuck")
    parser.add_argument(
        "--backup-dir",
        default="",
        help=(
            "Cartella backup contenente schema.sql/load.sql (e opzionale comments.sql). "
            "Se omessa, usa RESTORE_BACKUP_DIR dal file credenziali."
        ),
    )
    parser.add_argument(
        "--creds-file",
        default=str(DEFAULT_CREDS_PATH),
        help="File credenziali restore (default: ./restore_config.txt accanto allo script)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Esegue davvero il restore. Se assente, solo DRY-RUN.",
    )
    parser.add_argument(
        "--skip-comments",
        action="store_true",
        help="Non applica comments.sql anche se presente.",
    )
    parser.add_argument(
        "--on-existing",
        choices=["prompt", "overwrite", "rename-existing", "new-name"],
        default="prompt",
        help=(
            "Comportamento se il DB target esiste: "
            "prompt (interattivo), overwrite, rename-existing, new-name"
        ),
    )
    parser.add_argument(
        "--rename-existing-to",
        default="",
        help="Nuovo nome per il DB esistente (richiesto con --on-existing rename-existing).",
    )
    parser.add_argument(
        "--restore-db-name",
        default="",
        help="Nome DB target da usare per il restore (richiesto con --on-existing new-name).",
    )
    return parser.parse_args()


def validate_backup_dir(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Backup directory non valida: {path}")
    required = ["schema.sql", "load.sql"]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Backup incompleto in {path}. Mancano i file: {', '.join(missing)}"
        )


def safe_sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def run_restore(
    con: duckdb.DuckDBPyConnection,
    backup_dir: Path,
    apply_comments: bool,
) -> None:
    import_sql = f"IMPORT DATABASE '{safe_sql_path(backup_dir)}';"
    con.execute(import_sql)

    comments_file = backup_dir / "comments.sql"
    if apply_comments and comments_file.exists():
        comments_sql = comments_file.read_text(encoding="utf-8")
        if comments_sql.strip():
            con.execute(comments_sql)


def main() -> None:
    args = parse_args()
    creds_file = Path(args.creds_file).expanduser().resolve()
    creds = load_key_value_file(creds_file)
    backup_dir_raw = args.backup_dir.strip() or _cfg(creds, "RESTORE_BACKUP_DIR")
    if not backup_dir_raw:
        raise RuntimeError(
            "Backup directory non specificata. Usa --backup-dir oppure RESTORE_BACKUP_DIR nel file credenziali."
        )
    backup_dir = Path(backup_dir_raw).expanduser().resolve()
    validate_backup_dir(backup_dir)

    token, requested_db = get_restore_credentials(creds)
    target_db = resolve_target_db_name(
        token,
        requested_db,
        execute=args.execute,
        on_existing=args.on_existing,
        rename_existing_to=args.rename_existing_to,
        restore_db_name=args.restore_db_name,
    )
    ensure_database_exists(token, target_db)
    con = connect_target_motherduck(token, target_db)
    try:
        apply_comments = not args.skip_comments
        comments_exists = (backup_dir / "comments.sql").exists()
        print(f"Target DB: {target_db}")
        print(f"Credenziali restore da: {creds_file}")
        print(f"Backup directory: {backup_dir}")
        print(f"comments.sql presente: {'yes' if comments_exists else 'no'}")
        print(f"Apply comments: {'yes' if (apply_comments and comments_exists) else 'no'}")

        if not args.execute:
            print("[DRY-RUN] Nessuna modifica eseguita. Aggiungi --execute per procedere.")
            return

        print("Avvio restore (IMPORT DATABASE)...")
        run_restore(con, backup_dir, apply_comments=apply_comments)
        print("[OK] Restore completato.")
        if apply_comments and comments_exists:
            print("[OK] comments.sql applicato.")
    finally:
        con.close()
        cleanup_helper_database(token, target_db)


if __name__ == "__main__":
    main()


# ===== Logging: console (uvicorn) + CSV =====
import logging, sys, csv, os
from pathlib import Path
from datetime import datetime
from time import perf_counter

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
CSV_LOG = LOG_DIR / "api_events.csv"

# Console Handler (aparece no terminal do uvicorn)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))

# Logger principal do app
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
# evita adicionar handlers duplicados em hot-reload
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(console_handler)

# Garantir que registros do nosso logger também vão para o root (uvicorn pode capturar)
logger.propagate = True

# CSV logging (escreve cabeçalho se não existir)
CSV_HEADERS = [
    "ts_iso", "level", "event",
    "path", "method", "status",
    "job_id", "top_n",
    "n_candidates_input", "n_candidates_scored",
    "duration_ms", "error",
]

def _csv_write(row: dict):
    file_exists = CSV_LOG.exists()
    with open(CSV_LOG, mode="a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            w.writeheader()
        # garante todas as colunas
        safe = {k: row.get(k, "") for k in CSV_HEADERS}
        w.writerow(safe)

def log_event(
    level: str,
    event: str,
    path: str = "",
    method: str = "",
    status: int = 200,
    job_id: str | int | None = None,
    top_n: int | None = None,
    n_candidates_input: int | None = None,
    n_candidates_scored: int | None = None,
    duration_ms: float | None = None,
    error: str | None = None,
):
    # Console
    msg = (
        f"{event} | path={path} method={method} status={status} "
        f"job_id={job_id} top_n={top_n} "
        f"n_in={n_candidates_input} n_scored={n_candidates_scored} "
        f"dur_ms={duration_ms} err={error}"
    )
    if level.lower() == "error":
        logger.error(msg)
    elif level.lower() == "warning":
        logger.warning(msg)
    else:
        logger.info(msg)

    # CSV
    _csv_write({
        "ts_iso": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
        "level": level.upper(),
        "event": event,
        "path": path,
        "method": method,
        "status": status,
        "job_id": job_id if job_id is not None else "",
        "top_n": top_n if top_n is not None else "",
        "n_candidates_input": n_candidates_input if n_candidates_input is not None else "",
        "n_candidates_scored": n_candidates_scored if n_candidates_scored is not None else "",
        "duration_ms": f"{duration_ms:.2f}" if isinstance(duration_ms, (int, float)) else "",
        "error": (error or "")[:500],  # limita tamanho
    })
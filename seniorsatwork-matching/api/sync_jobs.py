"""
In-memory sync job registry for background candidate/job index syncs.

Poll GET /api/index/sync/status/{sync_id} from the UI or external clients.
Process-local only: use a single uvicorn worker, or replace with Redis later.
"""
from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

SyncKind = Literal["candidates", "jobs"]
SyncStatus = Literal["pending", "running", "completed", "failed", "timeout"]

_MAX_JOBS = int(os.getenv("SYNC_JOB_HISTORY_MAX", "20"))

_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}
_order: list[str] = []


def _trim_if_needed() -> None:
    while len(_order) > _MAX_JOBS:
        old = _order.pop(0)
        _jobs.pop(old, None)


def create_job(kind: SyncKind, script_basename: str) -> str:
    """Register a new job (pending). Returns sync_id."""
    sync_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    rec: dict[str, Any] = {
        "sync_id": sync_id,
        "kind": kind,
        "script": script_basename,
        "status": "pending",
        "started_at": None,
        "finished_at": None,
        "exit_code": None,
        "ok": None,
        "stdout_tail": None,
        "stderr_tail": None,
        "error": None,
        "created_at": now,
    }
    with _lock:
        _jobs[sync_id] = rec
        _order.append(sync_id)
        _trim_if_needed()
    return sync_id


def mark_running(sync_id: str) -> None:
    with _lock:
        rec = _jobs.get(sync_id)
        if not rec:
            return
        rec["status"] = "running"
        rec["started_at"] = datetime.now(timezone.utc).isoformat()


def finish_subprocess(
    sync_id: str,
    *,
    exit_code: int,
    stdout_tail: str,
    stderr_tail: str,
) -> None:
    ok = exit_code == 0
    status: SyncStatus = "completed" if ok else "failed"
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        rec = _jobs.get(sync_id)
        if not rec:
            return
        rec["status"] = status
        rec["finished_at"] = now
        rec["exit_code"] = exit_code
        rec["ok"] = ok
        rec["stdout_tail"] = stdout_tail
        rec["stderr_tail"] = stderr_tail


def finish_timeout(sync_id: str, timeout_s: int) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        rec = _jobs.get(sync_id)
        if not rec:
            return
        rec["status"] = "timeout"
        rec["finished_at"] = now
        rec["exit_code"] = None
        rec["ok"] = False
        rec["error"] = f"Subprocess timed out after {timeout_s}s"
        rec["stdout_tail"] = None
        rec["stderr_tail"] = None


def finish_exception(sync_id: str, exc: BaseException) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        rec = _jobs.get(sync_id)
        if not rec:
            return
        rec["status"] = "failed"
        rec["finished_at"] = now
        rec["exit_code"] = None
        rec["ok"] = False
        rec["error"] = str(exc)
        rec["stdout_tail"] = None
        rec["stderr_tail"] = None


def get_job(sync_id: str) -> dict[str, Any] | None:
    with _lock:
        rec = _jobs.get(sync_id)
        if rec is None:
            return None
        return dict(rec)

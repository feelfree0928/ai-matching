"""Unit tests for in-memory sync job registry."""
from __future__ import annotations

import pytest

from api import sync_jobs as sj


def test_create_mark_finish_roundtrip() -> None:
    jid = sj.create_job("candidates", "incremental_sync.py")
    assert len(jid) == 36
    r = sj.get_job(jid)
    assert r is not None
    assert r["status"] == "pending"
    assert r["script"] == "incremental_sync.py"

    sj.mark_running(jid)
    r = sj.get_job(jid)
    assert r["status"] == "running"
    assert r["started_at"] is not None

    sj.finish_subprocess(jid, exit_code=0, stdout_tail="ok", stderr_tail="")
    r = sj.get_job(jid)
    assert r["status"] == "completed"
    assert r["ok"] is True
    assert r["exit_code"] == 0
    assert r["stdout_tail"] == "ok"


def test_get_unknown_returns_none() -> None:
    assert sj.get_job("00000000-0000-0000-0000-000000000000") is None

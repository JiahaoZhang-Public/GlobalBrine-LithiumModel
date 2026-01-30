from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Literal, Optional
from uuid import uuid4

JobStatus = Literal["queued", "running", "completed", "failed"]


@dataclass
class JobRecord:
    id: str
    status: JobStatus
    input_path: Path
    output_path: Path | None = None
    error: str | None = None
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        # Serialize paths to strings for JSON
        data["input_path"] = str(self.input_path)
        if self.output_path is not None:
            data["output_path"] = str(self.output_path)
        if self.submitted_at:
            data["submitted_at"] = self.submitted_at.isoformat()
        if self.completed_at:
            data["completed_at"] = (
                self.completed_at.isoformat() if self.completed_at else None
            )
        return data

    @classmethod
    def from_json(cls, payload: dict) -> "JobRecord":
        return cls(
            id=payload["id"],
            status=payload["status"],
            input_path=Path(payload["input_path"]),
            output_path=Path(payload["output_path"])
            if payload.get("output_path")
            else None,
            error=payload.get("error"),
            submitted_at=datetime.fromisoformat(payload["submitted_at"]),
            completed_at=datetime.fromisoformat(payload["completed_at"])
            if payload.get("completed_at")
            else None,
        )


class JobStore:
    """Minimal in-memory + on-disk job registry."""

    def __init__(self, root: Path, ttl_hours: int = 48) -> None:
        self.root = root
        self.ttl = timedelta(hours=ttl_hours)
        self._jobs: dict[str, JobRecord] = {}
        self._lock = Lock()
        self.root.mkdir(parents=True, exist_ok=True)
        self._load_existing()

    def _load_existing(self) -> None:
        for path in self.root.glob("*/job.json"):
            try:
                payload = json.loads(path.read_text())
                job = JobRecord.from_json(payload)
                if self._is_expired(job):
                    continue
                self._jobs[job.id] = job
            except Exception:
                continue

    def _is_expired(self, job: JobRecord) -> bool:
        horizon = datetime.now(timezone.utc) - self.ttl
        return job.submitted_at < horizon

    def _persist(self, job: JobRecord) -> None:
        job_dir = self.root / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text(json.dumps(job.to_dict(), indent=2))

    def create(self, *, input_path: Path) -> JobRecord:
        with self._lock:
            job_id = uuid4().hex
            job = JobRecord(id=job_id, status="queued", input_path=input_path)
            self._jobs[job_id] = job
            self._persist(job)
            return job

    def set_running(self, job_id: str) -> JobRecord:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            self._persist(job)
            return job

    def update_input_path(self, job_id: str, input_path: Path) -> JobRecord:
        with self._lock:
            job = self._jobs[job_id]
            job.input_path = input_path
            self._persist(job)
            return job

    def set_completed(self, job_id: str, output_path: Path) -> JobRecord:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "completed"
            job.output_path = output_path
            job.completed_at = datetime.now(timezone.utc)
            self._persist(job)
            return job

    def set_failed(self, job_id: str, error: str) -> JobRecord:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "failed"
            job.error = error
            job.completed_at = datetime.now(timezone.utc)
            self._persist(job)
            return job

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def cleanup_expired(self) -> None:
        with self._lock:
            expired = [jid for jid, job in self._jobs.items() if self._is_expired(job)]
            for jid in expired:
                job = self._jobs.pop(jid)
                job_dir = self.root / jid
                if job_dir.exists():
                    for f in job_dir.iterdir():
                        f.unlink()
                    job_dir.rmdir()


class JobRunner:
    """Runs jobs in the background using the event loop."""

    def __init__(self, store: JobStore) -> None:
        self.store = store
        self._semaphore = asyncio.Semaphore(2)

    async def run(self, job_id: str, coro_func) -> None:
        async with self._semaphore:
            self.store.set_running(job_id)
            try:
                await coro_func()
            except Exception as exc:  # pragma: no cover - logged by caller
                self.store.set_failed(job_id, str(exc))

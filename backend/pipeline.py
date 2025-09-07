import datetime
import threading
import uuid
from typing import Dict, Optional

from fastapi import HTTPException

from main.main import run_once_or_loop

state: Dict[str, object] = {
    "current_job_id": None,
    "jobs": {},
    "stop_event": threading.Event(),
}

_lock = threading.Lock()


def start_job() -> Dict[str, str]:
    with _lock:
        current = state["current_job_id"]
        if current:
            job = state["jobs"].get(current, {})
            raise HTTPException(status_code=409, detail={"job_id": current, "status": job.get("status", "running")})

        job_id = str(uuid.uuid4())
        state["current_job_id"] = job_id
        state["jobs"][job_id] = {
            "status": "queued",
            "start_time": None,
            "end_time": None,
            "last_log": "",
            "exit_code": None,
        }
        state["stop_event"].clear()
        thread = threading.Thread(target=run_pipeline, args=(job_id,), daemon=True)
        thread.start()
        return {"job_id": job_id, "status": "queued"}


def on_log_factory(job_id: str):
    def _on_log(msg: str):
        with _lock:
            job = state["jobs"].get(job_id)
            if job is not None:
                job["last_log"] = msg
    return _on_log


def run_pipeline(job_id: str) -> None:
    on_log = on_log_factory(job_id)
    with _lock:
        job = state["jobs"][job_id]
        job["status"] = "running"
        job["start_time"] = datetime.datetime.utcnow().isoformat()

    exit_code: Optional[int] = None
    try:
        exit_code = run_once_or_loop(state["stop_event"], on_log)
        if state["stop_event"].is_set():
            status = "stopped"
        elif exit_code == 0:
            status = "success"
        else:
            status = "failed"
    except Exception as e:  # noqa: BLE001
        on_log(f"ERROR: {e}")
        status = "failed"
        exit_code = 1

    with _lock:
        job = state["jobs"][job_id]
        job["status"] = status
        job["end_time"] = datetime.datetime.utcnow().isoformat()
        job["exit_code"] = exit_code
        state["current_job_id"] = None
        state["stop_event"].clear()


def get_status(job_id: str) -> Dict[str, object]:
    with _lock:
        job = state["jobs"].get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return {"job_id": job_id, **job}


def stop_job(job_id: str) -> Dict[str, object]:
    with _lock:
        if state["current_job_id"] != job_id:
            raise HTTPException(status_code=404, detail="job not running")
        state["stop_event"].set()
    return get_status(job_id)

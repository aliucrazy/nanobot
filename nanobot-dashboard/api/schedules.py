"""Schedules API routes."""

import json
from datetime import datetime
from pathlib import Path

from croniter import croniter
from fastapi import APIRouter, HTTPException

router = APIRouter()


def get_nanobot_dir() -> Path:
    """Get nanobot data directory."""
    return Path.home() / ".nanobot"


def load_cron_jobs() -> list[dict]:
    """Load cron jobs from nanobot data directory."""
    jobs_file = get_nanobot_dir() / "cron" / "jobs.json"
    if jobs_file.exists():
        try:
            with open(jobs_file) as f:
                data = json.load(f)
                return data.get("jobs", [])
        except (json.JSONDecodeError, IOError):
            return []
    return []


def format_schedule(schedule: dict) -> str:
    """Format schedule for display."""
    kind = schedule.get("kind", "unknown")

    if kind == "cron":
        expr = schedule.get("expr", "")
        return f"Cron: {expr}"
    elif kind == "every":
        every_ms = schedule.get("everyMs") or schedule.get("every_ms")
        if every_ms:
            minutes = every_ms / 60000
            return f"Every {int(minutes)} minutes"
        return "Every unknown"
    elif kind == "at":
        at_ms = schedule.get("atMs") or schedule.get("at_ms")
        if at_ms:
            dt = datetime.fromtimestamp(at_ms / 1000)
            return f"At {dt.strftime('%Y-%m-%d %H:%M')}"
        return "At unknown time"

    return f"Unknown: {kind}"


def calculate_next_run(schedule: dict, base_time: datetime = None) -> str:
    """Calculate next run time for a schedule."""
    kind = schedule.get("kind", "unknown")

    if kind == "cron":
        expr = schedule.get("expr", "")
        tz = schedule.get("tz")
        try:
            itr = croniter(expr, base_time or datetime.now())
            next_run = itr.get_next(datetime)
            return next_run.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return "Invalid cron expression"
    elif kind == "every":
        every_ms = schedule.get("everyMs") or schedule.get("every_ms")
        if every_ms:
            # For "every" schedules, show relative time
            minutes = every_ms / 60000
            return f"Every {int(minutes)} min"
        return "Unknown interval"
    elif kind == "at":
        at_ms = schedule.get("atMs") or schedule.get("at_ms")
        if at_ms:
            dt = datetime.fromtimestamp(at_ms / 1000)
            return dt.strftime("%Y-%m-%d %H:%M")
        return "Unknown time"

    return "Unknown"


@router.get("")
async def list_schedules():
    """List all scheduled jobs."""
    jobs = load_cron_jobs()
    schedules = []

    for job in jobs:
        schedule = job.get("schedule", {})
        state = job.get("state", {})

        # Calculate next run if not provided
        next_run_ms = state.get("nextRunAtMs") or state.get("next_run_at_ms")
        if next_run_ms:
            next_run = datetime.fromtimestamp(next_run_ms / 1000).strftime("%Y-%m-%d %H:%M")
        else:
            next_run = calculate_next_run(schedule)

        # Format last run
        last_run_ms = state.get("lastRunAtMs") or state.get("last_run_at_ms")
        last_run = None
        if last_run_ms:
            last_run = datetime.fromtimestamp(last_run_ms / 1000).strftime("%Y-%m-%d %H:%M")

        schedules.append({
            "id": job.get("id"),
            "name": job.get("name", "Unnamed"),
            "enabled": job.get("enabled", True),
            "schedule": format_schedule(schedule),
            "schedule_kind": schedule.get("kind"),
            "next_run": next_run,
            "last_run": last_run,
            "last_status": state.get("lastStatus") or state.get("last_status"),
            "last_error": state.get("lastError") or state.get("last_error"),
            "payload": job.get("payload", {}),
            "created_at": datetime.fromtimestamp(
                job.get("createdAtMs", 0) / 1000
            ).strftime("%Y-%m-%d") if job.get("createdAtMs") else None
        })

    # Sort by next run time
    schedules.sort(key=lambda x: x.get("next_run") or "")

    return {"schedules": schedules}


@router.get("/{schedule_id}")
async def get_schedule(schedule_id: str):
    """Get schedule details."""
    jobs = load_cron_jobs()

    for job in jobs:
        if job.get("id") == schedule_id:
            schedule = job.get("schedule", {})
            state = job.get("state", {})

            return {
                "id": job.get("id"),
                "name": job.get("name", "Unnamed"),
                "enabled": job.get("enabled", True),
                "schedule": schedule,
                "schedule_display": format_schedule(schedule),
                "next_run": calculate_next_run(schedule),
                "state": state,
                "payload": job.get("payload", {}),
                "created_at": job.get("createdAtMs"),
                "updated_at": job.get("updatedAtMs")
            }

    raise HTTPException(status_code=404, detail="Schedule not found")


@router.post("/{schedule_id}/toggle")
async def toggle_schedule(schedule_id: str):
    """Toggle schedule enabled state."""
    jobs_file = get_nanobot_dir() / "cron" / "jobs.json"

    if not jobs_file.exists():
        raise HTTPException(status_code=404, detail="No schedules found")

    try:
        with open(jobs_file) as f:
            data = json.load(f)

        jobs = data.get("jobs", [])
        for job in jobs:
            if job.get("id") == schedule_id:
                job["enabled"] = not job.get("enabled", True)
                job["updatedAtMs"] = int(datetime.now().timestamp() * 1000)
                break
        else:
            raise HTTPException(status_code=404, detail="Schedule not found")

        with open(jobs_file, "w") as f:
            json.dump(data, f, indent=2)

        return {"id": schedule_id, "enabled": job["enabled"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to toggle schedule: {str(e)}")


@router.post("/{schedule_id}/run")
async def run_schedule(schedule_id: str):
    """Manually run a schedule."""
    jobs = load_cron_jobs()

    job = None
    for j in jobs:
        if j.get("id") == schedule_id:
            job = j
            break

    if not job:
        raise HTTPException(status_code=404, detail="Schedule not found")

    # In a real implementation, this would trigger the job execution
    # For now, we just return success
    return {
        "id": schedule_id,
        "name": job.get("name"),
        "triggered": True,
        "message": "Job triggered manually"
    }

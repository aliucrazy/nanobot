"""Nanobot Dashboard - FastAPI main entry point."""

import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.skills import router as skills_router
from api.schedules import router as schedules_router
from api.logs import router as logs_router


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


def get_stats() -> dict:
    """Get dashboard statistics."""
    nanobot_dir = get_nanobot_dir()

    # Count skills
    skills_dir = nanobot_dir / "skills"
    skill_count = len([d for d in skills_dir.iterdir() if d.is_dir()]) if skills_dir.exists() else 0

    # Count jobs
    jobs = load_cron_jobs()
    job_count = len(jobs)

    # Count today's executions
    today = datetime.now().strftime("%Y-%m-%d")
    sessions_dir = nanobot_dir / "sessions"
    today_count = 0
    if sessions_dir.exists():
        for session_file in sessions_dir.glob("*.jsonl"):
            if today in session_file.name:
                today_count += 1

    # Calculate success rate
    success_count = sum(1 for j in jobs if j.get("state", {}).get("lastStatus") == "ok")
    success_rate = (success_count / len(jobs) * 100) if jobs else 0

    return {
        "skill_count": skill_count,
        "job_count": job_count,
        "today_count": today_count,
        "success_rate": round(success_rate, 1)
    }


def get_recent_executions(limit: int = 10) -> list[dict]:
    """Get recent job executions."""
    jobs = load_cron_jobs()
    executions = []

    for job in jobs:
        state = job.get("state", {})
        if state.get("lastRunAtMs"):
            executions.append({
                "name": job.get("name", "Unknown"),
                "status": state.get("lastStatus", "unknown"),
                "time": datetime.fromtimestamp(state.get("lastRunAtMs", 0) / 1000).strftime("%Y-%m-%d %H:%M"),
                "error": state.get("lastError")
            })

    executions.sort(key=lambda x: x["time"], reverse=True)
    return executions[:limit]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    yield


app = FastAPI(
    title="Nanobot Dashboard",
    description="Visualization panel for nanobot AI assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Include API routers
app.include_router(skills_router, prefix="/api/skills", tags=["skills"])
app.include_router(schedules_router, prefix="/api/schedules", tags=["schedules"])
app.include_router(logs_router, prefix="/api/logs", tags=["logs"])


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    stats = get_stats()
    recent = get_recent_executions()
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "stats": stats,
        "recent": recent
    })


@app.get("/skills", response_class=HTMLResponse)
async def skills_page(request: Request):
    """Skills management page."""
    return templates.TemplateResponse("skills.html", {"request": request})


@app.get("/schedules", response_class=HTMLResponse)
async def schedules_page(request: Request):
    """Schedules timeline page."""
    return templates.TemplateResponse("schedules.html", {"request": request})


@app.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request):
    """Real-time logs page."""
    return templates.TemplateResponse("logs.html", {"request": request})


@app.get("/api/stats")
async def api_stats():
    """Get dashboard statistics API."""
    return get_stats()


@app.get("/api/recent")
async def api_recent():
    """Get recent executions API."""
    return get_recent_executions()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)

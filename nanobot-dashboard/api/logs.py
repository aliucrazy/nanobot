"""Logs API routes."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

router = APIRouter()


def get_nanobot_dir() -> Path:
    """Get nanobot data directory."""
    return Path.home() / ".nanobot"


def get_session_files() -> list[Path]:
    """Get all session log files."""
    sessions_dir = get_nanobot_dir() / "sessions"
    if not sessions_dir.exists():
        return []
    return sorted(sessions_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def parse_log_entry(line: str, session_name: str) -> dict | None:
    """Parse a JSONL log entry."""
    try:
        entry = json.loads(line)

        # Skip metadata entries
        if entry.get("_type") == "metadata":
            return None

        # Determine log level
        level = "INFO"
        if "error" in entry.get("content", "").lower() or "Error" in entry.get("content", ""):
            level = "ERROR"
        elif "warning" in entry.get("content", "").lower():
            level = "WARNING"

        return {
            "timestamp": entry.get("timestamp", ""),
            "role": entry.get("role", "unknown"),
            "content": entry.get("content", ""),
            "session": session_name,
            "level": level
        }
    except json.JSONDecodeError:
        return None


@router.get("")
async def get_logs(
    session: str = Query(None, description="Filter by session name"),
    level: str = Query(None, description="Filter by log level"),
    limit: int = Query(100, description="Number of logs to return"),
    search: str = Query(None, description="Search in log content")
):
    """Get logs with optional filtering."""
    logs = []

    if session:
        # Read specific session
        session_file = get_nanobot_dir() / "sessions" / f"{session}.jsonl"
        if session_file.exists():
            files = [session_file]
        else:
            files = []
    else:
        # Read all sessions
        files = get_session_files()

    for session_file in files:
        session_name = session_file.stem
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                entry = parse_log_entry(line, session_name)
                if entry:
                    # Apply filters
                    if level and entry["level"] != level.upper():
                        continue
                    if search and search.lower() not in entry["content"].lower():
                        continue
                    logs.append(entry)
        except Exception:
            continue

    # Sort by timestamp (newest first) and limit
    logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    logs = logs[:limit]

    return {"logs": logs}


@router.get("/sessions")
async def get_sessions():
    """Get list of available sessions."""
    files = get_session_files()
    sessions = []

    for f in files:
        stat = f.stat()
        sessions.append({
            "name": f.stem,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        })

    return {"sessions": sessions}


async def log_stream():
    """Generate SSE stream for real-time logs."""
    # Track last position for each session file
    last_positions = {}

    while True:
        files = get_session_files()
        new_logs = []

        for session_file in files:
            session_name = session_file.stem
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    # Seek to last position if known
                    last_pos = last_positions.get(session_name, 0)
                    f.seek(last_pos)

                    new_lines = f.readlines()
                    if new_lines:
                        last_positions[session_name] = f.tell()

                        for line in new_lines:
                            entry = parse_log_entry(line, session_name)
                            if entry:
                                new_logs.append(entry)
            except Exception:
                continue

        if new_logs:
            # Sort by timestamp
            new_logs.sort(key=lambda x: x.get("timestamp", ""))
            for entry in new_logs:
                yield f"data: {json.dumps(entry)}\n\n"

        await asyncio.sleep(2)  # Poll every 2 seconds


@router.get("/stream")
async def stream_logs():
    """Stream logs in real-time using SSE."""
    return StreamingResponse(
        log_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

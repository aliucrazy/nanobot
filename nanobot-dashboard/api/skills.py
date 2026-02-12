"""Skills API routes."""

import json
import os
import shutil
from pathlib import Path

import frontmatter
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


def get_nanobot_dir() -> Path:
    """Get nanobot data directory."""
    return Path.home() / ".nanobot"


def get_skill_metadata(skill_path: Path) -> dict:
    """Extract metadata from skill markdown file."""
    try:
        post = frontmatter.load(skill_path)
        return {
            "name": post.get("name", skill_path.parent.name),
            "description": post.get("description", ""),
            "enabled": post.get("enabled", True),
            "triggers": post.get("triggers", []),
            "metadata": post.get("metadata", {})
        }
    except Exception:
        return {
            "name": skill_path.parent.name,
            "description": "",
            "enabled": True,
            "triggers": [],
            "metadata": {}
        }


def check_skill_requirements(skill_dir: Path) -> dict:
    """Check if skill requirements are met."""
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        return {"available": True, "missing": []}

    try:
        post = frontmatter.load(skill_file)
        nanobot_meta = post.get("nanobot", {})
        requires = nanobot_meta.get("requires", {}) if isinstance(nanobot_meta, dict) else {}

        missing = []
        for binary in requires.get("bins", []):
            if not shutil.which(binary):
                missing.append(f"CLI: {binary}")
        for env_var in requires.get("env", []):
            if not os.environ.get(env_var):
                missing.append(f"ENV: {env_var}")

        return {
            "available": len(missing) == 0,
            "missing": missing
        }
    except Exception:
        return {"available": True, "missing": []}


@router.get("")
async def list_skills():
    """List all skills."""
    skills_dir = get_nanobot_dir() / "skills"
    skills = []

    if not skills_dir.exists():
        return {"skills": skills}

    for skill_dir in skills_dir.iterdir():
        if skill_dir.is_dir():
            skill_file = skill_dir / "SKILL.md"
            if skill_file.exists():
                metadata = get_skill_metadata(skill_file)
                requirements = check_skill_requirements(skill_dir)

                # Check for hooks.py
                has_hooks = (skill_dir / "hooks.py").exists()

                skills.append({
                    "id": skill_dir.name,
                    "name": metadata.get("name", skill_dir.name),
                    "description": metadata.get("description", ""),
                    "enabled": metadata.get("enabled", True),
                    "triggers": metadata.get("triggers", []),
                    "has_hooks": has_hooks,
                    "available": requirements["available"],
                    "missing_requirements": requirements["missing"]
                })

    return {"skills": skills}


@router.get("/{skill_id}")
async def get_skill(skill_id: str):
    """Get skill details."""
    skill_dir = get_nanobot_dir() / "skills" / skill_id
    skill_file = skill_dir / "SKILL.md"

    if not skill_file.exists():
        raise HTTPException(status_code=404, detail="Skill not found")

    metadata = get_skill_metadata(skill_file)
    requirements = check_skill_requirements(skill_dir)

    # Read full content
    content = skill_file.read_text(encoding="utf-8")

    # Check for hooks
    hooks_content = None
    hooks_file = skill_dir / "hooks.py"
    if hooks_file.exists():
        hooks_content = hooks_file.read_text(encoding="utf-8")

    return {
        "id": skill_id,
        "name": metadata.get("name", skill_id),
        "description": metadata.get("description", ""),
        "enabled": metadata.get("enabled", True),
        "triggers": metadata.get("triggers", []),
        "metadata": metadata.get("metadata", {}),
        "content": content,
        "hooks": hooks_content,
        "available": requirements["available"],
        "missing_requirements": requirements["missing"]
    }


@router.post("/{skill_id}/toggle")
async def toggle_skill(skill_id: str):
    """Toggle skill enabled state."""
    skill_dir = get_nanobot_dir() / "skills" / skill_id
    skill_file = skill_dir / "SKILL.md"

    if not skill_file.exists():
        raise HTTPException(status_code=404, detail="Skill not found")

    try:
        post = frontmatter.load(skill_file)
        current_enabled = post.get("enabled", True)
        post["enabled"] = not current_enabled

        with open(skill_file, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))

        return {"id": skill_id, "enabled": not current_enabled}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to toggle skill: {str(e)}")


@router.post("/{skill_id}/trigger")
async def trigger_skill(skill_id: str):
    """Manually trigger a skill."""
    skill_dir = get_nanobot_dir() / "skills" / skill_id

    if not skill_dir.exists():
        raise HTTPException(status_code=404, detail="Skill not found")

    # Check for trigger script
    trigger_script = skill_dir / "scripts" / "trigger.py"
    if trigger_script.exists():
        # Execute trigger script asynchronously
        import asyncio
        try:
            proc = await asyncio.create_subprocess_exec(
                "python", str(trigger_script),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                return {
                    "id": skill_id,
                    "triggered": True,
                    "output": stdout.decode() if stdout else None
                }
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Trigger failed: {stderr.decode() if stderr else 'Unknown error'}"
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to trigger skill: {str(e)}")

    return {"id": skill_id, "triggered": False, "message": "No trigger script found"}

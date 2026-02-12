"""Hooks system for nanobot extensions.

Skills can register hooks to extend nanobot functionality without modifying core code.

Example skill hooks.py:
    from nanobot.hooks import register, Context

    @register("after_conversation")
    async def reflect(context: Context, messages: list[dict]) -> None:
        # Run reflection after each conversation
        pass

    @register("daily_digest")
    async def digest(context: Context) -> None:
        # Run daily at scheduled time
        pass
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Coroutine

from loguru import logger

# Type definitions
HookHandler = Callable[..., Coroutine[Any, Any, Any]]


@dataclass
class Context:
    """Context passed to hook handlers."""

    workspace: Path
    provider: Any  # LLMProvider
    model: str | None
    memory: Any  # MemoryStore


class HookRegistry:
    """Registry for hook handlers."""

    _hooks: dict[str, list[HookHandler]] = {}

    @classmethod
    def register(cls, event: str) -> Callable[[HookHandler], HookHandler]:
        """Decorator to register a hook handler.

        Args:
            event: Event name (e.g., "after_conversation", "daily_digest")

        Returns:
            Decorator function
        """

        def decorator(handler: HookHandler) -> HookHandler:
            if event not in cls._hooks:
                cls._hooks[event] = []
            cls._hooks[event].append(handler)
            logger.debug(f"Registered hook '{event}' from {handler.__module__}")
            return handler

        return decorator

    @classmethod
    async def trigger(cls, event: str, context: Context, *args: Any, **kwargs: Any) -> None:
        """Trigger all handlers for an event.

        Args:
            event: Event name
            context: Execution context
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        handlers = cls._hooks.get(event, [])
        if not handlers:
            return

        logger.debug(f"Triggering {len(handlers)} handlers for '{event}'")

        for handler in handlers:
            try:
                # Check if handler is async
                if inspect.iscoroutinefunction(handler):
                    await handler(context, *args, **kwargs)
                else:
                    handler(context, *args, **kwargs)
            except Exception as e:
                logger.warning(f"Hook handler '{handler.__name__}' failed: {e}")

    @classmethod
    def clear(cls) -> None:
        """Clear all registered hooks."""
        cls._hooks.clear()

    @classmethod
    def list_events(cls) -> list[str]:
        """List all registered events."""
        return list(cls._hooks.keys())


class SkillLoader:
    """Loader for skill hooks."""

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir

    def load_all(self) -> int:
        """Load all skills from the skills directory.

        Returns:
            Number of skills loaded
        """
        if not self.skills_dir.exists():
            logger.debug(f"Skills directory not found: {self.skills_dir}")
            return 0

        count = 0
        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue

            hooks_file = skill_dir / "hooks.py"
            if hooks_file.exists():
                self._load_skill_hooks(skill_dir.name, hooks_file)
                count += 1

        logger.info(f"Loaded {count} skill(s) with hooks")
        return count

    def _load_skill_hooks(self, skill_name: str, hooks_file: Path) -> None:
        """Load hooks from a skill's hooks.py file."""
        try:
            spec = importlib.util.spec_from_file_location(
                f"skill.{skill_name}", hooks_file
            )
            if not spec or not spec.loader:
                logger.warning(f"Cannot load hooks for skill: {skill_name}")
                return

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            logger.debug(f"Loaded hooks from skill: {skill_name}")
        except Exception as e:
            logger.error(f"Failed to load hooks for skill '{skill_name}': {e}")


# Convenience function for skills to use
register = HookRegistry.register

__all__ = ["HookRegistry", "SkillLoader", "Context", "register"]

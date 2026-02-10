"""Post-conversation reflection module.

Analyzes conversations to extract insights, corrections, and improvements.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider
from nanobot.utils.helpers import ensure_dir, today_date


@dataclass
class ReflectionResult:
    """Result of a reflection analysis."""

    timestamp: str
    user_feedback: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    new_knowledge: list[str] = field(default_factory=list)
    summary: str = ""

    def to_markdown(self) -> str:
        """Convert reflection to markdown format."""
        lines = [
            f"## Reflection: {self.timestamp}",
            "",
            f"**Summary**: {self.summary}",
            "",
        ]

        if self.user_feedback:
            lines.extend(["### User Feedback/Corrections", ""])
            for item in self.user_feedback:
                lines.append(f"- {item}")
            lines.append("")

        if self.improvements:
            lines.extend(["### Areas for Improvement", ""])
            for item in self.improvements:
                lines.append(f"- {item}")
            lines.append("")

        if self.new_knowledge:
            lines.extend(["### New Knowledge Gained", ""])
            for item in self.new_knowledge:
                lines.append(f"- {item}")
            lines.append("")

        return "\n".join(lines)


class ReflectionEngine:
    """
    Post-conversation reflection engine.

    Analyzes conversation history to extract:
    - User corrections and feedback
    - Response quality issues
    - New knowledge acquired
    """

    REFLECTION_PROMPT = """You are a reflection analyzer for an AI assistant. Analyze the following conversation and extract insights.

Your task is to identify:
1. **User Feedback/Corrections**: Any corrections, clarifications, or feedback the user provided about your responses
2. **Areas for Improvement**: Defects, inaccuracies, or ways your responses could have been better
3. **New Knowledge**: Facts, patterns, or information learned from this conversation

Respond in JSON format:
{
    "user_feedback": ["list of feedback items"],
    "improvements": ["list of improvement areas"],
    "new_knowledge": ["list of new knowledge items"],
    "summary": "Brief one-sentence summary of the conversation outcome"
}

If a category has no items, return an empty array. Be specific and actionable in your observations.

Conversation to analyze:
"""

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str | None = None,
    ):
        self.workspace = workspace
        self.provider = provider
        self.model = model or provider.get_default_model()
        self.memory_dir = ensure_dir(workspace / "memory")
        self.reflection_file = self.memory_dir / "REFLECTION.md"

    async def reflect_after_conversation(
        self,
        messages: list[dict[str, Any]],
        session_key: str,
    ) -> ReflectionResult | None:
        """
        Perform reflection after a conversation ends.

        Args:
            messages: The conversation messages (user and assistant exchanges).
            session_key: Identifier for the session.

        Returns:
            ReflectionResult if reflection was performed, None if skipped.
        """
        # Skip reflection for very short conversations
        if len(messages) < 2:
            logger.debug("Skipping reflection: conversation too short")
            return None

        # Build conversation text
        conversation_text = self._format_conversation(messages)

        try:
            # Call LLM for reflection analysis
            response = await self.provider.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a reflection analyzer. Output valid JSON only.",
                    },
                    {
                        "role": "user",
                        "content": self.REFLECTION_PROMPT + conversation_text,
                    },
                ],
                model=self.model,
                temperature=0.3,
                max_tokens=2048,
            )

            if not response.content:
                logger.warning("Reflection received empty response")
                return None

            # Parse reflection result
            result = self._parse_reflection(response.content)
            result.timestamp = datetime.now().isoformat()

            # Write to reflection file
            self._append_reflection(result)

            logger.info(f"Reflection completed for session {session_key}")
            return result

        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            return None

    def _format_conversation(self, messages: list[dict[str, Any]]) -> str:
        """Format messages for reflection analysis."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")

        return "\n\n".join(lines)

    def _parse_reflection(self, content: str) -> ReflectionResult:
        """Parse LLM response into ReflectionResult."""
        # Try to extract JSON from markdown code blocks
        json_str = content
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(json_str)
            return ReflectionResult(
                timestamp=datetime.now().isoformat(),
                user_feedback=data.get("user_feedback", []),
                improvements=data.get("improvements", []),
                new_knowledge=data.get("new_knowledge", []),
                summary=data.get("summary", ""),
            )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse reflection JSON: {e}")
            # Return empty result with raw content as summary
            return ReflectionResult(
                timestamp=datetime.now().isoformat(),
                summary="Reflection parsing failed",
            )

    def _append_reflection(self, result: ReflectionResult) -> None:
        """Append reflection to the reflection file."""
        content = result.to_markdown()

        if self.reflection_file.exists():
            existing = self.reflection_file.read_text(encoding="utf-8")
            content = content + "\n\n---\n\n" + existing
        else:
            # Add header for new file
            header = "# Conversation Reflections\n\n"
            content = header + content

        self.reflection_file.write_text(content, encoding="utf-8")

    def read_reflections(self, days: int = 7) -> str:
        """
        Read recent reflections.

        Args:
            days: Number of days to look back.

        Returns:
            Reflection content or empty string.
        """
        if not self.reflection_file.exists():
            return ""

        content = self.reflection_file.read_text(encoding="utf-8")

        # TODO: Filter by date if needed
        # For now, return all content
        return content

    def get_weekly_summary(self) -> dict[str, Any] | None:
        """
        Generate a weekly summary of reflections.

        Returns:
            Summary dict with aggregated insights, or None if no data.
        """
        if not self.reflection_file.exists():
            return None

        content = self.reflection_file.read_text(encoding="utf-8")

        # Parse reflections (simple parsing)
        reflections = self._parse_reflections_content(content)

        # Filter to last 7 days
        cutoff = datetime.now() - timedelta(days=7)
        recent = [
            r for r in reflections
            if datetime.fromisoformat(r.timestamp) >= cutoff
        ]

        if not recent:
            return None

        # Aggregate
        all_feedback = []
        all_improvements = []
        all_knowledge = []

        for r in recent:
            all_feedback.extend(r.user_feedback)
            all_improvements.extend(r.improvements)
            all_knowledge.extend(r.new_knowledge)

        return {
            "period": "last_7_days",
            "reflection_count": len(recent),
            "user_feedback": list(set(all_feedback)),
            "improvements": list(set(all_improvements)),
            "new_knowledge": list(set(all_knowledge)),
        }

    def _parse_reflections_content(self, content: str) -> list[ReflectionResult]:
        """Parse reflection file content into ReflectionResult objects."""
        reflections = []

        # Split by separator
        sections = content.split("\n\n---\n\n")

        for section in sections:
            if not section.strip() or section.startswith("# "):
                continue

            # Simple parsing - extract timestamp and lists
            lines = section.strip().split("\n")
            timestamp = ""
            user_feedback = []
            improvements = []
            new_knowledge = []
            summary = ""

            current_section = None

            for line in lines:
                line = line.strip()
                if line.startswith("## Reflection:"):
                    timestamp = line.replace("## Reflection:", "").strip()
                elif line.startswith("**Summary**:"):
                    summary = line.replace("**Summary**:", "").strip()
                elif line == "### User Feedback/Corrections":
                    current_section = "feedback"
                elif line == "### Areas for Improvement":
                    current_section = "improvements"
                elif line == "### New Knowledge Gained":
                    current_section = "knowledge"
                elif line.startswith("-") and current_section:
                    item = line[1:].strip()
                    if current_section == "feedback":
                        user_feedback.append(item)
                    elif current_section == "improvements":
                        improvements.append(item)
                    elif current_section == "knowledge":
                        new_knowledge.append(item)

            if timestamp:
                reflections.append(
                    ReflectionResult(
                        timestamp=timestamp,
                        user_feedback=user_feedback,
                        improvements=improvements,
                        new_knowledge=new_knowledge,
                        summary=summary,
                    )
                )

        return reflections

    async def generate_weekly_reflection_update(self) -> str | None:
        """
        Generate content for updating SOUL.md or skills based on weekly reflections.

        Returns:
            Markdown content for updating SOUL.md, or None if no data.
        """
        summary = self.get_weekly_summary()
        if not summary:
            return None

        lines = [
            f"## Weekly Reflection Summary ({summary['period']})",
            f"",
            f"Based on {summary['reflection_count']} conversations this week.",
            f"",
        ]

        if summary["user_feedback"]:
            lines.extend(["### User Feedback Patterns", ""])
            for item in summary["user_feedback"][:10]:  # Limit to top 10
                lines.append(f"- {item}")
            lines.append("")

        if summary["improvements"]:
            lines.extend(["### Areas for Improvement", ""])
            for item in summary["improvements"][:10]:
                lines.append(f"- {item}")
            lines.append("")

        if summary["new_knowledge"]:
            lines.extend(["### New Knowledge", ""])
            for item in summary["new_knowledge"][:10]:
                lines.append(f"- {item}")
            lines.append("")

        return "\n".join(lines)

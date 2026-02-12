"""Claude tool - call Claude AI for various tasks."""

from typing import Any

from nanobot.agent.tools.base import Tool


class ClaudeTool(Tool):
    """
    Tool to call Claude AI for various tasks.

    Use this when you need Claude's help for:
    - Complex analysis or reasoning
    - Creative writing or editing
    - Code review or explanation
    - Summarization of large texts
    - Any task where you want a second AI opinion
    """

    name = "claude"
    description = "Call Claude AI to perform various tasks like analysis, writing, coding, summarization, etc."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The task or question for Claude",
                },
                "context": {
                    "type": "string",
                    "description": "Additional context or background information (optional)",
                },
                "task_type": {
                    "type": "string",
                    "enum": ["analysis", "writing", "coding", "summarize", "explain", "review", "general"],
                    "description": "Type of task to help Claude understand the goal",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens in response (default: 4000)",
                    "minimum": 100,
                    "maximum": 8000,
                },
            },
            "required": ["prompt"],
        }

    async def execute(
        self,
        prompt: str,
        context: str = "",
        task_type: str = "general",
        max_tokens: int = 4000,
    ) -> str:
        """Execute Claude call via the configured provider."""
        # Import here to avoid circular dependency
        from nanobot.config.loader import load_config
        from nanobot.providers.litellm_provider import LiteLLMProvider

        config = load_config()

        # Get provider config
        provider_config = config.get_provider()
        if not provider_config or not provider_config.api_key:
            return "Error: No API key configured for Claude"

        # Create provider instance
        provider = LiteLLMProvider(
            api_key=provider_config.api_key,
            api_base=config.get_api_base(),
            default_model=config.agents.defaults.model,
            extra_headers=provider_config.extra_headers,
            provider_name=config.get_provider_name(),
        )

        # Build the full prompt
        full_prompt = f"""[Task Type: {task_type}]

{prompt}
"""
        if context:
            full_prompt += f"""

[Context]
{context}
"""

        try:
            response = await provider.chat(
                messages=[
                    {"role": "system", "content": "You are Claude, a helpful AI assistant. Provide concise, accurate responses."},
                    {"role": "user", "content": full_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )

            if response.finish_reason == "error":
                return f"Error calling Claude: {response.content}"

            result = response.content or ""

            # Include reasoning if available
            if response.reasoning_content:
                result = f"[Claude's reasoning]\n{response.reasoning_content}\n\n[Response]\n{result}"

            return result

        except Exception as e:
            return f"Error calling Claude: {str(e)}"

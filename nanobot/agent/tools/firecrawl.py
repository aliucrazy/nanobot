"""Firecrawl tools for web scraping, search, and data extraction."""

import os
from typing import Any

import httpx

from nanobot.agent.tools.base import Tool


class FirecrawlScrapeTool(Tool):
    """Scrape a single webpage with JavaScript rendering support."""

    name = "firecrawl_scrape"
    description = "Scrape content from a single URL. Supports markdown, HTML, screenshots, and structured data extraction."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to scrape",
                },
                "formats": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["markdown", "html", "rawHtml", "screenshot", "links", "summary"]},
                    "description": "Output formats (default: ['markdown'])",
                },
                "onlyMainContent": {
                    "type": "boolean",
                    "description": "Extract only main content (default: true)",
                },
                "waitFor": {
                    "type": "integer",
                    "description": "Wait time in ms for JS rendering",
                },
            },
            "required": ["url"],
        }

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("FIRECRAWL_API_KEY", "")
        self.base_url = "https://api.firecrawl.dev/v1"

    async def execute(
        self,
        url: str,
        formats: list[str] | None = None,
        onlyMainContent: bool = True,
        waitFor: int | None = None,
    ) -> str:
        if not self.api_key:
            return "Error: FIRECRAWL_API_KEY not configured. Get one at https://firecrawl.dev"

        try:
            payload: dict[str, Any] = {
                "url": url,
                "formats": formats or ["markdown"],
                "onlyMainContent": onlyMainContent,
            }
            if waitFor:
                payload["waitFor"] = waitFor

            async with httpx.AsyncClient() as client:
                r = await client.post(
                    f"{self.base_url}/scrape",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=60.0,
                )
                r.raise_for_status()

            data = r.json()
            if not data.get("success"):
                return f"Error: {data.get('error', 'Unknown error')}"

            result = data.get("data", {})
            output = []

            if "markdown" in result:
                output.append(f"# Content\n{result['markdown'][:15000]}")
            if "html" in result:
                output.append(f"\n[HTML available: {len(result['html'])} chars]")
            if "screenshot" in result:
                output.append(f"\n[Screenshot URL: {result['screenshot'][:100]}...]")
            if "links" in result:
                links = result["links"]
                output.append(f"\n# Links ({len(links)} total)")
                for link in links[:20]:
                    output.append(f"- {link}")

            return "\n".join(output) if output else "No content extracted"

        except Exception as e:
            return f"Error scraping URL: {str(e)}"


class FirecrawlSearchTool(Tool):
    """Search the web and optionally scrape results."""

    name = "firecrawl_search"
    description = "Search the web. Optionally scrape search results for full content."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of results (default: 5)",
                    "minimum": 1,
                    "maximum": 10,
                },
                "scrapeResults": {
                    "type": "boolean",
                    "description": "Scrape full content of results (default: false)",
                },
                "formats": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["markdown", "html", "summary"]},
                    "description": "Formats when scraping results",
                },
            },
            "required": ["query"],
        }

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("FIRECRAWL_API_KEY", "")
        self.base_url = "https://api.firecrawl.dev/v1"

    async def execute(
        self,
        query: str,
        limit: int = 5,
        scrapeResults: bool = False,
        formats: list[str] | None = None,
    ) -> str:
        if not self.api_key:
            return "Error: FIRECRAWL_API_KEY not configured. Get one at https://firecrawl.dev"

        try:
            params: dict[str, Any] = {"q": query, "limit": min(limit, 10)}
            if scrapeResults:
                params["scrapeOptions"] = {"formats": formats or ["markdown"]}

            async with httpx.AsyncClient() as client:
                r = await client.get(
                    f"{self.base_url}/search",
                    params=params,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=60.0,
                )
                r.raise_for_status()

            data = r.json()
            if not data.get("success"):
                return f"Error: {data.get('error', 'Unknown error')}"

            results = data.get("data", [])
            if not results:
                return f"No results for: {query}"

            output = [f"# Search Results: {query}\n"]
            for i, item in enumerate(results[:limit], 1):
                output.append(f"{i}. **{item.get('title', 'No title')}**")
                output.append(f"   URL: {item.get('url', '')}")
                if desc := item.get("description"):
                    output.append(f"   {desc[:200]}")

                # If scraped content available
                if "markdown" in item:
                    output.append(f"\n   Content preview:")
                    content = item["markdown"][:500]
                    output.append(f"   ```{content}```")
                output.append("")

            return "\n".join(output)

        except Exception as e:
            return f"Error searching: {str(e)}"


class FirecrawlMapTool(Tool):
    """Map/discover all URLs on a website."""

    name = "firecrawl_map"
    description = "Discover all URLs on a website. Useful for sitemap generation or site analysis."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Website URL to map (e.g., https://example.com)",
                },
                "search": {
                    "type": "string",
                    "description": "Optional search term to filter URLs",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max URLs to return (default: 100)",
                    "minimum": 1,
                    "maximum": 500,
                },
            },
            "required": ["url"],
        }

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("FIRECRAWL_API_KEY", "")
        self.base_url = "https://api.firecrawl.dev/v1"

    async def execute(
        self,
        url: str,
        search: str = "",
        limit: int = 100,
    ) -> str:
        if not self.api_key:
            return "Error: FIRECRAWL_API_KEY not configured"

        try:
            params: dict[str, Any] = {"url": url, "limit": min(limit, 500)}
            if search:
                params["search"] = search

            async with httpx.AsyncClient() as client:
                r = await client.get(
                    f"{self.base_url}/map",
                    params=params,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=60.0,
                )
                r.raise_for_status()

            data = r.json()
            if not data.get("success"):
                return f"Error: {data.get('error', 'Unknown error')}"

            links = data.get("links", [])
            output = [f"# Site Map: {url}", f"Found {len(links)} URLs\n"]

            for link in links[:limit]:
                output.append(f"- {link}")

            return "\n".join(output)

        except Exception as e:
            return f"Error mapping site: {str(e)}"


class FirecrawlExtractTool(Tool):
    """Extract structured data from web pages using LLM."""

    name = "firecrawl_extract"
    description = "Extract structured data from web pages using AI. Define a schema to get specific data fields."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of URLs to extract from",
                },
                "prompt": {
                    "type": "string",
                    "description": "What data to extract (e.g., 'Extract product name, price, and description')",
                },
                "schema": {
                    "type": "object",
                    "description": "JSON schema for structured extraction (optional)",
                },
            },
            "required": ["urls", "prompt"],
        }

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("FIRECRAWL_API_KEY", "")
        self.base_url = "https://api.firecrawl.dev/v1"

    async def execute(
        self,
        urls: list[str],
        prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> str:
        if not self.api_key:
            return "Error: FIRECRAWL_API_KEY not configured"

        try:
            payload: dict[str, Any] = {
                "urls": urls,
                "prompt": prompt,
            }
            if schema:
                payload["schema"] = schema

            async with httpx.AsyncClient() as client:
                r = await client.post(
                    f"{self.base_url}/extract",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=120.0,
                )
                r.raise_for_status()

            data = r.json()
            if not data.get("success"):
                return f"Error: {data.get('error', 'Unknown error')}"

            result = data.get("data", {})
            output = ["# Extracted Data\n"]

            if "extracted_data" in result:
                extracted = result["extracted_data"]
                if isinstance(extracted, list):
                    for i, item in enumerate(extracted, 1):
                        output.append(f"## Item {i}")
                        for key, value in item.items():
                            output.append(f"- **{key}**: {value}")
                        output.append("")
                else:
                    for key, value in extracted.items():
                        output.append(f"- **{key}**: {value}")

            return "\n".join(output)

        except Exception as e:
            return f"Error extracting data: {str(e)}"

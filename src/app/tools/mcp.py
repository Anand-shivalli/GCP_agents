"""Load tools from MCP servers via `langchain.mcp.MCPAdapter`.

`langchain.mcp` (built on FastMCP) replaces the deprecated
`langchain-mcp-adapters` package. It is marked beta upstream, so we silence
the beta warning here and keep every MCP call behind this one module - if the
API shifts, only this file changes.

MCP servers are separate services (see mcp_servers/). The agent process stays
stateless: tools are discovered at startup and each replica connects on its own.
"""

import asyncio
import json
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langchain_core._api import LangChainBetaWarning
from langchain_core.tools import BaseTool

with warnings.catch_warnings():
    warnings.simplefilter("ignore", LangChainBetaWarning)
    from langchain.mcp import MCPAdapter


@asynccontextmanager
async def mcp_tools(config_path: str) -> AsyncIterator[list[BaseTool]]:
    """Yield LangChain tools for every server in an MCP config file.

    The connection stays open for the lifetime of the context (the app
    lifespan), so tool calls reuse sessions instead of reconnecting each time.
    """
    text = await asyncio.to_thread(Path(config_path).read_text, encoding="utf-8")
    config = json.loads(text)
    if not config.get("mcpServers"):
        yield []
        return
    async with MCPAdapter(config) as adapter:
        yield await adapter.list_tools()

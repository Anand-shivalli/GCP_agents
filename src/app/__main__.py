"""`python -m app` - local dev entry point (works on Windows too)."""

import asyncio
import sys

import uvicorn

if __name__ == "__main__":
    if sys.platform == "win32":
        # psycopg's async driver cannot run on the default Proactor loop.
        # Consequence: stdio MCP servers (subprocesses) need Linux/WSL; use HTTP MCP locally.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, loop="none")

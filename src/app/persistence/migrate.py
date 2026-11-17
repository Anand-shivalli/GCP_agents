"""Create/upgrade checkpoint tables. Run once per release: `python -m app.persistence.migrate`."""

import asyncio
import sys

from app.core.config import get_settings
from app.persistence.checkpointer import build_checkpointer, build_pool


async def main() -> None:
    settings = get_settings()
    async with build_pool(settings) as pool:
        await build_checkpointer(pool, settings).setup()
    print("checkpoint tables are up to date")


if __name__ == "__main__":
    if sys.platform == "win32":  # psycopg async needs the selector loop on Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

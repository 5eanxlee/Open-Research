from __future__ import annotations

import asyncio
import signal

from .config import get_settings
from .runtime import ResearchRuntime


async def _run() -> None:
    settings = get_settings().model_copy(update={"process_role": "worker"})
    runtime = ResearchRuntime.build(settings)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:  # pragma: no cover - platform specific
            pass

    await runtime.init()
    try:
        await stop_event.wait()
    finally:
        await runtime.shutdown()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()

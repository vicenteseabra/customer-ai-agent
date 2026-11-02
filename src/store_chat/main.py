"""Entry point for running the Store Chat server locally.

Binds to 127.0.0.1 only: this is a local, self-hosted tool and must never
be reachable from another machine on the network.
"""

from __future__ import annotations

import uvicorn

from store_chat.api import build_app


def main() -> None:
    """Run the Store Chat server, bound to 127.0.0.1 only."""
    app = build_app()
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()

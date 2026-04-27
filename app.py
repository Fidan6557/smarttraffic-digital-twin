"""Compatibility entrypoint for SmartTraffic Digital Twin.

The production demo lives in main.py. This file exists so platforms that
auto-detect `app.py` can still import the FastAPI application.
"""

import uvicorn

from main import app, get_server_config


if __name__ == "__main__":
    host, port, display_addr = get_server_config()
    print(f"SmartTraffic Digital Twin running at http://{display_addr}", flush=True)
    uvicorn.run(app, host=host, port=port)

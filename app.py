"""Compatibility entrypoint for SmartTraffic Digital Twin.

The production demo lives in main.py. This file exists so platforms that
auto-detect `app.py` can still import the FastAPI application.
"""

import uvicorn

from main import app, find_available_port


if __name__ == "__main__":
    port = find_available_port(8003)
    print(f"SmartTraffic Digital Twin running at http://127.0.0.1:{port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)

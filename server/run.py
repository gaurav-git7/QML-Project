"""Start the dashboard: API + the built React app.

    python server/run.py                # http://127.0.0.1:8000
    python server/run.py --port 8001
    python server/run.py --quick        # show the smoke-test results in artifacts_quick/
"""
import os
import sys

if "--quick" in sys.argv:
    os.environ["QML_QUICK"] = "1"
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn

if __name__ == "__main__":
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000
    print(f"\nQuantumML Diagnostic Workstation -> http://127.0.0.1:{port}\n")
    uvicorn.run("server.main:app", host="127.0.0.1", port=port, log_level="warning")

"""Point d'entrée unique de l'application Safety AI."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(port: int = 8501) -> int:
    """Lance l'application Streamlit."""
    env = os.environ.copy()
    pp = os.pathsep.join([str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "configs")])
    env["PYTHONPATH"] = f"{pp}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", str(port)],
        cwd=PROJECT_ROOT,
        env=env,
    ).returncode


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    sys.exit(run(port))

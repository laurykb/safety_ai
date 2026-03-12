"""Bootstrap pour les scripts — configure sys.path et PROJECT_ROOT."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "configs") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "configs"))

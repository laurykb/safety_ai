#!/usr/bin/env python3
"""
Point d'entree unique cross-platform (Windows, Linux, macOS).
Usage:
  python run.py                    # Lancer l'app (defaut)
  python run.py pipeline [N]       # Pipeline Safety AI (N echantillons)
  python run.py research-download  # Datasets recherche (CB1, sarcasme, etc.)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHONPATH = os.pathsep.join([str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "configs"), str(PROJECT_ROOT / "scripts")])


def _env() -> dict:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{PYTHONPATH}{os.pathsep}{existing}" if existing else PYTHONPATH
    return env


def _run(cmd: list[str], **kw) -> int:
    return subprocess.run(cmd, cwd=PROJECT_ROOT, env=_env(), **kw).returncode


def cmd_dev(port: int = 8501) -> int:
    """Lance Streamlit (app principale) via l'entrée unique."""
    from app.main import run
    return run(port)


def cmd_test() -> int:
    """Lance les tests."""
    return _run([sys.executable, "-m", "pytest", "tests", "-v", "--tb=short"])


def cmd_pipeline(n: int = 2000) -> int:
    """Pipeline Safety AI end-to-end (charge, embed, train)."""
    return _run([sys.executable, "scripts/run_pipeline.py", "-n", str(n)])


def cmd_research_download() -> int:
    """Télécharge les datasets recherche (CB1, sarcasme)."""
    return _run([sys.executable, "scripts/download_research_datasets.py"])


def cmd_embed(embedding: str = "tfidf", n: int = 1000) -> int:
    """Embedding + train rapide (run_embedding)."""
    return _run([sys.executable, "scripts/run_embedding.py", embedding, "-n", str(n)])


def cmd_aggregate() -> int:
    """Agrège les rapports et génère des graphiques."""
    return _run([sys.executable, "scripts/aggregate_results.py"])


def cmd_check_duplicates() -> int:
    """Vérifie les doublons dans les datasets."""
    return _run([sys.executable, "scripts/check_duplicates.py"])


def cmd_install() -> int:
    """Installe les dépendances (pip install -r requirements.txt)."""
    return _run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def main() -> int:
    commands = {
        "dev": (cmd_dev, "Lancer l'app Streamlit"),
        "install": (cmd_install, "Installer les dépendances"),
        "test": (cmd_test, "Tests unitaires"),
        "pipeline": (cmd_pipeline, "Pipeline Safety AI"),
        "embed": (None, "Embedding + train (embed tfidf|...)"),
        "aggregate": (cmd_aggregate, "Agréger rapports et graphiques"),
        "research-download": (cmd_research_download, "Telecharger datasets recherche"),
        "check-duplicates": (cmd_check_duplicates, "Verifier doublons dans les datasets"),
    }

    if len(sys.argv) < 2:
        return cmd_dev()

    arg = sys.argv[1].lower()
    if arg in ("--port", "-p") and len(sys.argv) >= 3:
        try:
            return cmd_dev(int(sys.argv[2]))
        except ValueError:
            print("Port invalide", file=sys.stderr)
            return 1
    if arg in ("-h", "--help"):
        print("Usage: python run.py [commande] [options]")
        print("\nOptions globales:")
        print("  --port N, -p N    Port Streamlit (défaut: 8501)")
        print("\nCommandes:")
        for name, (_, desc) in commands.items():
            print(f"  {name:<20} {desc}")
        return 0

    if arg not in commands:
        print(f"Commande inconnue: {arg}", file=sys.stderr)
        print("Commandes: " + ", ".join(commands.keys()), file=sys.stderr)
        return 1

    fn, _ = commands[arg]
    if arg == "embed":
        emb = sys.argv[2] if len(sys.argv) > 2 else "tfidf"
        n = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 1000
        return cmd_embed(emb, n)
    if arg == "dev":
        port = 8501
        if "--port" in sys.argv or "-p" in sys.argv:
            try:
                i = sys.argv.index("--port") if "--port" in sys.argv else sys.argv.index("-p")
                port = int(sys.argv[i + 1])
            except (ValueError, IndexError):
                print("Port invalide", file=sys.stderr)
                return 1
        return cmd_dev(port)
    if arg == "pipeline":
        n = next((int(a) for a in sys.argv[2:] if a.isdigit()), 2000)
        return cmd_pipeline(n)
    return fn()


if __name__ == "__main__":
    sys.exit(main())

"""Utilitaires MLflow pour scripts et Streamlit."""
from __future__ import annotations


def log_system_metrics_manual() -> None:
    """Log manuel CPU/RAM/GPU pour runs courts (observabilité type Datadog)."""
    try:
        import mlflow
        import psutil
        p = psutil.Process()
        mem = p.memory_info()
        vm = psutil.virtual_memory()
        metrics = {
            "system/cpu_utilization_percentage": float(psutil.cpu_percent(interval=0.1)),
            "system/system_memory_usage_megabytes": round(mem.rss / (1024 * 1024), 2),
            "system/system_memory_usage_percentage": float(vm.percent),
        }
        try:
            import torch
            if torch.cuda.is_available():
                metrics["system/gpu_memory_usage_megabytes"] = round(
                    torch.cuda.memory_allocated(0) / (1024**2), 2
                )
        except Exception:
            pass
        mlflow.log_metrics(metrics)
    except ImportError:
        pass
    except Exception:
        pass

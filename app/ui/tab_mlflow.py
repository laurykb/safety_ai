"""MLflow tab - observability, tracking and run comparison."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from cyberbullying.config import MLFLOW_DIR, get_mlflow_tracking_uri

# Metriques ML loguees par run_pipeline
ML_METRIC_COLS = [
    "metrics.f1",
    "metrics.accuracy",
    "metrics.precision",
    "metrics.recall",
]
SYSTEM_COLS = [
    "metrics.system/cpu_utilization_percentage",
    "metrics.system/system_memory_usage_percentage",
    "metrics.system/system_memory_usage_megabytes",
    "metrics.system/gpu_utilization_percentage",
    "metrics.system/gpu_memory_usage_megabytes",
]


def _fmt_dur(x):
    if pd.isna(x):
        return "-"
    try:
        v = float(x)
        return f"{v:.1f}s" if v == v else "-"
    except (TypeError, ValueError):
        return "-"


def _fmt_metric(x):
    if pd.isna(x):
        return "-"
    try:
        v = float(x)
        return f"{v:.3f}" if 0 <= v <= 1 else f"{v:.1f}"
    except (TypeError, ValueError):
        return "-"


def render(tab):
    with tab:
        st.subheader("MLflow – Observabilite & Suivi des runs")

        st.info(
            "**MLflow** enregistre chaque entrainement (Operations), inference, fine-tuning et prediction. "
            "Metriques ML (F1, accuracy, precision, recall), parametres, duree et ressources systeme (CPU, RAM, GPU)."
        )
        st.caption(f"Tracking URI: {MLFLOW_DIR}")

        try:
            import mlflow
            import warnings
            with warnings.catch_warnings(action="ignore"):
                mlflow.set_tracking_uri(get_mlflow_tracking_uri())
            try:
                mlflow.enable_system_metrics_logging()
                mlflow.set_system_metrics_sampling_interval(2)
            except Exception:
                pass
            MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
            experiments = mlflow.search_experiments()
            if not experiments:
                st.info("Aucun run MLflow. Lancez Operations (train), Inference, Fine-tuning ou Prediction pour creer des runs.")
            else:
                exp_names = [e.name for e in experiments]
                exp_ids = [e.experiment_id for e in experiments]
                selected_exp = st.selectbox("Experience", exp_names, key="mlflow_exp")
                exp_id = exp_ids[exp_names.index(selected_exp)]

                col_ml1, col_ml2 = st.columns([3, 1])
                with col_ml2:
                    if st.button("Rafraichir", key="refresh_mlflow"):
                        st.rerun()

                runs_df = mlflow.search_runs(experiment_ids=[exp_id], order_by=["start_time DESC"])
                if runs_df.empty:
                    st.info("Aucun run dans cette experience.")
                else:
                    run_name_col = "tags.mlflow.runName" if "tags.mlflow.runName" in runs_df.columns else "run_id"

                    if "duration" not in runs_df.columns or runs_df["duration"].isna().all():
                        try:
                            if "end_time" in runs_df.columns and "start_time" in runs_df.columns:
                                runs_df["duration"] = (
                                    pd.to_datetime(runs_df["end_time"]) - pd.to_datetime(runs_df["start_time"])
                                ).dt.total_seconds()
                            else:
                                runs_df["duration"] = None
                        except Exception:
                            runs_df["duration"] = None

                    # Filtres
                    param_cols = [c for c in runs_df.columns if c.startswith("params.")]
                    if "params.embedding" in runs_df.columns:
                        emb_vals = runs_df["params.embedding"].dropna().unique().tolist()
                        if emb_vals:
                            emb_filter = st.multiselect("Filtrer par embedding", emb_vals, default=emb_vals, key="mlflow_emb")
                            runs_df = runs_df[runs_df["params.embedding"].isin(emb_filter) | runs_df["params.embedding"].isna()]
                    if "params.model" in runs_df.columns:
                        mod_vals = runs_df["params.model"].dropna().unique().tolist()
                        if mod_vals:
                            mod_filter = st.multiselect("Filtrer par modele", mod_vals, default=mod_vals, key="mlflow_mod")
                            runs_df = runs_df[runs_df["params.model"].isin(mod_filter) | runs_df["params.model"].isna()]

                    # Sous-onglets
                    tab_overview, tab_ml_metrics, tab_sys_metrics, tab_compare, tab_delete = st.tabs([
                        "Vue d'ensemble", "Metriques ML", "Metriques systeme", "Comparer runs", "Supprimer"
                    ])

                    with tab_overview:
                        base_cols = ["run_id", run_name_col, "start_time", "duration"]
                        ml_cols = [c for c in ML_METRIC_COLS if c in runs_df.columns]
                        sys_cols = [c for c in SYSTEM_COLS if c in runs_df.columns]
                        disp_cols = [c for c in base_cols if c in runs_df.columns]
                        disp_cols += ml_cols[:4]
                        disp_cols += [c for c in param_cols if c in runs_df.columns][:5]
                        disp_cols += sys_cols[:2]

                        display_df = runs_df[disp_cols].head(50).copy()
                        display_df.columns = [
                            c.replace("metrics.", "").replace("params.", "").replace("system/", "sys_")
                            for c in display_df.columns
                        ]
                        if "duration" in display_df.columns:
                            display_df["duration"] = display_df["duration"].apply(_fmt_dur)
                        for mc in ["f1", "accuracy", "precision", "recall"]:
                            if mc in display_df.columns:
                                display_df[mc] = display_df[mc].apply(_fmt_metric)
                        st.dataframe(display_df, use_container_width=True, hide_index=True)

                    with tab_ml_metrics:
                        ml_available = [c for c in ML_METRIC_COLS if c in runs_df.columns and runs_df[c].notna().any()]
                        if not ml_available:
                            st.info(
                                "Aucune metrique ML (F1, accuracy) dans les runs. "
                                "Lancez Operations > Train ou Pipeline complet pour obtenir des runs avec metriques."
                            )
                        else:
                            plot_df = runs_df.head(30).copy()
                            x_col = run_name_col
                            if x_col in plot_df.columns:
                                for mcol in ml_available:
                                    if plot_df[mcol].notna().any():
                                        short = mcol.replace("metrics.", "").replace("_", " ").title()
                                        fig = px.bar(
                                            plot_df.dropna(subset=[mcol]),
                                            x=x_col,
                                            y=mcol,
                                            title=short,
                                            labels={x_col: "Run", mcol: short},
                                        )
                                        fig.update_layout(xaxis_tickangle=-45, height=280)
                                        st.plotly_chart(fig, use_container_width=True)
                            st.write("**Comparaison F1 / Accuracy**")
                            if "metrics.f1" in runs_df.columns and "metrics.accuracy" in runs_df.columns:
                                cmp = runs_df.dropna(subset=["metrics.f1", "metrics.accuracy"]).head(20)
                                if not cmp.empty:
                                    fig2 = px.scatter(
                                        cmp,
                                        x="metrics.accuracy",
                                        y="metrics.f1",
                                        hover_data=[run_name_col] if run_name_col in cmp.columns else [],
                                        color="params.embedding" if "params.embedding" in cmp.columns else None,
                                        size=[10] * len(cmp),
                                        labels={"metrics.accuracy": "Accuracy", "metrics.f1": "F1-Score"},
                                    )
                                    fig2.update_layout(height=350)
                                    st.plotly_chart(fig2, use_container_width=True)

                    with tab_sys_metrics:
                        sys_available = [c for c in SYSTEM_COLS if c in runs_df.columns and runs_df[c].notna().any()]
                        if not sys_available:
                            sys_available = [
                                c for c in runs_df.columns
                                if c.startswith("metrics.") and any(k in c.lower() for k in ("cpu", "memory", "gpu", "system"))
                                and runs_df[c].notna().any()
                            ]
                        if sys_available:
                            plot_df = runs_df.head(20).copy()
                            x_col = run_name_col
                            if x_col in plot_df.columns:
                                for mcol in sys_available[:4]:
                                    if mcol in plot_df.columns and plot_df[mcol].notna().any():
                                        short = mcol.replace("metrics.system/", "").replace("_", " ").title()
                                        fig = px.bar(
                                            plot_df.dropna(subset=[mcol]),
                                            x=x_col,
                                            y=mcol,
                                            title=short,
                                            labels={x_col: "Run", mcol: short},
                                        )
                                        fig.update_layout(xaxis_tickangle=-45, height=250)
                                        st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(
                                "Metriques systeme non disponibles (runs anciens). "
                                "Lancez une nouvelle inference ou fine-tuning, puis Rafraichir."
                            )

                    with tab_compare:
                        runs_with_ml = runs_df.dropna(
                            subset=[c for c in ML_METRIC_COLS if c in runs_df.columns],
                            how="all"
                        ).head(15)
                        if runs_with_ml.empty or not any(c in runs_with_ml.columns for c in ML_METRIC_COLS):
                            st.info("Selectionnez des runs avec metriques ML pour comparer.")
                        else:
                            compare_cols = [c for c in ML_METRIC_COLS if c in runs_with_ml.columns]
                            compare_cols += [c for c in param_cols if c in runs_with_ml.columns][:4]
                            cmp_df = runs_with_ml[compare_cols].copy()
                            for c in compare_cols:
                                if c.startswith("metrics."):
                                    cmp_df[c] = cmp_df[c].apply(lambda x: f"{x:.3f}" if pd.notna(x) and isinstance(x, (int, float)) else "-")
                            st.dataframe(cmp_df, use_container_width=True, hide_index=True)
                            if "params.embedding" in runs_with_ml.columns and "params.model" in runs_with_ml.columns:
                                agg = runs_with_ml.groupby(["params.embedding", "params.model"])[
                                    [c for c in ML_METRIC_COLS if c in runs_with_ml.columns]
                                ].mean()
                                st.write("**Moyenne par (embedding, modele)**")
                                st.dataframe(agg.style.format("{:.3f}"), use_container_width=True, hide_index=True)

                    with tab_delete:
                        run_options = list(zip(
                            runs_df["run_id"],
                            runs_df[run_name_col].astype(str) if run_name_col in runs_df.columns else runs_df["run_id"].astype(str),
                        ))
                        runs_to_delete = st.multiselect(
                            "Runs a supprimer",
                            options=run_options,
                            format_func=lambda x: f"{x[1]} ({x[0][:8]}...)" if len(str(x[0])) > 8 else str(x[1]),
                            key="mlflow_runs_to_delete",
                        )
                        if runs_to_delete and st.button("Supprimer les runs selectionnes", type="secondary"):
                            client = mlflow.MlflowClient()
                            for run_id, _ in runs_to_delete:
                                try:
                                    client.delete_run(run_id)
                                except Exception:
                                    pass
                            st.success("Runs supprimes.")
                            st.rerun()
        except ImportError:
            st.warning("MLflow non installe: pip install mlflow")
        except Exception as e:
            st.error(f"Erreur MLflow: {e}")

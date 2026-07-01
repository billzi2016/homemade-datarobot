#!/usr/bin/env python3
"""
task 运行入口。

这份脚本是当前测试阶段的最小可运行实现，目标只有一个：
把单个 task 目录里的配置、数据、analysis、sklearn、MLflow 记录串起来，
先跑通一条完整主链路。

设计说明：
1. 当前阶段不实现 task_id 自动发号，直接使用已有的 task 名称。
2. MLflow 不再额外创建主 task run；每个分析项和模型自己直接成为 experiment 顶层 run。
3. run_state.json 以“人类可读”为第一优先级，因此会明确记录已完成实验列表。
4. 为了控制写盘量，模型类只保留必要最终产物；analysis 类完整保留 2D 结果。
5. 搜索策略默认走 auto，不允许把默认路径退化为 none。
6. 分类任务默认分层切分，并尽量对每个模型启用类别平衡。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import matplotlib.pyplot as plt
import pandas as pd

from analysis_runner import run_analysis
from sklearn_common import infer_feature_columns
from sklearn_runner import run_sklearn
from task_runtime import (
    TaskPaths,
    build_task_paths,
    configure_runtime_env,
    ensure_task_dirs,
    load_or_create_run_state,
    load_yaml,
    mark_step_completed,
    read_dataset,
    save_run_state,
)
from torch_runner import run_torch


def build_leaderboard_rows(
    summaries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    统一拼装排行榜明细。

    这里不再按 sklearn / torch 做 UI 层级，而是把所有模型直接拉平到一个排行榜表里。
    """

    rows: List[Dict[str, Any]] = []
    for summary in summaries:
        if not summary:
            continue
        source_name = str(summary.get("source_name", "model"))
        for row in summary.get("summary_rows", []):
            rows.append({"source_name": source_name, **row})
    return rows


def render_visualization_overview(analysis_summary: Dict[str, Any], output_path: Path) -> None:
    """
    生成 data_visualization 总览图。

    用户点击 data_visualization run 时，应该先看到一张总览图，
    而不是只能一个个点 artifact 文件名去猜哪个是 PCA、哪个是 UMAP。
    """

    rows = analysis_summary.get("rows", [])
    if not rows:
        return

    fig, axes = plt.subplots(1, len(rows), figsize=(6 * len(rows), 5))
    if len(rows) == 1:
        axes = [axes]

    for ax, row in zip(axes, rows):
        image = plt.imread(row["plot_path"])
        ax.imshow(image)
        ax.set_title(str(row["name"]).upper())
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def log_data_visualization_run(
    config: Dict[str, Any],
    paths: TaskPaths,
    analysis_summary: Dict[str, Any],
) -> None:
    """把 PCA / t-SNE / UMAP 汇总为单独顶层 run，避免污染模型排行榜列表。"""

    if not analysis_summary:
        return

    overview_path = paths.plots_dir / "data_visualization_overview.png"
    render_visualization_overview(analysis_summary, overview_path)

    with mlflow.start_run(run_name="data_visualization"):
        mlflow.set_tag("task_id", config["task"]["task_id"])
        mlflow.set_tag("run_level", "item")
        mlflow.set_tag("item_name", "data_visualization")
        mlflow.set_tag("item_kind", "visualization")
        mlflow.set_tag(
            "available_methods",
            ",".join(str(row.get("name")) for row in analysis_summary.get("rows", [])),
        )
        mlflow.log_dict(config, "config_snapshot.yaml")
        mlflow.log_param("sample_count", analysis_summary["sample_count"])
        mlflow.log_param("feature_count", analysis_summary["feature_count"])
        mlflow.log_param("analysis_input_feature_count", analysis_summary["analysis_input_feature_count"])
        for row in analysis_summary.get("rows", []):
            if row["name"] == "pca" and "explained_variance_sum" in row:
                mlflow.log_metric("pca_explained_variance_sum", float(row["explained_variance_sum"]))
            mlflow.log_artifact(row["result_path"], artifact_path="analysis")
            mlflow.log_artifact(row["plot_path"], artifact_path="plots")
        mlflow.log_artifact(str(analysis_summary["index_path"]), artifact_path="analysis")
        if overview_path.exists():
            mlflow.log_artifact(str(overview_path), artifact_path="plots")


def render_leaderboard_plot(leaderboard_df: pd.DataFrame, output_path: Path) -> None:
    """
    生成排行榜总览图。

    这里不再做“单个 best_accuracy 柱子”的弱展示，而是直接把三类核心指标
    和前十模型清单一起画出来，用户点进 leaderboard run 之后可以立刻读。
    """

    top_df = leaderboard_df.head(10).copy()
    top_df["label"] = top_df["model_name"].astype(str)
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    metric_axes = [axes[0][0], axes[0][1], axes[1][0]]
    metrics = [("accuracy", "Accuracy"), ("f1_macro", "F1 Macro"), ("auc", "AUC")]
    for ax, (column_name, title) in zip(metric_axes, metrics):
        if column_name in top_df.columns:
            sorted_df = top_df.sort_values(by=column_name, ascending=True)
            ax.barh(sorted_df["label"], sorted_df[column_name], color="#2563eb")
            ax.set_title(title)
            ax.grid(axis="x", alpha=0.2)
        else:
            ax.set_visible(False)

    table_ax = axes[1][1]
    table_ax.axis("off")
    summary_columns = [column for column in ["model_name", "accuracy", "f1_macro", "auc"] if column in top_df.columns]
    display_df = top_df[summary_columns].copy()
    if "accuracy" in display_df.columns:
        display_df["accuracy"] = display_df["accuracy"].map(lambda value: f"{value:.4f}")
    if "f1_macro" in display_df.columns:
        display_df["f1_macro"] = display_df["f1_macro"].map(lambda value: f"{value:.4f}")
    if "auc" in display_df.columns:
        display_df["auc"] = display_df["auc"].map(lambda value: f"{value:.4f}")
    table_ax.set_title("Top Models Snapshot")
    table = table_ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.05, 1.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_leaderboard_payload(leaderboard_df: pd.DataFrame) -> Dict[str, Any]:
    """构建人类可读的排行榜摘要，便于同时落地到 JSON artifact。"""

    payload: Dict[str, Any] = {
        "row_count": int(len(leaderboard_df)),
        "metric_columns": [column for column in ["accuracy", "f1_macro", "auc", "rmse", "r2"] if column in leaderboard_df.columns],
        "top_models": leaderboard_df.head(10).to_dict(orient="records"),
    }
    best_by_metric: Dict[str, Any] = {}
    for metric_name in ["accuracy", "f1_macro", "auc"]:
        if metric_name not in leaderboard_df.columns:
            continue
        metric_series = leaderboard_df[metric_name].dropna()
        if metric_series.empty:
            continue
        best_row = leaderboard_df.loc[metric_series.idxmax()].to_dict()
        best_by_metric[metric_name] = {
            "model_name": best_row.get("model_name"),
            "metric_value": float(best_row.get(metric_name)),
        }
    payload["best_by_metric"] = best_by_metric
    return payload


def log_leaderboard_run(
    config: Dict[str, Any],
    paths: TaskPaths,
    leaderboard_rows: List[Dict[str, Any]],
) -> None:
    """把综合排行榜作为单独顶层 run 写入 MLflow。"""

    if not leaderboard_rows:
        return

    leaderboard_df = pd.DataFrame(leaderboard_rows)
    preferred_metrics = ["accuracy", "f1_macro", "auc", "rmse", "r2"]
    sort_metric = next((name for name in preferred_metrics if name in leaderboard_df.columns), None)
    if sort_metric is not None:
        ascending = sort_metric == "rmse"
        leaderboard_df = leaderboard_df.sort_values(by=sort_metric, ascending=ascending)

    leaderboard_path = paths.metrics_dir / "leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_path, index=False)
    leaderboard_payload = build_leaderboard_payload(leaderboard_df)
    leaderboard_payload_path = paths.metrics_dir / "leaderboard_summary.json"
    leaderboard_payload_path.write_text(
        json.dumps(leaderboard_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    leaderboard_plot_path = paths.plots_dir / "leaderboard_top_models.png"
    render_leaderboard_plot(leaderboard_df, leaderboard_plot_path)

    with mlflow.start_run(run_name="leaderboard"):
        mlflow.set_tag("task_id", config["task"]["task_id"])
        mlflow.set_tag("run_level", "item")
        mlflow.set_tag("item_name", "leaderboard")
        mlflow.set_tag("item_kind", "leaderboard")
        mlflow.set_tag("metric_columns", ",".join(leaderboard_payload["metric_columns"]))
        mlflow.log_dict(config, "config_snapshot.yaml")
        mlflow.log_param("row_count", int(len(leaderboard_df)))
        if sort_metric is not None:
            mlflow.log_param("primary_sort_metric", sort_metric)
            best_row = leaderboard_df.iloc[0].to_dict()
            mlflow.set_tag("best_model_name", str(best_row.get("model_name")))
            mlflow.set_tag("best_source_name", str(best_row.get("source_name")))
            if sort_metric in best_row and pd.notna(best_row[sort_metric]):
                mlflow.log_metric(f"best_{sort_metric}", float(best_row[sort_metric]))
        for metric_name in ["accuracy", "f1_macro", "auc"]:
            if metric_name in leaderboard_df.columns:
                top_metric_series = leaderboard_df[metric_name].dropna()
                if not top_metric_series.empty:
                    mlflow.log_metric(f"best_{metric_name}", float(top_metric_series.max()))
                    best_metric_row = leaderboard_df.loc[top_metric_series.idxmax()]
                    mlflow.log_param(f"best_{metric_name}_model", str(best_metric_row["model_name"]))
        mlflow.log_artifact(str(leaderboard_path), artifact_path="leaderboard")
        mlflow.log_artifact(str(leaderboard_payload_path), artifact_path="leaderboard")
        mlflow.log_artifact(str(leaderboard_plot_path), artifact_path="leaderboard")


def run_task(task_dir: Path) -> None:
    """单个 task 的总执行入口。"""

    paths = build_task_paths(task_dir)
    ensure_task_dirs(paths)
    configure_runtime_env(paths)

    config = load_yaml(paths.config_path)
    task_id = config["task"]["task_id"]
    state = load_or_create_run_state(paths, task_id)

    if state["status"] in {"failed", "terminated", "paused"}:
        state["resume_count"] = int(state.get("resume_count", 0)) + 1

    state["status"] = "running"
    state["current_stage"] = "bootstrap"
    state["main_run_id"] = None
    save_run_state(paths, state)

    mlflow.set_tracking_uri(config.get("mlflow", {}).get("tracking_uri", str(paths.mlruns_dir)))
    mlflow.set_experiment(config.get("mlflow", {}).get("experiment_name", "homemade_datarobot"))

    try:
        df = read_dataset(paths, config)
        mark_step_completed(state, "data_loaded")
        save_run_state(paths, state)

        target_column = config["data"]["target_column"]
        numeric_columns, categorical_columns = infer_feature_columns(config, df)
        feature_columns = numeric_columns + categorical_columns
        feature_df = df[feature_columns].copy()
        target_series = df[target_column].copy()

        mark_step_completed(state, "schema_checked")
        save_run_state(paths, state)

        analysis_summary = run_analysis(config, target_series, feature_df, paths, state)
        log_data_visualization_run(config, paths, analysis_summary)
        sklearn_summary = run_sklearn(config, df, paths, state)
        if sklearn_summary:
            sklearn_summary["source_name"] = "sklearn"
        torch_summary = run_torch(config, df, paths, state)
        if torch_summary:
            torch_summary["source_name"] = "torch"
        leaderboard_rows = build_leaderboard_rows([sklearn_summary, torch_summary])
        log_leaderboard_run(config, paths, leaderboard_rows)

        state["status"] = "completed"
        state["current_stage"] = "completed"
        save_run_state(paths, state)
    except Exception:
        state["status"] = "failed"
        save_run_state(paths, state)
        raise


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="运行单个 task 的 MLflow 实验链路。")
    parser.add_argument("task_dir", type=Path, help="例如 storage/task_000001")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_task(args.task_dir.resolve())

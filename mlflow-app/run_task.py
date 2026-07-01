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
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import pandas as pd

from analysis_runner import run_analysis
from sklearn_runner import infer_feature_columns, run_sklearn
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


def log_leaderboard_run(
    config: Dict[str, Any],
    paths: TaskPaths,
    leaderboard_rows: List[Dict[str, Any]],
) -> None:
    """把综合排行榜作为单独顶层 run 写入 MLflow。"""

    if not leaderboard_rows:
        return

    leaderboard_df = pd.DataFrame(leaderboard_rows)
    preferred_metrics = ["accuracy", "f1_macro", "roc_auc", "roc_auc_ovr", "rmse", "r2"]
    sort_metric = next((name for name in preferred_metrics if name in leaderboard_df.columns), None)
    if sort_metric is not None:
        ascending = sort_metric == "rmse"
        leaderboard_df = leaderboard_df.sort_values(by=sort_metric, ascending=ascending)

    leaderboard_path = paths.metrics_dir / "leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_path, index=False)

    with mlflow.start_run(run_name="leaderboard"):
        mlflow.set_tag("task_id", config["task"]["task_id"])
        mlflow.set_tag("run_level", "item")
        mlflow.set_tag("item_name", "leaderboard")
        mlflow.set_tag("item_kind", "leaderboard")
        mlflow.log_dict(config, "config_snapshot.yaml")
        mlflow.log_param("row_count", int(len(leaderboard_df)))
        if sort_metric is not None:
            mlflow.log_param("primary_sort_metric", sort_metric)
            best_row = leaderboard_df.iloc[0].to_dict()
            mlflow.set_tag("best_model_name", str(best_row.get("model_name")))
            if sort_metric in best_row and pd.notna(best_row[sort_metric]):
                mlflow.log_metric(f"best_{sort_metric}", float(best_row[sort_metric]))
        mlflow.log_artifact(str(leaderboard_path), artifact_path="leaderboard")


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

        run_analysis(config, target_series, feature_df, paths, state)
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

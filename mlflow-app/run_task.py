#!/usr/bin/env python3
"""
task 运行入口。

这份脚本是当前测试阶段的最小可运行实现，目标只有一个：
把单个 task 目录里的配置、数据、analysis、sklearn、MLflow 记录串起来，
先跑通一条完整主链路。

设计说明：
1. 当前阶段不实现 task_id 自动发号，直接使用已有的 task_000001 / task_000002。
2. 一个 task 对应一个主 MLflow run；analysis / sklearn 各子项用 nested run 记录。
3. run_state.json 以“人类可读”为第一优先级，因此会明确记录已完成实验列表。
4. 为了控制写盘量，模型类只保留必要最终产物；analysis 类完整保留 2D 结果。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import mlflow
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


def log_task_level_summary(
    summary: Dict[str, Any],
    summary_prefix: str,
    task_id: str,
    run_state_path: Path,
) -> None:
    """
    把子模块结果汇总回主 run。

    目的很明确：
    - 主 run 页面不能再是空的
    - 用户点开 task 时，至少能直接看到最佳模型和关键指标
    """

    if not summary:
        return

    best_model_name = summary["best_model_name"]
    metric_name = summary["primary_metric_name"]
    metric_value = summary["primary_metric_value"]

    mlflow.log_param(f"{summary_prefix}_best_model", best_model_name)
    mlflow.log_metric(f"{summary_prefix}_{metric_name}", metric_value)

    summary_path = summary.get("summary_path")
    if summary_path:
        mlflow.log_artifact(str(summary_path), artifact_path="task_summary")

    if run_state_path.exists():
        mlflow.log_artifact(str(run_state_path), artifact_path="task_summary")


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
    save_run_state(paths, state)

    mlflow.set_tracking_uri(config.get("mlflow", {}).get("tracking_uri", str(paths.mlruns_dir)))
    mlflow.set_experiment(config.get("mlflow", {}).get("experiment_name", "homemade_datarobot"))

    existing_run_id = state.get("main_run_id")
    run_context = (
        mlflow.start_run(run_id=existing_run_id)
        if existing_run_id
        else mlflow.start_run(run_name=task_id)
    )

    with run_context as main_run:
        state["main_run_id"] = main_run.info.run_id
        save_run_state(paths, state)

        mlflow.set_tag("task_id", task_id)
        mlflow.set_tag("run_level", "task")
        mlflow.log_dict(config, "config_snapshot.yaml")

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
        torch_summary = run_torch(config, df, paths, state)

        state["status"] = "completed"
        state["current_stage"] = "completed"
        save_run_state(paths, state)
        log_task_level_summary(
            sklearn_summary,
            summary_prefix="sklearn",
            task_id=task_id,
            run_state_path=paths.run_state_path,
        )
        log_task_level_summary(
            torch_summary,
            summary_prefix="torch",
            task_id=task_id,
            run_state_path=paths.run_state_path,
        )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="运行单个 task 的 MLflow 实验链路。")
    parser.add_argument("task_dir", type=Path, help="例如 storage/task_000001")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_task(args.task_dir.resolve())

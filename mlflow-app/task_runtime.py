"""
task 运行期基础设施。

这个模块只负责与“单个 task 的运行上下文”相关的通用能力：
- task 目录路径组织
- 配置读取
- 运行时环境变量
- run_state.json 读写

这样拆开之后，analysis 和 sklearn 模块都不需要再重复处理这些基础问题。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml


@dataclass
class TaskPaths:
    """集中保存当前 task 会用到的路径，避免路径字符串散落在代码里。"""

    task_dir: Path
    config_path: Path
    raw_data_dir: Path
    run_state_path: Path
    checkpoints_dir: Path
    artifacts_dir: Path
    outputs_dir: Path
    predictions_dir: Path
    metrics_dir: Path
    models_dir: Path
    plots_dir: Path
    analysis_dir: Path
    mlruns_dir: Path
    mpl_config_dir: Path
    numba_cache_dir: Path
    tmp_dir: Path


def build_task_paths(task_dir: Path) -> TaskPaths:
    """按照 PRD 里约定的目录结构，构造当前 task 运行所需的全部路径。"""

    outputs_dir = task_dir / "outputs"
    shared_runtime_cache_dir = task_dir.parent / ".runtime_cache"
    return TaskPaths(
        task_dir=task_dir,
        config_path=task_dir / "config.yaml",
        raw_data_dir=task_dir / "data" / "raw",
        run_state_path=task_dir / "run_state.json",
        checkpoints_dir=task_dir / "checkpoints",
        artifacts_dir=task_dir / "artifacts",
        outputs_dir=outputs_dir,
        predictions_dir=outputs_dir / "predictions",
        metrics_dir=outputs_dir / "metrics",
        models_dir=outputs_dir / "models",
        plots_dir=outputs_dir / "plots",
        analysis_dir=outputs_dir / "analysis",
        mlruns_dir=task_dir / "mlruns",
        mpl_config_dir=task_dir / ".mplconfig",
        # numba/umap 缓存不再直接塞进 task 目录，避免 task 目录被大量编译缓存污染。
        # 这里仍然按 task_name 分子目录，原因是不同 task 的运行环境与依赖版本可能不同，
        # 完全共用一个平铺缓存目录更容易出现缓存相互覆盖、排查困难的问题。
        numba_cache_dir=shared_runtime_cache_dir / "numba" / task_dir.name,
        tmp_dir=task_dir / ".tmp",
    )


def ensure_task_dirs(paths: TaskPaths) -> None:
    """确保 task 目录下的核心输出路径存在。"""

    for path in [
        paths.checkpoints_dir,
        paths.artifacts_dir,
        paths.raw_data_dir,
        paths.predictions_dir,
        paths.metrics_dir,
        paths.models_dir,
        paths.plots_dir,
        paths.analysis_dir,
        paths.mlruns_dir,
        paths.mpl_config_dir,
        paths.numba_cache_dir,
        paths.tmp_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def configure_runtime_env(paths: TaskPaths) -> None:
    """
    配置运行时环境变量。

    这里主要解决两个现实问题：
    1. matplotlib 在当前机器上默认缓存目录不可写。
    2. umap/numba 在当前机器上默认缓存目录也可能不可写。
    """

    os.environ["MPLCONFIGDIR"] = str(paths.mpl_config_dir)
    os.environ["NUMBA_CACHE_DIR"] = str(paths.numba_cache_dir)
    os.environ["TMPDIR"] = str(paths.tmp_dir)


def load_yaml(path: Path) -> Dict[str, Any]:
    """读取单个 task 的配置文件。"""

    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_input_path(paths: TaskPaths, config: Dict[str, Any]) -> Path:
    """
    解析数据文件路径。

    设计约束：
    1. 测试阶段也要尽量和最终目录结构一致。
    2. 因此优先支持 task 目录内的相对路径，例如 `data/raw/Iris.csv`。
    3. 如果用户配置的是绝对路径，也允许直接使用。
    """

    raw_value = config["data"]["input_path"]
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return paths.task_dir / candidate


def read_dataset(paths: TaskPaths, config: Dict[str, Any]):
    """按配置读取 CSV 数据。当前首版先只支持 CSV。"""

    input_path = resolve_input_path(paths, config)
    if input_path.suffix.lower() != ".csv":
        raise ValueError("当前首版实现只支持 CSV 输入。")
    return pd.read_csv(input_path)


def init_run_state(task_id: str) -> Dict[str, Any]:
    """初始化 run_state.json 的内存结构。"""

    return {
        "task_id": task_id,
        "status": "created",
        "current_stage": "created",
        "last_completed_experiment": None,
        "completed_experiments": {
            "analysis": [],
            "sklearn": [],
            "torch": [],
        },
        "completed_steps": [],
        "pending_steps": [],
        "resume_supported": True,
        "resume_count": 0,
        "updated_at": pd.Timestamp.utcnow().isoformat(),
        "main_run_id": None,
    }


def load_or_create_run_state(paths: TaskPaths, task_id: str) -> Dict[str, Any]:
    """如果已有 run_state.json 就读取，否则创建新的状态结构。"""

    if paths.run_state_path.exists():
        with paths.run_state_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return init_run_state(task_id)


def save_run_state(paths: TaskPaths, state: Dict[str, Any]) -> None:
    """把当前 task 的状态落到 run_state.json。"""

    state["updated_at"] = pd.Timestamp.utcnow().isoformat()
    with paths.run_state_path.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)


def mark_step_completed(state: Dict[str, Any], step_name: str) -> None:
    """记录一个已完成步骤，避免重复写入。"""

    if step_name not in state["completed_steps"]:
        state["completed_steps"].append(step_name)


def mark_experiment_completed(
    state: Dict[str, Any], domain: str, experiment_name: str
) -> None:
    """
    记录一个已完成实验项。

    这里同时更新：
    - last_completed_experiment
    - completed_experiments 分组列表
    - completed_steps
    """

    if experiment_name not in state["completed_experiments"][domain]:
        state["completed_experiments"][domain].append(experiment_name)
    state["last_completed_experiment"] = f"{domain}.{experiment_name}"
    mark_step_completed(state, f"{domain}.{experiment_name}.completed")


def is_experiment_completed(
    state: Dict[str, Any], domain: str, experiment_name: str
) -> bool:
    """判断某个实验是否已在历史运行中完成，用于续跑跳过。"""

    return experiment_name in state["completed_experiments"].get(domain, [])

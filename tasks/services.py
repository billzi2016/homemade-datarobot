"""
tasks 应用的服务层。

这里故意把“目录扫描 / 配置写入 / 子进程启动 / 下载枚举”从 view 里拆出来，
避免 Django 视图文件再次长成一个几百行的筐。
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml
from django.conf import settings
from django.utils import timezone


SKLEARN_CLASSIFICATION_MODELS = [
    "logistic_regression",
    "svm",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "decision_tree",
    "knn",
    "gaussian_nb",
    "bernoulli_nb",
    "multinomial_nb",
    "xgboost",
    "lightgbm",
]

SKLEARN_REGRESSION_MODELS = [
    "linear_regression",
    "ridge",
    "lasso",
    "elasticnet",
    "svr",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "decision_tree",
    "knn",
    "xgboost",
    "lightgbm",
]

TORCH_CLASSIFICATION_MODELS = ["mlp", "cnn1d", "tabnet"]
TORCH_REGRESSION_MODELS: List[str] = []


@dataclass
class TaskSummary:
    """任务列表与详情页共用的轻量任务摘要。"""

    task_id: str
    task_dir: Path
    config: Dict[str, Any]
    state: Dict[str, Any]


def storage_base_root() -> Path:
    """
    返回用户目录的父目录。

    兼容当前 .env 里的历史写法：
    TASK_STORAGE_ROOT=storage/user_bizi

    如果配置值本身是 user_xxx，就取它的 parent 作为用户空间根目录；
    否则把配置值当作用户空间根目录。
    """

    configured_root = Path(settings.TASK_STORAGE_ROOT)
    if configured_root.name.startswith("user_"):
        return configured_root.parent
    return configured_root


def safe_username_slug(username: str) -> str:
    """把用户名转成可用于目录名的安全片段。"""

    lowered = username.strip().lower()
    slug = re.sub(r"[^a-z0-9_]+", "_", lowered)
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        raise ValueError("用户名无法转换为有效存储目录。")
    return slug


def storage_root(user=None) -> Path:
    """
    当前登录用户的 storage 根目录。

    - user.username == bizi  -> storage/user_bizi
    - user.username == admin -> storage/user_admin

    如果没有传 user，保留历史 fallback，便于少量非请求上下文代码继续工作。
    """

    if user is None:
        return Path(settings.TASK_STORAGE_ROOT)
    return storage_base_root() / f"user_{safe_username_slug(user.username)}"


def parse_csv_text(raw_value: str) -> List[str]:
    """把逗号分隔字符串转成列名列表。"""

    if not raw_value.strip():
        return []
    return [item.strip() for item in raw_value.replace("\n", ",").split(",") if item.strip()]


def task_directories(user=None) -> List[Path]:
    """扫描当前用户空间下的全部任务目录。"""

    root = storage_root(user)
    if not root.exists():
        return []
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("task_")],
        key=lambda item: item.name,
    )


def load_yaml_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_task_summary(task_dir: Path) -> TaskSummary:
    """加载单个任务的配置和状态。"""

    config = load_yaml_file(task_dir / "config.yaml")
    state = load_json_file(task_dir / "run_state.json")
    return TaskSummary(task_id=task_dir.name, task_dir=task_dir, config=config, state=state)


def list_task_summaries(user=None) -> List[TaskSummary]:
    """加载全部任务，供列表页展示。"""

    return [load_task_summary(task_dir) for task_dir in task_directories(user)]


def ensure_task_layout(task_dir: Path) -> None:
    """
    确保任务目录骨架存在。

    这里故意和 mlflow-app 里的目录约定保持一致，避免 Django 和训练子系统各写一套路径。
    """

    for path in [
        task_dir / "data" / "raw",
        task_dir / "outputs" / "predictions",
        task_dir / "outputs" / "metrics",
        task_dir / "outputs" / "models",
        task_dir / "outputs" / "plots",
        task_dir / "outputs" / "analysis",
        task_dir / "checkpoints",
        task_dir / "artifacts",
        task_dir / ".django_runtime",
        task_dir / ".tmp",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def default_task_config(cleaned_data: Dict[str, Any], uploaded_filename: str, user_root: Path) -> Dict[str, Any]:
    """
    根据表单生成首版 config.yaml。

    这里刻意生成一份完整配置，而不是依赖运行时再去拼接半成品，
    因为“任务目录自带完整配置”是当前项目的重要约束。
    """

    task_type = cleaned_data["task_type"]
    task_id = cleaned_data["task_name"]
    sklearn_classification_models = SKLEARN_CLASSIFICATION_MODELS if cleaned_data["enable_sklearn"] else []
    sklearn_regression_models = SKLEARN_REGRESSION_MODELS if cleaned_data["enable_sklearn"] else []
    torch_classification_models = TORCH_CLASSIFICATION_MODELS if cleaned_data["enable_torch"] else []
    torch_regression_models = TORCH_REGRESSION_MODELS if cleaned_data["enable_torch"] else []

    return {
        "task": {
            "task_id": task_id,
        },
        "project": {
            "random_seed": cleaned_data["random_seed"],
        },
        "data": {
            "input_path": f"data/raw/{uploaded_filename}",
            "target_column": cleaned_data["target_column"],
            "task_type": task_type,
            "test_size": cleaned_data["test_size"],
            "stratify": task_type == "classification",
            "drop_columns": [],
        },
        "features": {
            "numeric_columns": parse_csv_text(cleaned_data.get("numeric_columns", "")),
            "categorical_columns": parse_csv_text(cleaned_data.get("categorical_columns", "")),
            "onehot_columns": parse_csv_text(cleaned_data.get("onehot_columns", "")),
        },
        "preprocess": {
            "numeric_imputer": "median",
            "categorical_imputer": "most_frequent",
            "scaler": "standard",
        },
        "analysis": {
            "enabled": cleaned_data["enable_analysis"],
            "pca": {"enabled": True},
            "tsne": {"enabled": True, "perplexity": 20},
            "umap": {"enabled": True, "n_neighbors": 10, "min_dist": 0.1},
        },
        "imbalance": {
            "enabled": cleaned_data["imbalance_enabled"],
        },
        "models": {
            "sklearn": {
                "classification": sklearn_classification_models if task_type == "classification" else [],
                "regression": sklearn_regression_models if task_type == "regression" else [],
            },
            "torch": {
                "classification": torch_classification_models if task_type == "classification" else [],
                "regression": torch_regression_models if task_type == "regression" else [],
            },
        },
        "search": {
            "method": "auto",
            "n_trials": 4,
        },
        "cv": {
            "n_splits": cleaned_data["cv_n_splits"],
        },
        "training": {
            "torch": {
                "epochs": 30,
                "batch_size": 64,
                "learning_rate": 0.001,
                "dropout": 0.2,
                "hidden_dims": [128, 64],
                "early_stopping": {
                    "patience": 5,
                },
            }
        },
        "metrics": {
            "primary_metric": cleaned_data["primary_metric"],
        },
        "mlflow": {
            "tracking_uri": f"file://{user_root / 'mlruns'}",
            "experiment_name": task_id,
        },
    }


def write_task_config(task_dir: Path, payload: Dict[str, Any]) -> None:
    """把配置写到任务目录。"""

    (task_dir / "config.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def save_uploaded_dataset(task_dir: Path, uploaded_file) -> str:
    """
    保存上传文件。

    当前阶段不做对象存储，直接落本地磁盘。
    """

    file_name = Path(uploaded_file.name).name
    target_path = task_dir / "data" / "raw" / file_name
    with target_path.open("wb") as fh:
        for chunk in uploaded_file.chunks():
            fh.write(chunk)
    return file_name


def create_task_from_form(cleaned_data: Dict[str, Any], uploaded_file, user=None) -> Path:
    """从前端表单创建一个完整的任务目录。"""

    user_root = storage_root(user)
    task_dir = user_root / cleaned_data["task_name"]
    if task_dir.exists():
        raise ValueError(f"任务目录已存在：{task_dir.name}")

    ensure_task_layout(task_dir)
    uploaded_filename = save_uploaded_dataset(task_dir, uploaded_file)
    config = default_task_config(cleaned_data, uploaded_filename, user_root)
    write_task_config(task_dir, config)
    return task_dir


def runtime_dir(task_dir: Path) -> Path:
    return task_dir / ".django_runtime"


def process_meta_path(task_dir: Path) -> Path:
    return runtime_dir(task_dir) / "process_meta.json"


def task_log_path(task_dir: Path) -> Path:
    return runtime_dir(task_dir) / "train.log"


def load_process_meta(task_dir: Path) -> Dict[str, Any]:
    return load_json_file(process_meta_path(task_dir))


def save_process_meta(task_dir: Path, payload: Dict[str, Any]) -> None:
    process_meta_path(task_dir).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_run_state(task_dir: Path) -> Dict[str, Any]:
    return load_json_file(task_dir / "run_state.json")


def update_run_state_for_launch(task_dir: Path, pid: int) -> None:
    """
    由 Django 在启动子进程前后补写状态。

    mlflow-app 会继续维护 run_state.json，但 Django 至少要在任务刚启动时把 pid 写进去，
    否则前端详情页和 SSE 没法准确知道“是谁在跑”。
    """

    state = read_run_state(task_dir)
    if not state:
        state = {
            "task_id": task_dir.name,
            "status": "pending",
            "current_stage": "pending",
            "completed_experiments": {"analysis": [], "sklearn": [], "torch": []},
            "completed_steps": [],
            "resume_supported": True,
            "resume_count": 0,
        }
    state["pid"] = pid
    state["status"] = "pending"
    state["current_stage"] = state.get("current_stage") or "pending"
    state["updated_at"] = timezone.now().isoformat()
    (task_dir / "run_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def start_task_process(task_dir: Path) -> Dict[str, Any]:
    """
    启动训练子进程。

    当前阶段刻意使用 subprocess，而不是 Celery。
    这是你之前已经确认的架构约束。
    """

    ensure_task_layout(task_dir)
    log_file = task_log_path(task_dir)
    log_handle = log_file.open("ab")
    command = [sys.executable, str(settings.MLFLOW_RUN_TASK_SCRIPT), str(task_dir)]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    process = subprocess.Popen(
        command,
        cwd=settings.BASE_DIR,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
    )
    meta = {
        "pid": process.pid,
        "command": command,
        "started_at": timezone.now().isoformat(),
        "log_path": str(log_file),
    }
    save_process_meta(task_dir, meta)
    update_run_state_for_launch(task_dir, pid=process.pid)
    return meta


def process_is_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def build_task_status_payload(task_dir: Path) -> Dict[str, Any]:
    """
    组装详情页和 SSE 共用状态。

    这里故意做一个“宽一点”的结构，因为后续前端展示、
    API 输出、SSE 推送都会复用这份载荷，别让字段散在 view 里。
    """

    config = load_yaml_file(task_dir / "config.yaml")
    state = read_run_state(task_dir)
    process_meta = load_process_meta(task_dir)
    pid = state.get("pid") or process_meta.get("pid")
    outputs_dir = task_dir / "outputs"
    finished_files = list(iter_downloadable_files(task_dir))
    progress = estimate_progress(config=config, state=state)
    return {
        "task_id": task_dir.name,
        "task_type": config.get("data", {}).get("task_type"),
        "target_column": config.get("data", {}).get("target_column"),
        "status": state.get("status", "created"),
        "current_stage": state.get("current_stage", "created"),
        "last_completed_experiment": state.get("last_completed_experiment"),
        "completed_steps": state.get("completed_steps", []),
        "completed_experiments": state.get("completed_experiments", {}),
        "pid": pid,
        "is_process_running": process_is_running(pid),
        "updated_at": state.get("updated_at"),
        "progress_percent": progress["progress_percent"],
        "progress_label": progress["progress_label"],
        "completed_unit_count": progress["completed_unit_count"],
        "total_unit_count": progress["total_unit_count"],
        "mlflow_url": "/mlflow/open/",
        "task_dir": str(task_dir),
        "config_path": str(task_dir / "config.yaml"),
        "outputs_dir": str(outputs_dir),
        "downloadable_count": len(finished_files),
        "log_tail": read_log_tail(task_dir),
    }


def estimate_progress(config: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    估算 task 综合进度。

    这里不追求“精确到某个 epoch 的百分之几”，而是做任务级进度。
    对当前这套系统来说，最稳妥的分母是：
    - 两个固定基础步骤：data_loaded / schema_checked
    - analysis 启用项数量
    - sklearn 启用模型数量
    - torch 启用模型数量
    """

    completed_steps = set(state.get("completed_steps", []))
    completed_experiments = state.get("completed_experiments", {})

    total_unit_count = 2
    completed_unit_count = 0
    if "data_loaded" in completed_steps:
        completed_unit_count += 1
    if "schema_checked" in completed_steps:
        completed_unit_count += 1

    analysis_cfg = config.get("analysis", {})
    if analysis_cfg.get("enabled", True):
        for analysis_name in ["pca", "tsne", "umap"]:
            if analysis_cfg.get(analysis_name, {}).get("enabled", True):
                total_unit_count += 1
                if analysis_name in completed_experiments.get("analysis", []):
                    completed_unit_count += 1

    task_type = config.get("data", {}).get("task_type", "classification")
    sklearn_models = config.get("models", {}).get("sklearn", {}).get(task_type, [])
    torch_models = config.get("models", {}).get("torch", {}).get(task_type, [])

    total_unit_count += len(sklearn_models) + len(torch_models)
    completed_unit_count += len(completed_experiments.get("sklearn", []))
    completed_unit_count += len(completed_experiments.get("torch", []))

    status = state.get("status")
    if status in {"completed", "success"}:
        progress_percent = 100
    elif total_unit_count <= 0:
        progress_percent = 0
    else:
        progress_percent = int(min(100, round(completed_unit_count / total_unit_count * 100)))

    return {
        "progress_percent": progress_percent,
        "progress_label": f"{completed_unit_count} / {total_unit_count}",
        "completed_unit_count": completed_unit_count,
        "total_unit_count": total_unit_count,
    }


def read_log_tail(task_dir: Path, max_lines: int = 40) -> List[str]:
    """读取最近日志，给详情页做轻量展示。"""

    path = task_log_path(task_dir)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return lines[-max_lines:]


def iter_downloadable_files(task_dir: Path) -> Iterable[Path]:
    """
    枚举可下载文件。

    当前阶段不做“下载权限模型”，但会把能下载的东西集中到一个列表里。
    """

    preferred_roots = [
        task_dir / "config.yaml",
        task_dir / "run_state.json",
        task_dir / "outputs",
    ]
    for root in preferred_roots:
        if root.is_file():
            yield root
            continue
        if root.exists():
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    yield path


def download_records(task_dir: Path) -> List[Dict[str, Any]]:
    """把可下载文件列表转成模板更好渲染的结构。"""

    records: List[Dict[str, Any]] = []
    for path in iter_downloadable_files(task_dir):
        mime_type, _ = mimetypes.guess_type(path.name)
        records.append(
            {
                "name": path.name,
                "relative_path": str(path.relative_to(task_dir)),
                "size_bytes": path.stat().st_size,
                "mime_type": mime_type or "application/octet-stream",
            }
        )
    return records


def safe_download_path(task_dir: Path, relative_path: str) -> Path:
    """确保下载路径不会逃逸出 task 目录。"""

    candidate = (task_dir / relative_path).resolve()
    task_root = task_dir.resolve()
    if task_root not in candidate.parents and candidate != task_root:
        raise ValueError("非法下载路径。")
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(relative_path)
    return candidate


def delete_task_runtime_cache(task_dir: Path) -> None:
    """必要时可清理运行期缓存；当前先保留为内部工具函数。"""

    cache_dir = task_dir / ".django_runtime"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

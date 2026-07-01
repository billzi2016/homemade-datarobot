"""
django-ninja API 入口。

这里提供真实的 OpenAPI / Swagger UI，而不是手写一个“看起来像 API 文档”的页面。
当前 API 先覆盖 Django 页面已经具备的核心能力：
- 任务列表
- 单任务状态
- 启动任务
- 下载文件列表
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ninja import NinjaAPI, Schema
from django.views.decorators.csrf import csrf_protect

from tasks.services import (
    build_task_status_payload,
    download_records,
    list_task_summaries,
    process_is_running,
    read_run_state,
    start_task_process,
    storage_root,
)


api = NinjaAPI(
    title="homemade-datarobot API",
    version="0.1.0",
    description="Local Django API for task management, subprocess execution, MLflow tracking entrypoints, and downloadable artifacts.",
)


class TaskListItem(Schema):
    """任务列表响应。"""

    task_id: str
    task_type: str
    target_column: str
    status: str
    current_stage: str
    updated_at: Optional[str] = None


class TaskStatus(Schema):
    """任务详情状态响应。"""

    task_id: str
    task_type: Optional[str] = None
    target_column: Optional[str] = None
    status: str
    current_stage: str
    last_completed_experiment: Optional[str] = None
    completed_steps: List[str]
    completed_experiments: Dict[str, List[str]]
    pid: Optional[int] = None
    is_process_running: bool
    updated_at: Optional[str] = None
    progress_percent: int
    progress_label: str
    completed_unit_count: int
    total_unit_count: int
    mlflow_url: str
    task_dir: str
    config_path: str
    outputs_dir: str
    downloadable_count: int
    log_tail: List[str]


class RunTaskResponse(Schema):
    """启动任务响应。"""

    task_id: str
    started: bool
    pid: Optional[int] = None
    message: str


class DownloadRecord(Schema):
    """下载文件记录。"""

    name: str
    relative_path: str
    size_bytes: int
    mime_type: str


def resolve_task_dir_or_404(task_id: str):
    task_dir = storage_root() / task_id
    if not task_dir.exists():
        # django-ninja 会把这个响应直接序列化给调用方。
        raise FileNotFoundError(f"任务不存在：{task_id}")
    return task_dir


@api.get("/tasks", response=List[TaskListItem], tags=["tasks"])
def list_tasks(request) -> List[Dict[str, Any]]:
    """列出当前用户空间里的全部 task。"""

    rows: List[Dict[str, Any]] = []
    for summary in list_task_summaries():
        state = summary.state
        rows.append(
            {
                "task_id": summary.task_id,
                "task_type": summary.config.get("data", {}).get("task_type", "-"),
                "target_column": summary.config.get("data", {}).get("target_column", "-"),
                "status": state.get("status", "created"),
                "current_stage": state.get("current_stage", "created"),
                "updated_at": state.get("updated_at"),
            }
        )
    return rows


@api.get("/tasks/{task_id}", response=TaskStatus, tags=["tasks"])
def get_task_status(request, task_id: str) -> Dict[str, Any]:
    """读取单个 task 的状态、进度、日志尾部和 MLflow 入口。"""

    task_dir = resolve_task_dir_or_404(task_id)
    return build_task_status_payload(task_dir)


@api.post("/tasks/{task_id}/run", response=RunTaskResponse, tags=["tasks"])
@csrf_protect
def run_task(request, task_id: str) -> Dict[str, Any]:
    """启动单个 task 的本地训练子进程。"""

    task_dir = resolve_task_dir_or_404(task_id)
    state = read_run_state(task_dir)
    pid = state.get("pid")
    if process_is_running(pid):
        return {
            "task_id": task_id,
            "started": False,
            "pid": pid,
            "message": f"任务已有运行中的进程：PID={pid}",
        }
    meta = start_task_process(task_dir)
    return {
        "task_id": task_id,
        "started": True,
        "pid": int(meta["pid"]),
        "message": "训练子进程已启动。",
    }


@api.get("/tasks/{task_id}/downloads", response=List[DownloadRecord], tags=["tasks"])
def list_downloads(request, task_id: str) -> List[Dict[str, Any]]:
    """列出单个 task 目录下可下载的配置、状态、指标、图表和模型文件。"""

    task_dir = resolve_task_dir_or_404(task_id)
    return download_records(task_dir)

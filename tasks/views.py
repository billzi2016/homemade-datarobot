"""
tasks 应用视图。

第一版目标不是做成花哨前端，而是先把：
- 创建任务
- 启动任务
- 看状态
- SSE 轮询流
- 下载产物
这些关键业务路径串起来。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from django.contrib import messages
from django.http import FileResponse, Http404, HttpRequest, HttpResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_GET, require_POST

from tasks.forms import TaskCreateForm
from tasks.services import (
    build_task_status_payload,
    create_task_from_form,
    download_records,
    list_task_summaries,
    load_task_summary,
    process_is_running,
    read_run_state,
    safe_download_path,
    start_task_process,
    storage_root,
)


def resolve_task_dir(task_id: str) -> Path:
    task_dir = storage_root() / task_id
    if not task_dir.exists():
        raise Http404(task_id)
    return task_dir


@require_GET
def task_list_view(request: HttpRequest) -> HttpResponse:
    """任务列表页。"""

    task_cards = []
    for summary in list_task_summaries():
        state = summary.state
        task_cards.append(
            {
                "task_id": summary.task_id,
                "task_type": summary.config.get("data", {}).get("task_type", "-"),
                "target_column": summary.config.get("data", {}).get("target_column", "-"),
                "status": state.get("status", "created"),
                "current_stage": state.get("current_stage", "created"),
                "updated_at": state.get("updated_at", "-"),
            }
        )
    return render(request, "tasks/task_list.html", {"task_cards": task_cards})


def task_create_view(request: HttpRequest) -> HttpResponse:
    """任务创建页。"""

    if request.method == "POST":
        form = TaskCreateForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                task_dir = create_task_from_form(form.cleaned_data, request.FILES["data_file"])
            except Exception as exc:
                form.add_error(None, str(exc))
            else:
                messages.success(request, f"任务 {task_dir.name} 创建完成。")
                return redirect("task-detail", task_id=task_dir.name)
    else:
        form = TaskCreateForm()
    return render(request, "tasks/task_form.html", {"form": form})


@require_GET
def task_detail_view(request: HttpRequest, task_id: str) -> HttpResponse:
    """任务详情页。"""

    task_dir = resolve_task_dir(task_id)
    summary = load_task_summary(task_dir)
    status_payload = build_task_status_payload(task_dir)
    context = {
        "task": summary,
        "status_payload": status_payload,
        "download_url": reverse("task-download-list", kwargs={"task_id": task_id}),
        "run_url": reverse("task-run", kwargs={"task_id": task_id}),
        "events_url": reverse("task-events", kwargs={"task_id": task_id}),
    }
    return render(request, "tasks/task_detail.html", context)


@require_POST
def task_run_view(request: HttpRequest, task_id: str) -> HttpResponse:
    """启动单个任务。"""

    task_dir = resolve_task_dir(task_id)
    state = read_run_state(task_dir)
    pid = state.get("pid")
    if process_is_running(pid):
        messages.warning(request, f"{task_id} 当前已有运行中的进程（PID={pid}）。")
        return redirect("task-detail", task_id=task_id)

    start_task_process(task_dir)
    messages.success(request, f"{task_id} 已启动训练子进程。")
    return redirect("task-detail", task_id=task_id)


@require_GET
def task_events_view(request: HttpRequest, task_id: str) -> StreamingHttpResponse:
    """SSE 状态流。"""

    task_dir = resolve_task_dir(task_id)

    def event_stream():
        last_payload = None
        while True:
            payload = build_task_status_payload(task_dir)
            serialized = json.dumps(payload, ensure_ascii=False)
            if serialized != last_payload:
                yield f"data: {serialized}\n\n"
                last_payload = serialized
            if payload["status"] in {"completed", "failed", "cancelled", "success"} and not payload["is_process_running"]:
                break
            time.sleep(2)

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    return response


@require_GET
def task_download_list_view(request: HttpRequest, task_id: str) -> HttpResponse:
    """下载列表页。"""

    task_dir = resolve_task_dir(task_id)
    records = download_records(task_dir)
    return render(request, "tasks/download_list.html", {"task_id": task_id, "records": records})


@require_GET
def task_file_download_view(request: HttpRequest, task_id: str) -> FileResponse:
    """下载单个文件。"""

    task_dir = resolve_task_dir(task_id)
    relative_path = request.GET.get("path", "")
    try:
        target_path = safe_download_path(task_dir, relative_path)
    except FileNotFoundError as exc:
        raise Http404(str(exc)) from exc
    return FileResponse(target_path.open("rb"), as_attachment=True, filename=target_path.name)

"""
MLflow UI 本地进程管理。

当前阶段不是把 MLflow 做成公网多租户服务，而是在 Django 登录边界后，
为每个本地用户启动一个只绑定 127.0.0.1 的 MLflow UI 进程。

设计约束：
- 一个 Django 用户对应一个 storage/user_xxx/mlruns；
- 一个用户最多复用一个 MLflow UI 进程；
- 端口从配置的端口池里选择空闲值，避免多人同时打开时冲突；
- 进程元信息写入用户自己的 .django_runtime/mlflow_ui.json，便于人类排查。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from django.conf import settings
from django.utils import timezone

from tasks.services import load_json_file, storage_root


@dataclass
class MlflowUiInstance:
    """Django 返回给页面和 API 的 MLflow UI 实例信息。"""

    username: str
    pid: int
    port: int
    url: str
    backend_store_uri: str
    meta_path: Path


def user_runtime_dir(user) -> Path:
    """返回用户级运行时目录。"""

    path = storage_root(user) / ".django_runtime"
    path.mkdir(parents=True, exist_ok=True)
    return path


def mlflow_meta_path(user) -> Path:
    """返回用户 MLflow UI 进程元信息文件路径。"""

    return user_runtime_dir(user) / "mlflow_ui.json"


def mlruns_dir(user) -> Path:
    """返回当前用户的 MLflow backend store 目录。"""

    path = storage_root(user) / "mlruns"
    path.mkdir(parents=True, exist_ok=True)
    return path


def process_is_running(pid: int | None) -> bool:
    """判断本机进程是否还存在。"""

    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def port_is_open(host: str, port: int) -> bool:
    """判断端口是否已有进程监听。"""

    # 本地开发和 Codex 沙箱里，直接 socket connect 可能因为权限限制误判；
    # lsof 查询 LISTEN 状态更贴近我们真正关心的“端口是否已被占用”。
    return bool(listener_pids(port))


def listener_pids(port: int) -> list[int]:
    """
    查询当前监听指定端口的进程 PID。

    MLflow UI 在本机使用 Uvicorn 时会派生监听进程；Popen 拿到的启动器 PID
    可能很快退出，所以这里以实际 LISTEN 端口的进程为准。
    """

    command = ["lsof", "-nP", f"-tiTCP:{port}", "-sTCP:LISTEN"]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError:
        return []
    if result.returncode != 0:
        return []
    pids: list[int] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if line.isdigit():
            pids.append(int(line))
    return sorted(set(pids))


def find_free_port() -> int:
    """
    从配置的端口池里找一个空闲端口。

    这里不用随机端口，便于本地排查，也避免端口散得太开不好管理。
    """

    for port in range(settings.MLFLOW_UI_PORT_START, settings.MLFLOW_UI_PORT_END + 1):
        if not port_is_open(settings.MLFLOW_UI_HOST, port):
            return port
    raise RuntimeError(
        f"MLflow UI 端口池已用尽：{settings.MLFLOW_UI_PORT_START}-{settings.MLFLOW_UI_PORT_END}"
    )


def load_mlflow_meta(user) -> Dict[str, Any]:
    """读取用户级 MLflow UI 元信息。"""

    return load_json_file(mlflow_meta_path(user))


def save_mlflow_meta(user, payload: Dict[str, Any]) -> None:
    """写入用户级 MLflow UI 元信息。"""

    mlflow_meta_path(user).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def refresh_listener_pid(user, meta: Dict[str, Any]) -> Dict[str, Any]:
    """如果端口已经监听，用实际监听 PID 更新元信息。"""

    port = meta.get("port")
    if not port:
        return meta
    pids = listener_pids(int(port))
    if not pids:
        return meta
    refreshed = dict(meta)
    refreshed["pid"] = pids[0]
    refreshed["listener_pids"] = pids
    refreshed["refreshed_at"] = timezone.now().isoformat()
    save_mlflow_meta(user, refreshed)
    return refreshed


def instance_from_meta(user, meta: Dict[str, Any]) -> MlflowUiInstance:
    """把 JSON 元信息转成结构化对象。"""

    port = int(meta["port"])
    return MlflowUiInstance(
        username=user.username,
        pid=int(meta["pid"]),
        port=port,
        url=f"http://{settings.MLFLOW_UI_HOST}:{port}",
        backend_store_uri=meta["backend_store_uri"],
        meta_path=mlflow_meta_path(user),
    )


def reusable_instance(user) -> MlflowUiInstance | None:
    """
    如果用户已有可复用 MLflow UI，就直接返回。

    同时检查 pid 和端口，避免元信息还在但进程已经死掉，或者端口被别的服务占用。
    """

    meta = load_mlflow_meta(user)
    if not meta:
        return None
    pid = meta.get("pid")
    port = meta.get("port")
    if not port:
        return None
    if not port_is_open(settings.MLFLOW_UI_HOST, int(port)):
        return None
    meta = refresh_listener_pid(user, meta)
    return instance_from_meta(user, meta)


def start_mlflow_ui(user) -> MlflowUiInstance:
    """
    为当前用户启动一个 MLflow UI。

    进程只绑定 127.0.0.1；外部访问必须先经过 Django 登录入口拿到跳转链接。
    """

    user_root = storage_root(user)
    user_root.mkdir(parents=True, exist_ok=True)
    backend_store_uri = f"file://{mlruns_dir(user)}"
    port = find_free_port()
    log_path = user_runtime_dir(user) / "mlflow_ui.log"
    log_handle = log_path.open("ab")
    command = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        backend_store_uri,
        "--host",
        settings.MLFLOW_UI_HOST,
        "--port",
        str(port),
    ]
    process = subprocess.Popen(
        command,
        cwd=settings.BASE_DIR,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    listener_pid = process.pid
    listener_pid_list: list[int] = []
    for _ in range(30):
        if port_is_open(settings.MLFLOW_UI_HOST, port):
            listener_pid_list = listener_pids(port)
            if listener_pid_list:
                listener_pid = listener_pid_list[0]
            break
        if process.poll() is not None and not port_is_open(settings.MLFLOW_UI_HOST, port):
            break
        time.sleep(0.2)
    payload = {
        "username": user.username,
        "pid": listener_pid,
        "launcher_pid": process.pid,
        "listener_pids": listener_pid_list,
        "port": port,
        "host": settings.MLFLOW_UI_HOST,
        "url": f"http://{settings.MLFLOW_UI_HOST}:{port}",
        "backend_store_uri": backend_store_uri,
        "command": command,
        "log_path": str(log_path),
        "started_at": timezone.now().isoformat(),
    }
    save_mlflow_meta(user, payload)
    return instance_from_meta(user, payload)


def ensure_mlflow_ui(user) -> MlflowUiInstance:
    """返回当前用户可用的 MLflow UI；没有则自动启动。"""

    existing = reusable_instance(user)
    if existing is not None:
        return existing
    return start_mlflow_ui(user)


def stop_mlflow_ui(user) -> Dict[str, Any]:
    """停止当前用户的 MLflow UI，并更新元信息。"""

    meta = load_mlflow_meta(user)
    port = meta.get("port")
    pids: list[int] = []
    if port:
        pids.extend(listener_pids(int(port)))
    for raw_pid in meta.get("listener_pids", []):
        if isinstance(raw_pid, int):
            pids.append(raw_pid)
    if isinstance(meta.get("pid"), int):
        pids.append(int(meta["pid"]))

    stopped: list[int] = []
    for pid in sorted(set(pids), reverse=True):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
        stopped.append(pid)

    payload = dict(meta)
    payload["status"] = "stopped"
    payload["stopped_pids"] = stopped
    payload["stopped_at"] = timezone.now().isoformat()
    save_mlflow_meta(user, payload)
    return {
        "stopped": bool(stopped),
        "stopped_pids": stopped,
        "port": port,
        "meta_path": str(mlflow_meta_path(user)),
    }


def mlflow_status(user) -> Dict[str, Any]:
    """返回当前用户 MLflow UI 状态，供页面或 API 展示。"""

    meta = load_mlflow_meta(user)
    pid = meta.get("pid")
    port = meta.get("port")
    if port and port_is_open(settings.MLFLOW_UI_HOST, int(port)):
        meta = refresh_listener_pid(user, meta)
        pid = meta.get("pid")
    running = bool(port and port_is_open(settings.MLFLOW_UI_HOST, int(port)))
    return {
        "running": running,
        "pid": pid,
        "port": port,
        "url": meta.get("url"),
        "backend_store_uri": meta.get("backend_store_uri"),
        "meta_path": str(mlflow_meta_path(user)),
        "log_path": meta.get("log_path"),
    }

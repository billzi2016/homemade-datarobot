"""
Django 全局配置。

这里有两个原则：
1. 第一版先追求本地可跑、结构清楚，不提前引入复杂部署组件；
2. 训练执行仍然复用 mlflow-app，所以 Django 主要负责业务壳层。
"""

from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


def load_local_env(env_path: Path) -> None:
    """
    读取本地 .env。

    这里不引入 python-dotenv，避免为了一个本地原型增加额外依赖。
    规则故意保持简单：KEY=VALUE，一行一个；已存在的环境变量优先。
    """

    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_path(name: str, default: Path) -> Path:
    raw_value = os.environ.get(name)
    if not raw_value:
        return default
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return BASE_DIR / candidate


load_local_env(BASE_DIR / ".env")

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "django-insecure-homemade-datarobot-local-dev-key")
DEBUG = env_bool("DJANGO_DEBUG", True)
ALLOWED_HOSTS = [item.strip() for item in os.environ.get("DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",") if item.strip()]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "tasks",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "homemade_datarobot_web.security.SecurityHeadersMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "homemade_datarobot_web.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "homemade_datarobot_web.wsgi.application"
ASGI_APPLICATION = "homemade_datarobot_web.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

LANGUAGE_CODE = "zh-hans"
TIME_ZONE = "America/Indiana/Indianapolis"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# 浏览器侧基础安全配置。当前是本地开发项目，所以不强制 HTTPS；
# 后续如果部署到公网，再开启 SECURE_SSL_REDIRECT / secure cookie。
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = "DENY"
CSRF_COOKIE_SAMESITE = "Lax"
SESSION_COOKIE_SAMESITE = "Lax"
CSRF_COOKIE_HTTPONLY = False
CSRF_TRUSTED_ORIGINS = [
    "http://127.0.0.1:18743",
    "http://localhost:18743",
]

# 当前阶段默认使用 user_bizi 这层目录；后面接入鉴权后再替换成真实用户空间。
TASK_STORAGE_ROOT = env_path("TASK_STORAGE_ROOT", BASE_DIR / "storage" / "user_bizi")

# Django 页面里提供一个直达 MLflow 的链接。
# README 里约定 MLflow UI 使用 5001；如果本机换端口，只需要改这里。
MLFLOW_UI_BASE_URL = os.environ.get("MLFLOW_UI_BASE_URL", "http://127.0.0.1:5001")

# 本地子进程执行时，仍然复用现有训练入口。
MLFLOW_RUN_TASK_SCRIPT = BASE_DIR / "mlflow-app" / "run_task.py"

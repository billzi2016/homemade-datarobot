# Local Run

本文档记录本地以接近部署形态启动项目的方法。日常最短路径见 `QUICKSTART.md`。

## 1. 准备环境变量

```bash
cp .env.example .env
```

`.env` 不进入 git。当前关键项：

```text
DJANGO_SECRET_KEY=django-insecure-homemade-datarobot-local-dev-key
DJANGO_DEBUG=true
DJANGO_ALLOWED_HOSTS=127.0.0.1,localhost
DJANGO_RUNSERVER_HOST=127.0.0.1
DJANGO_RUNSERVER_PORT=18743

TASK_STORAGE_ROOT=storage/user_bizi
MLFLOW_UI_HOST=127.0.0.1
MLFLOW_UI_PORT_START=5001
MLFLOW_UI_PORT_END=5099

DJANGO_ADMIN_USERNAME=admin
DJANGO_ADMIN_EMAIL=admin@example.local
DJANGO_ADMIN_PASSWORD=admin123456

DJANGO_APP_USERNAME=bizi
DJANGO_APP_EMAIL=user@example.com
DJANGO_APP_PASSWORD=123456
```

当前用户名和 storage 目录同步：

```text
bizi  -> storage/user_bizi
admin -> storage/user_admin
```

## 2. 初始化数据库和用户

创建或更新 admin：

```bash
./scripts/create_admin.sh
```

创建或更新普通用户：

```bash
./scripts/create_user.sh
```

`scripts/create_user.sh` 可以进入 git；它读取 `.env` 中的 `DJANGO_APP_*`。真实邮箱和密码只留在 `.env`，不会进入版本库。

## 3. 启动 Django

开发调试可以用：

```bash
python3 manage.py runserver 127.0.0.1:18743
```

如果想用更接近生产的 WSGI 方式，可以用 gunicorn：

```bash
gunicorn homemade_datarobot_web.wsgi:application \
  --bind 127.0.0.1:18743 \
  --workers 2 \
  --timeout 300 \
  --access-logfile -
```

说明：

- 当前训练任务由 Django 启动本地子进程。
- 训练可能较久，所以 gunicorn `--timeout` 不要设太短。
- 当前阶段不使用 Celery / RabbitMQ，避免本地原型复杂度过高。

## 4. 打开 MLflow UI

当前一个用户一个 `mlruns`，Django 会在用户点击“打开 MLflow”时自动启动或复用 MLflow UI；如果页面显示正在运行，也可以停止当前用户自己的 MLflow UI。

端口池由 `.env` 控制：

```text
MLFLOW_UI_HOST=127.0.0.1
MLFLOW_UI_PORT_START=5001
MLFLOW_UI_PORT_END=5099
```

进程元信息会写到：

```text
storage/user_bizi/.django_runtime/mlflow_ui.json
```

## 5. 页面和 API

Django 页面：

```text
http://127.0.0.1:18743/
```

Swagger UI 需要登录后访问：

```text
http://127.0.0.1:18743/api/docs
```

OpenAPI JSON 也需要登录后访问：

```text
http://127.0.0.1:18743/api/openapi.json
```

## 6. 当前限制

- sqlite 只适合本地开发，不适合作为多人并发环境的最终数据库。
- 本地子进程适合第一版验证，不适合大规模并发训练。
- MLflow 当前按用户目录和本地进程隔离，后续上服务端部署时需要统一 tracking server 和 artifact store。

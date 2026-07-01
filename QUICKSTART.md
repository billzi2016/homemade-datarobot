# Quickstart

本文件只写最短启动路径。更完整的系统说明见 `README.md`。

## 1. 进入项目

```bash
cd /Users/bizi/Desktop/GitHub/homemade-datarobot
```

## 2. 准备本地环境变量

```bash
cp .env.example .env
```

`.env` 不会进入版本库。默认本地配置包括：

```text
DJANGO_RUNSERVER_HOST=127.0.0.1
DJANGO_RUNSERVER_PORT=18743
TASK_STORAGE_ROOT=storage/user_bizi
MLFLOW_UI_HOST=127.0.0.1
MLFLOW_UI_PORT_START=5001
MLFLOW_UI_PORT_END=5099
DJANGO_ADMIN_USERNAME=admin
DJANGO_ADMIN_PASSWORD=admin123456
```

## 3. 启动 Django 页面

```bash
python3 manage.py runserver 127.0.0.1:18743
```

页面入口：

```text
http://127.0.0.1:18743/
```

Swagger / OpenAPI 需要先登录 Django：

```text
Swagger UI: http://127.0.0.1:18743/api/docs
OpenAPI JSON: http://127.0.0.1:18743/api/openapi.json
API 示例: http://127.0.0.1:18743/api/tasks
```

## 4. 初始化 Django admin

```bash
./scripts/create_admin.sh
```

账号来自 `.env`：

```text
DJANGO_ADMIN_USERNAME=admin
DJANGO_ADMIN_EMAIL=admin@example.local
DJANGO_ADMIN_PASSWORD=admin123456
```

Admin 页面：

```text
http://127.0.0.1:18743/admin/
```

## 5. 初始化普通用户

`.env` 中已经给出本地普通用户示例：

```text
DJANGO_APP_USERNAME=bizi
DJANGO_APP_EMAIL=user@example.com
DJANGO_APP_PASSWORD=123456
```

创建或更新该用户：

```bash
./scripts/create_user.sh
```

该脚本可以进入 git；真实邮箱和密码只放在 `.env`，不会进入版本库。

登录后，用户名会和 storage 目录同步：

```text
bizi -> storage/user_bizi
```

## 6. 打开 MLflow UI

登录 Django 后进入 task 详情页，点击“打开 MLflow”。

Django 会自动为当前用户启动或复用一个 MLflow UI，端口从 `5001-5099` 中选择空闲值，避免多个用户同时使用时冲突。

## 7. 当前使用方式

- Django 页面负责创建任务、启动训练、看状态、下载结果。
- API 使用 `django-ninja`，Swagger UI 位于 `/api/docs`。
- 训练仍然由 `mlflow-app/run_task.py` 执行。
- MLflow UI 由 Django 按登录用户自动启动或复用。
- 当前不使用 Celery / RabbitMQ，Django 通过本地子进程启动训练。
- 调试时默认只跑 `task_iris`，避免 Titanic 全量调试太慢。

## 8. 直接运行已有示例 task

```bash
python3 mlflow-app/run_task.py storage/user_bizi/task_iris
```

运行完成后看：

```text
storage/user_bizi/task_iris/outputs/
storage/user_bizi/task_iris/run_state.json
```

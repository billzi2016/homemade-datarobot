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
MLFLOW_UI_BASE_URL=http://127.0.0.1:5001
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

Swagger / OpenAPI：

```text
Swagger UI: http://127.0.0.1:18743/api/docs
OpenAPI JSON: http://127.0.0.1:18743/api/openapi.json
API 示例: http://127.0.0.1:18743/api/tasks
```

## 4. 初始化 Django admin

```bash
./scripts/create_admin.sh
```

默认本地账号：

```text
DJANGO_ADMIN_USERNAME=admin
DJANGO_ADMIN_PASSWORD=admin123456
```

Admin 页面：

```text
http://127.0.0.1:18743/admin/
```

## 5. 启动 MLflow UI

```bash
mlflow ui --backend-store-uri file:///Users/bizi/Desktop/GitHub/homemade-datarobot/storage/user_bizi/mlruns --port 5001
```

MLflow 页面：

```text
http://127.0.0.1:5001
```

## 6. 当前使用方式

- Django 页面负责创建任务、启动训练、看状态、下载结果。
- API 使用 `django-ninja`，Swagger UI 位于 `/api/docs`。
- 训练仍然由 `mlflow-app/run_task.py` 执行。
- 当前不使用 Celery / RabbitMQ，Django 通过本地子进程启动训练。
- 调试时默认只跑 `task_iris`，避免 Titanic 全量调试太慢。

## 7. 直接运行已有示例 task

```bash
python3 mlflow-app/run_task.py storage/user_bizi/task_iris
```

运行完成后看：

```text
storage/user_bizi/task_iris/outputs/
storage/user_bizi/task_iris/run_state.json
```

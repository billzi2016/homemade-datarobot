# K8s Plan

本文档是未来部署设想，不是当前阶段已经实现的内容。

当前项目先用本地 Django + 本地子进程 + 文件型 MLflow，原因是资源有限、验证重点在可控 AutoML 流程本身。等功能稳定后，再把任务调度、数据库和 MLflow 服务拆成可扩展组件。

## 1. 目标架构

未来上 K8s 后，建议拆成这些组件：

```text
Browser
  -> Ingress
  -> Django Web/API
  -> PostgreSQL
  -> Celery Worker
  -> RabbitMQ 或 Redis
  -> MLflow Tracking Server
  -> Artifact Store
```

## 2. Django 层

Django 负责：

- 用户注册、登录、权限判断
- task 创建、配置写入、文件上传
- 任务状态查询
- 下载入口
- MLflow 页面跳转

K8s 中 Django 可以横向扩容多个 pod，但需要注意：

- session 建议放到数据库或 Redis，不依赖单机内存
- 上传文件不能只写容器本地磁盘
- task 状态不能只靠本地进程 PID
- `/api/docs` 和 `/api/openapi.json` 继续要求登录后访问

## 3. 数据库

当前本地使用 sqlite，未来应切到 PostgreSQL。

PostgreSQL 保存：

- 用户
- 用户 profile
- task 元数据
- task 状态
- 子任务状态
- 文件索引
- MLflow run 关联信息

sqlite 在本地非常方便，但多人并发、容器多副本、连接管理和备份恢复都不适合作为最终方案。

## 4. 任务队列

当前 Django 直接启动本地 Python 子进程：

```text
Django -> subprocess -> mlflow-app/run_task.py
```

K8s 环境建议改成：

```text
Django -> Celery -> RabbitMQ/Redis -> Worker Pod -> mlflow-app/run_task.py
```

这样可以获得：

- 任务排队
- worker 横向扩容
- 失败重试
- 任务取消
- 运行日志集中管理
- Django Web pod 不被长任务拖住

RabbitMQ 更适合严肃任务队列；Redis 配置更轻，原型阶段也可以接受。

## 5. MLflow 层

当前本地使用文件型 `mlruns`：

```text
storage/user_bizi/mlruns
```

未来建议部署独立 MLflow Tracking Server：

```text
MLflow Tracking Server
  -> PostgreSQL 或兼容数据库保存 tracking metadata
  -> S3 / MinIO / PVC 保存 artifacts
```

扩容方式：

- MLflow tracking server 可以多副本，但后端 store 必须共享
- artifact store 不能用单个 pod 本地磁盘
- 每个 task 继续映射为一个 MLflow experiment
- 每个模型、分析项继续映射为 experiment 下的独立 run

## 6. 存储

未来需要把当前 `storage/user_xxx/task_xxx` 抽象成共享存储：

- 小规模可以用 PVC
- 更通用可以用对象存储，例如 S3 或 MinIO
- 数据上传、模型文件、图表、预测结果都应该进入统一 artifact 管理

权限模型仍然按用户隔离：

```text
user_bizi/task_iris
user_alice/task_churn
```

## 7. 当前为何暂不实现

现在没有必要马上引入 Celery、RabbitMQ、PostgreSQL 和完整 K8s 编排。这样会显著增加部署、排错和资源成本，而当前阶段更需要先确认：

- sklearn / torch / analysis 流程是否稳定
- MLflow 展示结构是否符合预期
- task 目录结构是否合理
- Django 页面是否能支撑上传、启动、进度、下载

因此当前版本先保留轻量本地架构，等核心流程稳定后再升级到队列化和 K8s 部署。

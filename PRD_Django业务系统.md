# PRD：Django 业务系统

## 1. 文档定位

本文档定义仓库最外层的 Django 业务系统。它不是实验训练内核，而是整个项目对用户暴露的业务外壳。

它负责：

- 数据上传
- 结果下载
- 任务创建
- 任务状态展示
- MLflow 入口跳转

真正的数据分析、降维、训练、评估、实验记录，由 `mlflow-app` 下的 MLflow 实验子系统负责。

---

## 2. 系统目标

构建一个轻量但结构清楚的 Web 系统，作为本地 DataRobot 风格平台的最外层入口，让用户可以完成：

- 上传表格数据
- 选择任务参数
- 提交实验
- 实时查看状态
- 下载模型与结果
- 进入 MLflow 查看实验详情

---

## 3. 首版架构原则

### 3.1 当前阶段不引入消息队列

首版不使用 `Celery`、`RabbitMQ`，避免在原型阶段引入过多部署复杂度。

### 3.2 采用方案 A

当前阶段采用你确认的方案：

- 用户提交任务
- Django 启动本地 Python 子进程去跑训练脚本
- Django 记录任务 ID 和进程状态
- 前端用 SSE 订阅状态

也就是：

`Django + subprocess + SSE + MLflow + 本地存储`

### 3.3 共享配置

系统不再使用一个全局混放的 `config.yaml`。后续采用 `storage/user_bizi/tasks/task_iris` 这种任务目录作为基本管理单元，每个任务目录自带自己的 `config.yaml`，由 Django 和 MLflow 实验子系统共享。

### 3.4 当前阶段先做 MLflow，Django PRD 同步维护

当前阶段实现顺序上先做 `MLflow` 相关部分，但 Django PRD 仍要同步维护，保证整体规范一致。

在 spec 未收敛前，不进入 `config.yaml`、代码结构、任务执行逻辑的正式实现。

---

## 4. 系统边界

### 4.1 Django 负责什么

- 创建与管理任务目录，例如 `storage/user_bizi/tasks/task_iris`
- 上传数据文件到对应实验目录下的 `data/`
- 管理数据集元信息
- 创建实验任务
- 维护任务状态
- 提供 SSE 状态流
- 汇总可下载结果
- 提供 MLflow run 链接

### 4.2 Django 不负责什么

- 不直接实现 `GridSearchCV`
- 不直接实现 `torch` 训练循环
- 不直接实现 `PCA / t-SNE / UMAP`
- 不直接实现模型评估逻辑

这些全部交给 MLflow 实验子系统。

---

## 5. 功能模块

### 5.1 数据上传模块

首期支持：

- CSV
- Parquet
- Excel（可选）

上传后至少记录：

- 所属 `task_id`
- 数据集名称
- 文件路径
- 文件类型
- 文件大小
- 上传时间
- 行数与列数

### 5.2 任务创建模块

任务创建页至少允许用户指定：

- `task_id`
- 数据文件
- 目标列
- 任务类型
- 是否启用分析
- 是否启用 sklearn
- 是否启用 torch
- 主指标
- CV 参数

### 5.3 任务执行模块

执行流程：

1. Django 接收任务请求
2. 生成或确认当前 `task_id`
3. 确保对应的 `storage/user_bizi/tasks/task_iris` 这类任务目录存在
4. 生成 `task_id`
5. 记录任务状态为 `pending`
6. 用 `subprocess` 启动 Python 子进程
7. 子进程调用 `mlflow-app` 内部训练/分析逻辑
8. Django 持续更新任务状态
9. 任务结束后记录 `mlflow_run_id`、结果路径、错误信息

### 5.4 SSE 状态推送模块

SSE 负责推送：

- 当前阶段
- 当前模型
- 当前 fold
- 当前 epoch
- 当前指标
- 当前进度百分比
- 成功 / 失败状态

### 5.5 结果下载模块

用户可下载：

- 预测结果 CSV
- 评估结果
- 图表文件
- sklearn 模型文件
- torch 权重文件
- 配置快照

### 5.6 MLflow 入口模块

任务详情页应显示：

- `MLflow Run` 链接
- 数据集关联实验链接
- 最优模型对应链接

---

## 6. 共享 config.yaml 要求

`config.yaml` 放在每个 `storage/user_bizi/tasks/task_iris` 这类任务目录内，并由 Django 与 MLflow 共用。

必须包含列类型配置：

```yaml
features:
  numeric_columns:
    - age
    - income
  categorical_columns:
    - city
    - gender
  onehot_columns:
    - city
    - gender
```

字段含义：

- `numeric_columns`：按数值列处理
- `categorical_columns`：按类别列处理
- `onehot_columns`：明确需要 one-hot 编码的列

同时需要明确：

- 一个 `task_iris` 对应一份独立配置
- 不同任务目录的配置不能混在一起
- Django 创建任务时应围绕 `task_id` 组织配置读写

---

## 7. 任务状态模型

至少需要记录：

- `task_id`
- `task_id`
- `status`
- `progress`
- `current_stage`
- `current_model`
- `pid`
- `mlflow_run_id`
- `result_path`
- `error_message`
- `created_at`
- `updated_at`

建议状态枚举：

- `pending`
- `running`
- `success`
- `failed`
- `cancelled`

---

## 8. 首版页面建议

首版建议至少有以下页面：

- 实验列表页
- 数据集列表页
- 数据集上传页
- 任务创建页
- 任务详情页
- 下载页

任务详情页重点展示：

- 当前状态
- 当前阶段
- 当前模型
- 简化日志
- 下载列表
- MLflow 链接

---

## 9. 后续升级路径

当前阶段：

- `Django + subprocess + SSE`

后续如果任务量增加，再升级到：

- `Django + Redis/Celery`
或
- `Django + RQ + Redis`

同时在下个阶段需要补上任务级权限控制，即：

- 用户是否可以读取某个 `storage/user_bizi/tasks/task_iris`
- 用户是否可以下载某个 `storage/user_bizi/tasks/task_iris` 下的结果
- 用户是否可以访问某个 `storage/user_bizi/tasks/task_iris` 对应的 MLflow 链接

当前版本只在 PRD 中预留这部分要求，不实现鉴权逻辑。

---

## 10. 结论

Django 系统在这套架构里的角色非常明确：它是业务外壳，不是训练内核。

它把用户请求接进来，把任务启动起来，把状态展示出来，把结果交付出去，并把更细的实验详情导向 MLflow。

在资源组织上，Django 后续要以 `storage/user_bizi/tasks/task_iris` 这类任务目录作为最小管理单位，围绕 `task_id` 来管理数据、配置、结果和访问权限。

## 9.1 task 与 run_state 管理要求

Django 侧需要明确支持 `task` 这一综合体概念：

- 一个 `task` 表示一次完整任务
- 一个 `task` 对应一个主 MLflow run
- 一个 `task` 目录中既有上传数据，也有可下载结果

因此 Django 需要感知并展示：

- `task_id`
- 当前运行状态
- 是否支持恢复
- 最近保存点时间
- 已完成步骤
- 已完成模型

其中 `run_state.json` 必须作为 Django 可读取的状态来源之一，用于展示对人类友好的任务进度，而不是只显示一个抽象的“running / failed”。

# homemade-datarobot

这是一个面向表格数据的本地 AutoML / Analysis / MLflow 原型项目。

当前阶段已经落下来的核心能力：

- `analysis`
  - `PCA`
  - `t-SNE`
  - `UMAP`
- `sklearn`
  - `logistic_regression`
  - `random_forest`
  - `xgboost`
- `torch`
  - `mlp`
  - `cnn1d`
- `MLflow`
  - 一个 `task` 对应一个主 run
  - 子实验使用 nested run
  - 每个 `task` 有自己的 `run_state.json`

当前还没有正式接入的部分：

- `torch.tabnet`
  - 原因是当前环境里没有安装 `pytorch-tabnet`

---

## 目录说明

当前仓库的核心目录如下：

```text
homemade-datarobot/
├── README.md
├── PRD_Django业务系统.md
├── mlflow-app/
│   ├── PRD_可控AutoML方案.md
│   ├── run_task.py
│   ├── task_runtime.py
│   ├── analysis_runner.py
│   ├── sklearn_runner.py
│   └── torch_runner.py
├── storage/
│   └── user_bizi/
│       └── tasks/
│           ├── task_iris/
│           └── task_titanic_dataset/
└── test/
    └── test_data/
```

其中：

- `mlflow-app/run_task.py`
  - 单个 task 的总入口
- `mlflow-app/task_runtime.py`
  - task 路径、配置、运行状态管理
- `mlflow-app/analysis_runner.py`
  - `PCA / t-SNE / UMAP`
- `mlflow-app/sklearn_runner.py`
  - sklearn 主线
- `mlflow-app/torch_runner.py`
  - torch 主线

---

## task 目录结构

每个 task 都是一个独立任务综合体，例如：

```text
storage/user_bizi/tasks/task_iris/
├── config.yaml
├── data/
│   └── raw/
├── outputs/
│   ├── analysis/
│   ├── metrics/
│   ├── models/
│   └── predictions/
├── artifacts/
├── checkpoints/
└── run_state.json
```

说明：

- `config.yaml`
  - 单个 task 的配置文件
- `data/raw/`
  - task 输入数据
- `outputs/analysis/`
  - `PCA / t-SNE / UMAP` 的 2D 结果
- `outputs/metrics/`
  - 各模型指标与汇总结果
- `outputs/models/`
  - 模型最终文件
- `outputs/predictions/`
  - 最终预测结果
- `checkpoints/`
  - torch 训练保存点
- `run_state.json`
  - 当前 task 的人类可读运行状态

同时，当前用户 `bizi` 共享一个用户级 MLflow 仓库：

```text
storage/user_bizi/mlruns/
```

也就是说：

- task 自己保存自己的数据、结果、状态
- 同一个用户名下的多个 task 共用一个 `mlruns`
- 因此一个用户只需要启动一个 MLflow UI

---

## 已准备好的示例 task

当前本地已经准备了两个示例 task：

### task_iris

- 数据：`Iris.csv`
- 类型：多分类

### task_titanic_dataset

- 数据：`Titanic-Dataset.csv`
- 类型：二分类

---

## 如何运行

以下命令都在仓库根目录执行：

```bash
cd /Users/bizi/Desktop/GitHub/homemade-datarobot
```

### 运行 task_iris

```bash
python3 mlflow-app/run_task.py storage/user_bizi/tasks/task_iris
```

### 运行 task_titanic_dataset

```bash
python3 mlflow-app/run_task.py storage/user_bizi/tasks/task_titanic_dataset
```

运行完成后，会在对应 task 目录下生成：

- `run_state.json`
- `outputs/analysis/*.csv`
- `outputs/metrics/*`
- `outputs/predictions/*`
- `outputs/models/*`
- `mlruns/`

---

## MLflow 网页怎么开

当前设计是：

- `user_bizi` 下的所有 task 共用一个 `mlruns`
- 因此只需要启动一个 MLflow UI

### 打开 user_bizi 的 MLflow 网页

```bash
mlflow ui --backend-store-uri file:///Users/bizi/Desktop/GitHub/homemade-datarobot/storage/user_bizi/mlruns --port 5001
```

网页地址：

```text
http://127.0.0.1:5001
```

这个页面里会同时看到：

- `task_iris`
- `task_titanic_dataset`

---

## 当前实现状态

### 已完成

- 单 task 配置读取
- 单 task `run_state.json`
- `analysis` nested run
- `sklearn` nested run
- `torch` nested run
- `mlp` / `cnn1d` 真实训练
- 同一个用户下的多个 task 共用一个 `mlruns`

### 当前限制

- `tabnet` 尚未接入运行，原因是缺少 `pytorch-tabnet`
- 目前还没有 Django 页面层
- 目前还没有正式的 task 自动发号逻辑
- 当前更偏向本地原型和实验内核

---

## 运行结果怎么看

建议优先看这几个文件：

### task 运行状态

- `storage/user_bizi/tasks/task_iris/run_state.json`
- `storage/user_bizi/tasks/task_titanic_dataset/run_state.json`

### sklearn 汇总结果

- `storage/user_bizi/tasks/task_iris/outputs/metrics/sklearn_summary.csv`
- `storage/user_bizi/tasks/task_titanic_dataset/outputs/metrics/sklearn_summary.csv`

### torch 汇总结果

- `storage/user_bizi/tasks/task_iris/outputs/metrics/torch_summary.csv`
- `storage/user_bizi/tasks/task_titanic_dataset/outputs/metrics/torch_summary.csv`

### analysis 结果

- `storage/user_bizi/tasks/task_iris/outputs/analysis/`
- `storage/user_bizi/tasks/task_titanic_dataset/outputs/analysis/`

---

## 说明

当前代码里已经加了大量中文注释，重点函数、状态流转、路径规则、保存策略、恢复逻辑入口都写了中文说明，后续继续拆分时也保持同样风格。

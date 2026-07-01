# homemade-datarobot

这是一个面向表格数据的本地 AutoML / Analysis / MLflow 原型项目。

当前阶段已经落下来的核心能力：

- `analysis`
  - `PCA`
  - `t-SNE`
  - `UMAP`
- `sklearn`
  - `logistic_regression`
  - `svm`
  - `random_forest`
  - `extra_trees`
  - `gradient_boosting`
  - `hist_gradient_boosting`
  - `decision_tree`
  - `knn`
  - `gaussian_nb`
  - `bernoulli_nb`
  - `multinomial_nb`
  - `xgboost`
  - `lightgbm`
- `torch`
  - `mlp`
  - `cnn1d`
  - `tabnet`
- `MLflow`
  - 一个 `task` 对应一个主 run
  - 子实验使用 nested run
  - 每个 `task` 有自己的 `run_state.json`
- `search`
  - `auto`
  - `grid`
  - `halving_grid`
  - `optuna`
- `classification default`
  - `stratify split`
  - `imbalance balance`

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
│       ├── mlruns/
│       ├── task_iris/
│       └── task_titanic_dataset/
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
storage/user_bizi/task_iris/
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
python3 mlflow-app/run_task.py storage/user_bizi/task_iris
```

### 运行 task_titanic_dataset

```bash
python3 mlflow-app/run_task.py storage/user_bizi/task_titanic_dataset
```

运行完成后，会在对应 task 目录下生成：

- `run_state.json`
- `outputs/analysis/*.csv`
- `outputs/plots/*`
- `outputs/metrics/*`
- `outputs/predictions/*`
- `outputs/models/*`

注意：

- task 目录本身不再单独维护一个私有 `mlruns`
- 同一个用户下的多个 task 统一写入 `storage/user_bizi/mlruns`

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

进入单个 experiment 后，会看到：

- `analysis.pca / analysis.tsne / analysis.umap`
- `sklearn.xxx`
- `torch.xxx`

这些具体模型和分析项以 nested run 方式挂在对应 task run 下。

---

## 搜索策略

当前支持四种写法：

- `auto`
- `grid`
- `halving_grid`
- `optuna`

默认推荐使用：

```yaml
search:
  method: auto
  n_trials: 12
```

当前 `auto` 的策略是：

- `torch` 模型优先 `optuna`
- `xgboost / lightgbm / svm / svr` 优先 `optuna`
- 小数据集优先 `grid`
- 中等数据集优先 `halving_grid`
- 更大数据集优先 `optuna`

这里有一个明确约束：

- 默认链路不能退化成 `none`

---

## 分类默认行为

分类任务默认启用以下策略：

- `train_test_split(..., stratify=y)`
- 尽量对模型启用原生类别权重
- 如果模型原生不支持，则对训练集做随机过采样
- `torch` 额外在 loss 中加入类别权重

也就是说，首版不是“部分模型做平衡”，而是尽量让每个分类模型都吃到平衡机制。

---

## 当前实现状态

### 已完成

- 单 task 配置读取
- 单 task `run_state.json`
- `analysis` nested run
- `sklearn` nested run
- `torch` nested run
- `mlp` / `cnn1d` / `tabnet` 训练链路接入
- 同一个用户下的多个 task 共用一个 `mlruns`
- `auto / grid / halving_grid / optuna` 搜索策略接入
- 分类任务默认分层切分与类别平衡

### 当前限制

- 目前还没有 Django 页面层
- 目前还没有正式的 task 自动发号逻辑
- 当前更偏向本地原型和实验内核

---

## 运行结果怎么看

建议优先看这几个文件：

### task 运行状态

- `storage/user_bizi/task_iris/run_state.json`
- `storage/user_bizi/task_titanic_dataset/run_state.json`

### sklearn 汇总结果

- `storage/user_bizi/task_iris/outputs/metrics/sklearn_summary.csv`
- `storage/user_bizi/task_titanic_dataset/outputs/metrics/sklearn_summary.csv`

### torch 汇总结果

- `storage/user_bizi/task_iris/outputs/metrics/torch_summary.csv`
- `storage/user_bizi/task_titanic_dataset/outputs/metrics/torch_summary.csv`

### analysis 结果

- `storage/user_bizi/task_iris/outputs/analysis/`
- `storage/user_bizi/task_titanic_dataset/outputs/analysis/`

---

## 说明

当前代码里已经加了大量中文注释，重点函数、状态流转、路径规则、保存策略、恢复逻辑入口都写了中文说明，后续继续拆分时也保持同样风格。

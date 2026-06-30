# PRD：MLflow 实验子系统方案

## 1. 文档目标

本文档用于定义一个面向表格数据（Tabular Data）的 `MLflow 实验子系统`。该子系统不是整个站点的业务外壳，而是专门负责实验执行、实验记录、实验结果展示的内核部分。目标不是引入高度封装、行为不透明的 AutoML 框架，而是基于 `scikit-learn`、`PyTorch`、`MLflow` 自行组织一套结构清晰、行为可控、便于调试和扩展的训练与实验管理框架。

本方案重点解决以下问题：

- 将常见机器学习模型统一纳入一个可配置的训练框架。
- 将 `sklearn` 传统机器学习模型与 `torch` 深度学习模型分开实现，降低耦合。
- 使用按实验目录独立存放的 `config.yaml` 管理数据、目标列、任务类型、交叉验证、评分指标、模型启用开关等关键参数，并与 Django 业务系统共享同一个实验单元配置。
- 所有训练过程、参数、指标、模型产物、配置快照统一挂载到 `MLflow`。
- 代码与文档要求包含大量中文注释，确保后续维护者可以直接理解设计意图、函数职责和关键实现逻辑。

需要特别强调的是：本 PRD 只描述 `MLflow` 以及它下面真正干活的所有实验模块，包括分析实验、降维实验、`sklearn` 实验、`torch` 实验，不再把 Django 业务层混在同一份文档里。

本项目后续采用 `storage/task_000001` 这种目录作为最小任务管理单元。每个任务目录必须自带该任务自己的配置、数据、结果与 artifact，不能和其他任务混放。

---

## 2. 产品定位

### 2.1 产品目标

构建一个“类似 DataRobot，但更轻量、更透明、更可控”的本地实验子系统，优先服务以下场景：

- 表格数据建模
- 二分类任务
- 多分类任务
- 回归任务

该系统暂不追求“大而全”，而追求：

- 行为可预测
- 训练逻辑可追踪
- 搜索过程可解释
- 错误来源可定位
- 模块职责清晰

### 2.2 产品边界

当前版本不作为全自动黑盒 AutoML 平台，不包含以下重点能力：

- 不优先支持时间序列 AutoML
- 不优先支持图像、文本、多模态任务
- 不优先支持复杂特征工程自动发现
- 不优先支持超大规模分布式训练
- 不优先支持在线部署平台

当前版本的重点是：先把“本地可控 AutoML / Analysis / MLflow 实验核心骨架”搭起来。

### 2.3 与 Django 业务系统的关系

本子系统与 Django 的分工必须明确：

- Django 负责业务外壳：上传、下载、任务创建、任务状态页面、MLflow 入口
- 本子系统负责实验执行：分析、降维、训练、评估、记录、可视化产物生成

也就是说：

- Django 不是训练引擎
- MLflow 不是上传下载门户
- 两者通过同一个 `task` 目录协同工作
- Django 管理 `task`
- MLflow 子系统消费 `task`，并把实验结果写回 `task`

---

## 3. 核心设计原则

### 3.1 可控优先

之所以优先选用 `sklearn` 常规模型，是因为很多自动化封装很深的 AutoML 库虽然方便，但在以下方面经常出现问题：

- 报错信息不清晰
- 内部预处理和搜索流程隐藏过深
- 不容易定位是数据问题、模型问题还是搜索空间问题
- 扩展自定义模型时成本高
- 训练日志与实验结果不够透明

因此，本项目采用“自己机械化组织常用模型”的方式，把模型、搜索、评估、追踪明确拆开。

### 3.2 结构隔离

`sklearn` 与 `torch` 分为两条主线，原因如下：

- 二者训练范式不同
- 二者参数搜索方式不同
- 二者模型保存与加载方式不同
- 二者训练日志粒度不同
- 二者早停与验证控制机制不同

如果强行做成一套完全一致的内部实现，短期会让代码结构更混乱。因此本项目要求：

- `sklearn/` 单独组织传统机器学习管线
- `torch/` 单独组织深度学习管线
- 通过最外层配置和统一的实验记录接口实现“表面统一”

### 3.3 配置驱动

所有关键行为尽量通过每个 `task` 目录下独立的 `config.yaml` 控制，而不是散落在代码常量中，包括：

- 数据路径
- 目标列
- 任务类型
- 特征列策略
- 数值列
- 类别列
- 需要 one-hot 的列
- 交叉验证参数
- 指标
- 启用模型列表
- 搜索空间
- 训练参数
- MLflow 参数

### 3.4 注释要求极高

本项目明确要求：

- 文档为中文
- 代码为中文注释
- 函数职责、关键分支、长逻辑链、容易误解的实现点必须写详细中文注释
- 不只写“做了什么”，还要写“为什么这么做”

---

## 4. 目标用户与使用场景

### 4.1 目标用户

- 希望快速试多种模型的数据科学工作者
- 不想依赖黑盒 AutoML 的算法工程师
- 需要保留训练透明度和实验追踪能力的研发人员
- 想从传统机器学习逐步过渡到深度学习表格建模的开发者

### 4.2 典型使用流程

1. 用户通过 Django 创建一个新的任务单元，例如 `storage/task_000001`。
2. 用户通过 Django 上传数据，或直接将数据放入该实验目录下的 `data/`。
3. 用户在该实验目录下的 `config.yaml` 中指定数据路径、目标列、任务类型、CV、模型清单等参数。
4. 系统按配置自动执行：
   - 数据读取
   - 数据分析 / 降维（按需）
   - 数据预处理
   - 模型训练
   - 参数搜索
   - 指标评估
   - 最优模型筛选
   - MLflow 实验记录
5. 用户在 `MLflow UI` 中查看该实验目录对应的各模型实验结果、分析图表、降维结果。
6. 用户对最优模型进行导出、复盘和后续迭代。

---

## 5. 总体目录设计

建议项目结构如下：

```text
homemade-datarobot/
├── storage/
│   ├── task_000001/
│   │   ├── config.yaml
│   │   ├── data/
│   │   │   ├── raw/
│   │   │   ├── processed/
│   │   │   └── splits/
│   │   ├── outputs/
│   │   │   ├── predictions/
│   │   │   ├── metrics/
│   │   │   └── models/
│   │   ├── artifacts/
│   │   ├── checkpoints/
│   │   └── run_state.json
│   └── task_000002/
│       ├── config.yaml
│       ├── data/
│       ├── outputs/
│       ├── artifacts/
│       ├── checkpoints/
│       └── run_state.json
└── mlflow-app/
    ├── sklearn/
    ├── torch/
    ├── analysis/
    ├── mlflow/
    └── PRD_可控AutoML方案.md
```

### 5.1 顶层目录说明

- `storage/task_000001/config.yaml`
  - 单个任务的主配置文件。
  - Django 与 MLflow 实验子系统共享。
  - 一个任务一份，不同任务之间不能混用。

- `storage/task_000001/data/`
  - 单个任务的数据目录。
  - 包括原始数据、训练集、验证集、测试集、处理后的数据、中间缓存数据等。
  - 数据必须和所属任务绑定。

- `storage/task_000001/outputs/`
  - 单个任务的输出目录。
  - 放预测结果、评估汇总、模型文件、图表等最终产物。

- `storage/task_000001/run_state.json`
  - 单个任务的可读状态文件。
  - 用于记录当前状态、已完成步骤、已完成模型、待执行步骤、恢复信息。

- `storage/task_000001/checkpoints/`
  - 单个任务的保存点目录。
  - 用于中断恢复，尤其是 `torch` 训练恢复。

- `sklearn/`
  - 传统机器学习主线。
  - 负责预处理、建模、CV 搜索、评估、推理。

- `torch/`
  - 深度学习主线。
  - 负责 `MLP`、`1D CNN`、`TabNet` 等模型及对应训练流程。

- `mlflow/`
  - 负责实验跟踪。
  - 统一封装 run 创建、参数记录、指标记录、模型记录、artifact 记录等能力。

- `outputs/`
  - 放预测结果、评估结果汇总、导出的模型等最终产物。

---

## 6. 功能范围设计

## 6.1 数据层

### 6.1.1 数据输入能力

首期建议支持：

- CSV
- Parquet
- Excel（可选，优先级低于 CSV / Parquet）

### 6.1.2 数据任务类型

首期支持：

- 二分类
- 多分类
- 回归

### 6.1.3 数据处理要求

数据处理层需要具备以下能力：

- 指定目标列 `target`
- 指定忽略列 `drop_columns`
- 支持显式指定数值列与类别列
- 支持显式指定需要 one-hot 的类别列
- 缺失值处理
- 类别编码
- 数值标准化或归一化
- 可选的数据切分
- 可复用的数据预处理配置保存

### 6.1.4 数据目录约束

所有数据都统一进入对应任务目录下的 `data/`，包括：

- 原始导入数据
- 训练集
- 验证集
- 测试集
- 预处理后数据
- 中间缓存数据
- 模型推理输入样本

这样做的原因是：

- 每个任务相互隔离
- 不会因为多个任务并行而混淆数据来源
- 后续排查、归档、恢复执行更直接

---

## 6.2 sklearn 传统机器学习模块

### 6.2.1 模块目标

`sklearn/` 模块负责承载常用传统机器学习模型，并基于 `Pipeline`、`ColumnTransformer`、`GridSearchCV` 形成稳定、可控的训练体系。

### 6.2.2 纳入模型范围

#### 分类模型

- `LogisticRegression`
- `SVC`
- `RandomForestClassifier`
- `ExtraTreesClassifier`
- `GradientBoostingClassifier`
- `HistGradientBoostingClassifier`
- `DecisionTreeClassifier`
- `KNeighborsClassifier`
- `GaussianNB`
- `BernoulliNB`
- `MultinomialNB`（仅在数据条件满足时启用）
- `XGBoostClassifier`
- `LightGBMClassifier`

#### 回归模型

- `LinearRegression`
- `Ridge`
- `Lasso`
- `ElasticNet`
- `SVR`
- `RandomForestRegressor`
- `ExtraTreesRegressor`
- `GradientBoostingRegressor`
- `HistGradientBoostingRegressor`
- `DecisionTreeRegressor`
- `KNeighborsRegressor`
- `XGBoostRegressor`
- `LightGBMRegressor`

### 6.2.3 为什么纳入这些模型

这些模型覆盖了表格数据中最常见的建模范式：

- 线性类模型：速度快、解释性强、适合作为基线
- 树模型：鲁棒、适合非线性关系、对特征缩放不敏感
- 集成模型：通常在表格数据上效果稳定
- 核方法：适合中小规模数据、非线性表达能力较强
- 朴素贝叶斯：适合某些稀疏特征和轻量级快速实验
- KNN：作为局部邻域方法的补充基线
- XGBoost / LightGBM：工业界高频使用，表格任务常见强基线

### 6.2.4 sklearn 搜索能力

首期以 `GridSearchCV` 为标准实现，原因如下：

- 行为稳定
- 参数空间清晰
- 结果结构标准化
- 易于和 `Pipeline` 结合
- 易于将每组实验结果落到 MLflow

后续可扩展：

- `RandomizedSearchCV`
- 自定义分层搜索

### 6.2.6 sklearn 可视化产物要求

`sklearn` 实验除了指标与模型，还需要产出并记录常见可视化结果：

- 分类任务：
  - confusion matrix
  - ROC 曲线
  - PR 曲线
  - 预测概率分布图

- 回归任务：
  - 预测值 vs 真实值图
  - 残差图
  - 误差分布图

### 6.2.5 sklearn 模块关键能力

- 自动构建预处理管线
- 按模型类型装配 `Pipeline`
- 按任务类型加载可用模型
- 执行 `GridSearchCV`
- 记录每折 CV 指标
- 输出最优参数
- 输出最优模型
- 输出预测结果
- 输出评估汇总
- 统一挂接 MLflow

---

## 6.3 torch 深度学习模块

### 6.3.1 模块目标

`torch/` 模块用于承载表格数据上的深度学习方法，重点不是追求方法堆砌，而是建立一套：

- 可训练
- 可验证
- 可早停
- 可记录
- 可扩展

的深度学习训练骨架。

### 6.3.2 首期纳入模型

- `MLP`
- `1D CNN`
- `TabNet`

### 6.3.3 模型设计要求

#### MLP

要求支持：

- 可配置层数
- 可配置隐藏层维度
- 可配置激活函数
- 可配置 dropout
- 可配置 batch size
- 可配置学习率
- 可配置 epoch
- 必须支持 `EarlyStopping`

#### 1D CNN

要求支持：

- 将表格特征重排后输入一维卷积结构
- 可配置卷积层数、通道数、卷积核大小
- 可配置池化策略
- 可配置全连接输出头
- 可配置训练参数

#### TabNet

要求支持：

- 明确写成独立模型路线
- 和自定义训练器解耦
- 可配置主要超参数
- 可记录训练过程和最佳验证结果

### 6.3.4 torch 搜索能力

`torch` 模型不强制直接套 `GridSearchCV`，原因如下：

- `GridSearchCV` 更适合 `sklearn estimator` 接口
- 深度学习训练过程需要显式控制 epoch、batch、optimizer、scheduler、early stopping
- 每次训练成本更高，参数搜索往往需要更精细的调度

因此首期建议：

- 由自定义训练循环负责单次训练
- 由自定义搜索器负责参数组合枚举与结果汇总
- 搜索结果统一写入 MLflow

后续如需要，也可以补一层 sklearn-style wrapper，但不作为首期重点。

### 6.3.5 torch 模块关键能力

- 数据集与 DataLoader 构建
- 模型构建器
- 优化器构建
- 损失函数选择
- 训练循环
- 验证循环
- EarlyStopping
- 最优权重保存
- 参数搜索
- 评估
- 推理
- 统一挂接 MLflow

### 6.3.6 torch 可视化产物要求

`torch` 实验除了最终指标与权重文件，还要记录：

- train loss 曲线
- valid loss 曲线
- 主指标随 epoch 变化曲线
- 最优 epoch 信息
- 分类或回归对应的结果图

---

## 6.4 MLflow 实验追踪模块

### 6.4.1 模块目标

`MLflow` 作为实验追踪中心，需要统一承接 `sklearn` 与 `torch` 两条主线的训练记录，确保实验可复盘。

### 6.4.2 记录范围

每次训练或搜索至少记录以下内容：

- 任务类型
- 数据集名称
- 目标列
- 特征列数量
- 模型类型
- 超参数
- CV 配置
- 训练指标
- 验证指标
- 测试指标（如有）
- 最佳参数
- 最佳分数
- 配置文件快照
- 特征清单
- 预测结果文件
- 模型文件

### 6.4.3 sklearn 与 torch 的统一记录规范

虽然两条主线实现不同，但在 MLflow 层建议统一元数据字段，例如：

- `framework = sklearn / torch`
- `task = classification / regression`
- `model_name`
- `run_type = baseline / grid_search / final_train`
- `dataset_name`
- `target_column`

这样做的价值是：

- 后续筛选实验更方便
- 横向对比不同模型更方便
- 后续做汇总报表更方便

### 6.4.4 MLflow 产物要求

至少记录以下 artifact：

- `config.yaml` 快照
- 模型参数 JSON / YAML
- 评估报告
- 预测结果 CSV
- 特征列清单
- confusion matrix / regression plots（可选）
- PCA / t-SNE / UMAP 结果图
- 相关性热力图、缺失值图、类别分布图等分析图
- 最优模型文件

### 6.4.5 MLflow 内部可视化能力

MLflow 在本项目中的角色不只是“记日志”，还需要承担实验可视化结果的统一展示入口。首期至少要能在 MLflow 中看到：

- 各模型 run 的参数对比
- 各模型 run 的指标对比
- 各模型 run 的 artifact 图表
- 降维实验图
- 分析实验图
- 最优模型文件与权重文件

---

## 7. config.yaml 设计要求

## 7.1 设计目标

单个实验目录下的 `config.yaml` 是该实验的唯一主配置入口，需要满足以下要求：

- 结构清晰
- 人工可读
- 可直接修改
- 能覆盖绝大多数训练控制需求
- 便于后续扩展

## 7.2 建议配置结构

```yaml
project:
  name: homemade_datarobot
  experiment_name: tabular_automl
  random_seed: 42

data:
  input_path: data/raw/train.csv
  file_type: csv
  target_column: target
  drop_columns: []
  task_type: classification
  test_size: 0.2
  valid_size: 0.1
  stratify: true

features:
  numeric_columns:
    - age
    - income
    - balance
  categorical_columns:
    - city
    - gender
    - segment
  onehot_columns:
    - city
    - gender

cv:
  enabled: true
  n_splits: 5
  shuffle: true
  stratified: true

metric:
  primary: roc_auc
  secondary:
    - accuracy
    - f1

preprocess:
  numeric_imputer: median
  categorical_imputer: most_frequent
  scaler: standard
  encoder: onehot

analysis:
  enabled: true
  pca:
    enabled: true
    n_components: 2
  tsne:
    enabled: true
    n_components: 2
    perplexity: 30
  umap:
    enabled: true
    n_components: 2
    n_neighbors: 15
    min_dist: 0.1

models:
  sklearn:
    enabled: true
    classification:
      - logistic_regression
      - svm
      - random_forest
      - extra_trees
      - gradient_boosting
      - hist_gradient_boosting
      - decision_tree
      - knn
      - gaussian_nb
      - xgboost
      - lightgbm
    regression:
      - linear_regression
      - ridge
      - lasso
      - elasticnet
      - svr
      - random_forest
      - extra_trees
      - gradient_boosting
      - hist_gradient_boosting
      - decision_tree
      - knn
      - xgboost
      - lightgbm

  torch:
    enabled: true
    classification:
      - mlp
      - cnn1d
      - tabnet
    regression:
      - mlp
      - cnn1d
      - tabnet

search:
  sklearn:
    method: grid_search
    n_jobs: -1
    refit: true
    verbose: 1
  torch:
    method: manual_grid

training:
  torch:
    epochs: 100
    batch_size: 256
    learning_rate: 0.001
    weight_decay: 0.0
    early_stopping:
      enabled: true
      patience: 10
      min_delta: 0.0

mlflow:
  tracking_uri: ./mlruns
  experiment_name: homemade_datarobot_experiment
  log_models: true
  log_artifacts: true
  exp_id: exp01
```

## 7.3 配置设计要求

PRD 中后续要约束实现时做到以下几点：

- 所有字段都要写清楚默认值
- 所有字段都要有中文注释
- `numeric_columns`、`categorical_columns`、`onehot_columns` 的优先级与适用条件要写清楚
- 错误配置要有明确报错
- 不允许悄悄忽略非法字段
- 不允许默认行为过于隐蔽
- 配置必须和所属 `storage/exp_000001` 这类实验目录绑定，不能跨实验复用时产生歧义

## 7.4 exp 作为基本管理单元

后续实现时必须把 `exp` 作为系统里的基本单位来看待：

- 一个 `exp` = 一套配置
- 一个 `exp` = 一套输入数据
- 一个 `exp` = 一套中间结果
- 一个 `exp` = 一套输出结果
- 一个 `exp` = 一组 MLflow runs

建议每次实验都带有明确的实验目录编号，例如：

- `exp_000001`
- `exp_000002`
- `exp_000123`

MLflow 侧需要明确把 `task_id` 记录为关键字段，便于 Django 与 MLflow 双向关联。

## 7.5 task 与主 run 的关系

后续实现时必须统一以下定义：

- 一个 `task` 表示一次完整任务综合体
- 一个 `task` 对应一个主 MLflow run
- 任务未完成时必须保留状态与保存点，以支持后续恢复执行
- `task_id` 采用严格递增的 6 位编号，历史编号不复用

这里的“任务综合体”包括：

- 上传的数据
- 任务配置
- 分析与训练过程
- 过程状态
- 中间产物
- 可下载结果
- 模型文件与权重文件

因此 `task` 不只是实验概念，而是业务与实验合并后的基本交付单元。

## 7.6 run_state.json 设计要求

每个 `task` 目录下必须有一个 `run_state.json`，并且它不能只记录最终状态，而必须记录“已经做完了什么”的可读信息，既方便程序恢复，也方便人类直接查看。

至少应包含：

- `task_id`
- `status`
- `resume_supported`
- `resume_count`
- `current_stage`
- `completed_steps`
- `pending_steps`
- `last_completed_model`
- `last_checkpoint_at`

建议同时增加分组后的实验完成状态，例如：

- 哪些 `analysis` 步骤已完成
- 哪些 `sklearn` 模型已完成
- 哪些 `torch` 模型已完成

例如可以采用类似结构：

```json
{
  "task_id": "task_000001",
  "status": "running",
  "resume_supported": true,
  "resume_count": 1,
  "current_stage": "sklearn_training",
  "last_completed_experiment": "sklearn.random_forest",
  "completed_experiments": {
    "analysis": [
      "pca"
    ],
    "sklearn": [
      "logistic_regression",
      "random_forest"
    ],
    "torch": []
  },
  "completed_steps": [
    "data_validation",
    "schema_loaded",
    "pca_completed",
    "sklearn.logistic_regression.completed",
    "sklearn.random_forest.completed"
  ],
  "pending_steps": [
    "sklearn.xgboost",
    "torch.mlp",
    "final_report"
  ],
  "last_checkpoint_at": "2026-06-30T15:10:00"
}
```

## 7.7 恢复执行要求

中断恢复必须作为正式能力写入设计：

- 如果任务异常崩溃，后续应支持继续执行
- 如果任务被人为终止，后续应支持从保存点恢复
- 恢复后仍然归属于同一个 `task`

建议恢复粒度如下：

- `sklearn`：按模型粒度恢复，已完成模型不重复跑
- `torch`：按 checkpoint 恢复到最近可用 epoch
- `analysis`：按步骤粒度恢复，例如 `pca_completed`、`umap_completed`

## 7.8 nested run 命名规则

由于一个 `task` 对应的是一整套分析、训练、可视化流程，而不是单一模型，因此主 run 下的子步骤统一使用 nested run 记录。

命名规则固定为：

- `<domain>.<name>`

其中 `domain` 只允许以下三类：

- `analysis`
- `sklearn`
- `torch`

首期固定命名如下：

- 分析类：
  - `analysis.pca`
  - `analysis.tsne`
  - `analysis.umap`

- sklearn 类：
  - `sklearn.logistic_regression`
  - `sklearn.random_forest`
  - `sklearn.xgboost`

- torch 类：
  - `torch.mlp`
  - `torch.cnn1d`
  - `torch.tabnet`

后续新增实验项时，也必须遵守同样的命名格式，避免 run 命名漂移。

## 7.9 存储与落盘策略

本项目需要在“可复现”和“减少不必要 SSD 写入”之间保持平衡，因此默认不追求保存所有中间产物，而是按类型控制保存范围。

总体原则如下：

- 能用状态和参数复现的内容，尽量不重复落盘
- 不保存体积大的重复中间结果
- 不保存非必要的多份模型副本
- 但不能因为过度节省，导致最终结果不可分析、不可交付、不可恢复

### 7.9.1 analysis 类产物

`PCA`、`t-SNE`、`UMAP` 需要完整保留分析结果，不能只保留一张图。

每个分析项至少保留：

- 2D 坐标结果
- 对应散点图
- 关键参数配置

例如：

- `pca_2d.csv`
- `tsne_2d.csv`
- `umap_2d.csv`
- 对应的图像文件

### 7.9.2 sklearn 类产物

`sklearn` 模型默认只保留必要的最终产物，不保存大量中间副本。

每个模型至少保留：

- 最终指标
- 最终预测结果
- 最终可视化结果
- 最佳参数
- 模型完成状态

如有必要，可额外保留一个最终模型文件，但默认不保留大量中间模型副本。

### 7.9.3 torch 类产物

`torch` 模型需要兼顾恢复能力与写盘控制。

每个模型至少保留：

- 最终指标
- 最终预测结果
- 最终可视化结果
- 最优权重文件
- 恢复执行所需的最小 checkpoint

默认不保留过多训练中间文件，也不保留不必要的多份权重副本。

---

## 8. 模型搜索与调度设计

## 8.1 sklearn 搜索逻辑

标准流程如下：

1. 根据任务类型加载候选模型。
2. 根据模型类型加载对应参数网格。
3. 构建预处理器。
4. 组装 `Pipeline`。
5. 执行 `GridSearchCV`。
6. 记录全部搜索结果到 MLflow。
7. 输出最优模型及最优参数。

## 8.2 torch 搜索逻辑

标准流程如下：

1. 根据配置构建候选参数组合。
2. 对每个参数组合执行独立训练。
3. 每次训练都记录训练曲线和验证指标。
4. 根据主指标选出最优参数组合。
5. 保存最佳模型权重。
6. 将结果统一写入 MLflow。

## 8.3 搜索空间设计原则

- 首期参数空间不要过大
- 优先覆盖高价值超参数
- 避免一开始把训练成本拉爆
- 支持按模型单独配置参数网格

---

## 9. 评估体系设计

### 9.1 分类指标

可支持：

- `accuracy`
- `precision`
- `recall`
- `f1`
- `roc_auc`
- `log_loss`

### 9.2 回归指标

可支持：

- `rmse`
- `mae`
- `mse`
- `r2`

### 9.3 评估产物

应支持输出：

- 总评估汇总表
- 单模型评估报告
- 测试集预测结果
- 分类混淆矩阵
- 回归误差分析结果

---

## 10. 代码实现规范

### 10.1 注释规范

本项目的代码注释要求必须高于普通项目，必须强调以下几类注释：

- 模块级注释：说明该模块负责什么、和其他模块的边界是什么
- 类注释：说明该类解决什么问题、输入输出是什么
- 函数注释：说明参数含义、返回值含义、调用时机
- 关键逻辑注释：说明为什么这样写
- 长函数内分段注释：帮助读者理解处理链条

### 10.2 命名规范

- 文件名尽量表达职责
- 模型注册表统一命名
- 搜索器、训练器、评估器等命名清晰区分
- 不使用含义模糊的缩写

### 10.3 错误处理规范

- 配置错误要明确报错
- 数据列缺失要明确报错
- 模型与任务类型不匹配要明确报错
- 指标与任务类型不匹配要明确报错
- 训练失败要记录失败原因

---

## 11. 分阶段开发计划

### 阶段一：基础骨架

目标：

- 建立目录结构
- 建立 `config.yaml`
- 打通数据读取
- 打通 MLflow 基础记录

### 阶段二：sklearn 主线

目标：

- 接入常见分类模型
- 接入常见回归模型
- 接入 `Pipeline`
- 接入 `GridSearchCV`
- 输出统一评估结果

### 阶段三：torch 主线

目标：

- 接入 `MLP`
- 接入 `1D CNN`
- 接入 `TabNet`
- 完成 `EarlyStopping`
- 完成基础参数搜索

### 阶段四：统一调度层

目标：

- 通过统一入口调度 `sklearn` 和 `torch`
- 支持按配置启用或禁用模型
- 支持最终最优模型汇总

### 阶段五：增强能力

可选增强项：

- 特征重要性输出
- SHAP 解释
- 更灵活的搜索空间配置
- 模型集成
- 排行榜报告

---

## 12. 风险与注意事项

### 12.1 sklearn 风险

- 某些模型对预处理非常敏感
- 不同模型对稀疏矩阵支持不同
- `Naive Bayes` 对输入分布有要求
- `SVM` 在大数据上训练成本可能较高

### 12.2 torch 风险

- 表格数据上深度学习未必一定优于树模型
- 超参数敏感度更高
- 训练波动更明显
- 早停策略对最终结果影响较大

### 12.3 工程风险

- 如果配置设计不清楚，后续会快速失控
- 如果模型注册与搜索空间绑定方式不清楚，扩展成本会升高
- 如果 MLflow 记录字段不统一，后期实验对比会很难做

---

## 13. 验收标准

当以下条件满足时，可认为首版 PRD 对应系统达标：

- 可以通过单个 `exp` 目录下的 `config.yaml` 驱动一次完整实验
- `sklearn` 路线可完成多模型训练与 `GridSearchCV`
- `torch` 路线可完成 `MLP` / `1D CNN` / `TabNet` 训练
- `MLP` 明确支持 `EarlyStopping`
- 所有实验都可在 MLflow 中查看
- 数据、配置、输出按 `exp` 独立隔离
- 代码与配置具备丰富中文注释

---

## 14. 结论

本项目的本质不是复刻一个庞大的 DataRobot 平台，而是先做一个“可控型、结构清晰、实验可追踪”的本地 AutoML 原型。

其核心思想是：

- 用 `sklearn` 承担稳定、透明的传统机器学习主线
- 用 `torch` 承担可扩展的深度学习主线
- 用单个 `exp` 目录下的 `config.yaml` 管理实验入口
- 用 `MLflow` 统一管理实验
- 用大量中文注释确保项目长期可维护

这份 PRD 的目标，是为后续代码落地提供明确边界、统一术语和实现路线，避免项目一开始就陷入“自动化太多、控制太少、报错难查、结构混乱”的问题。

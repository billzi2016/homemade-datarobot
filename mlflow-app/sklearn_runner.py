"""
sklearn 执行模块。

这里单独承接传统机器学习主线，避免 run_task.py 继续膨胀。
当前首版重点是把已经跑通的 sklearn 链路独立出来：
- 特征列解析
- 预处理
- 模型注册
- 训练与评估
- 最终结果落盘
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlflow
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    RocCurveDisplay,
    r2_score,
    roc_auc_score,
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from task_runtime import (
    TaskPaths,
    is_experiment_completed,
    mark_experiment_completed,
    save_run_state,
)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """把字典以 JSON 形式落盘。"""

    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def save_dataframe(path: Path, df: pd.DataFrame) -> None:
    """统一保存 DataFrame。"""

    df.to_csv(path, index=False)


def save_classification_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    model_name: str,
    paths: TaskPaths,
) -> None:
    """保存 sklearn 分类图，并由调用方统一记录到 MLflow。"""

    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax_cm, colorbar=False)
    ax_cm.set_title(f"{model_name} Confusion Matrix")
    fig_cm.tight_layout()
    cm_path = paths.plots_dir / f"{model_name}_confusion_matrix.png"
    fig_cm.savefig(cm_path, dpi=160)
    plt.close(fig_cm)

    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == 2:
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_true, y_prob[:, 1], ax=ax_roc)
        ax_roc.set_title(f"{model_name} ROC Curve")
        fig_roc.tight_layout()
        roc_path = paths.plots_dir / f"{model_name}_roc_curve.png"
        fig_roc.savefig(roc_path, dpi=160)
        plt.close(fig_roc)


def infer_feature_columns(config: Dict[str, Any], df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    推断或读取特征列。

    规则：
    1. 如果配置里显式给了 numeric_columns / categorical_columns，就直接使用。
    2. 否则基于 pandas dtype 做保守推断。
    """

    target = config["data"]["target_column"]
    drop_columns = set(config["data"].get("drop_columns", []))
    usable_columns = [col for col in df.columns if col != target and col not in drop_columns]
    features_cfg = config.get("features", {})

    numeric_columns = list(features_cfg.get("numeric_columns") or [])
    categorical_columns = list(features_cfg.get("categorical_columns") or [])

    if not numeric_columns and not categorical_columns:
        inferred_numeric = df[usable_columns].select_dtypes(include=[np.number]).columns.tolist()
        inferred_categorical = [col for col in usable_columns if col not in inferred_numeric]
        numeric_columns = inferred_numeric
        categorical_columns = inferred_categorical

    return numeric_columns, categorical_columns


def build_preprocessor(
    config: Dict[str, Any],
    numeric_columns: List[str],
    categorical_columns: List[str],
    model_name: str,
) -> ColumnTransformer:
    """构造 sklearn 预处理器。"""

    preprocess_cfg = config.get("preprocess", {})
    numeric_imputer = preprocess_cfg.get("numeric_imputer", "median")
    categorical_imputer = preprocess_cfg.get("categorical_imputer", "most_frequent")
    scaler_name = preprocess_cfg.get("scaler", "standard")

    numeric_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy=numeric_imputer))]
    nb_models = {"bernoulli_nb", "multinomial_nb"}
    if model_name in nb_models:
        numeric_steps.append(("scaler", MinMaxScaler()))
    elif scaler_name == "standard":
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy=categorical_imputer)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_columns),
            ("cat", Pipeline(categorical_steps), categorical_columns),
        ],
        remainder="drop",
    )


def classification_model_registry(random_seed: int) -> Dict[str, Any]:
    """首版分类模型注册表。"""

    from xgboost import XGBClassifier

    registry: Dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_seed),
        "svm": SVC(probability=True, random_state=random_seed),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=random_seed, n_jobs=1
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=200, random_state=random_seed, n_jobs=1
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=random_seed),
        "decision_tree": DecisionTreeClassifier(random_state=random_seed),
        "knn": KNeighborsClassifier(),
        "gaussian_nb": GaussianNB(),
        "bernoulli_nb": BernoulliNB(),
        "multinomial_nb": MultinomialNB(),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_seed,
            eval_metric="mlogloss",
            n_jobs=1,
        ),
    }
    try:
        from lightgbm import LGBMClassifier

        registry["lightgbm"] = LGBMClassifier(
            random_state=random_seed,
            n_estimators=200,
            learning_rate=0.05,
            verbosity=-1,
        )
    except Exception:
        pass
    return registry


def regression_model_registry(random_seed: int) -> Dict[str, Any]:
    """首版回归模型注册表。"""

    from xgboost import XGBRegressor

    registry: Dict[str, Any] = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(random_state=random_seed),
        "lasso": Lasso(random_state=random_seed),
        "elasticnet": ElasticNet(random_state=random_seed),
        "svr": SVR(),
        "random_forest": RandomForestRegressor(
            n_estimators=200, random_state=random_seed, n_jobs=1
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=200, random_state=random_seed, n_jobs=1
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_seed),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=random_seed),
        "decision_tree": DecisionTreeRegressor(random_state=random_seed),
        "knn": KNeighborsRegressor(),
        "xgboost": XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_seed,
            n_jobs=1,
        ),
    }
    try:
        from lightgbm import LGBMRegressor

        registry["lightgbm"] = LGBMRegressor(
            random_state=random_seed,
            n_estimators=200,
            learning_rate=0.05,
            verbosity=-1,
        )
    except Exception:
        pass
    return registry


def build_model_registry(task_type: str, random_seed: int) -> Dict[str, Any]:
    """根据任务类型返回可用模型注册表。"""

    if task_type == "classification":
        return classification_model_registry(random_seed)
    if task_type == "regression":
        return regression_model_registry(random_seed)
    raise ValueError(f"不支持的 task_type: {task_type}")


def detect_imbalance(y: pd.Series | np.ndarray) -> bool:
    """
    判断当前分类标签是否明显不平衡。

    当前规则保持简单可解释：
    - 如果最小类样本数为 0，直接视为异常不平衡
    - 否则当最大类 / 最小类 >= 1.5 时，认为值得主动平衡
    """

    value_counts = pd.Series(y).value_counts()
    if value_counts.empty:
        return False
    min_count = int(value_counts.min())
    if min_count == 0:
        return True
    return float(value_counts.max() / min_count) >= 1.5


def oversample_training_data(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    对训练集做最朴素但最通用的随机过采样。

    这样做的目的很明确：
    - 不依赖 imbalanced-learn 额外包
    - 让几乎所有 sklearn 分类模型都能“吃到”平衡后的训练数据
    - 即便模型本身不支持 class_weight，也能获得统一的类别平衡能力
    """

    joined = x_train.copy()
    joined["__target__"] = y_train.values
    grouped = joined.groupby("__target__", group_keys=False)
    max_count = int(grouped.size().max())
    balanced_parts: List[pd.DataFrame] = []
    for _, group in grouped:
        replace = len(group) < max_count
        balanced_parts.append(
            group.sample(
                n=max_count,
                replace=replace,
                random_state=random_seed,
            )
        )
    balanced_df = (
        pd.concat(balanced_parts, axis=0)
        .sample(frac=1.0, random_state=random_seed)
        .reset_index(drop=True)
    )
    y_balanced = balanced_df.pop("__target__")
    return balanced_df, y_balanced


def apply_class_balance_to_model(
    model_name: str,
    model: Any,
    y_train: pd.Series,
) -> Any:
    """
    尽量把类别平衡信息下沉到模型参数层。

    这里不只依赖过采样，还尽可能启用模型原生能力：
    - 大多数 sklearn 分类器支持 class_weight="balanced"
    - xgboost 二分类可以设置 scale_pos_weight
    - lightgbm 支持 class_weight="balanced"
    """

    if hasattr(model, "get_params"):
        params = model.get_params()
        if "class_weight" in params:
            model.set_params(class_weight="balanced")

    if model_name == "xgboost":
        class_counts = pd.Series(y_train).value_counts().sort_index()
        if len(class_counts) == 2:
            negative = int(class_counts.get(0, class_counts.iloc[0]))
            positive = int(class_counts.get(1, class_counts.iloc[-1]))
            if positive > 0:
                model.set_params(scale_pos_weight=float(negative / positive))

    return model


def resolve_search_method(
    config: Dict[str, Any],
    model_name: str,
    task_type: str,
    n_rows: int,
    domain: str,
) -> str:
    """
    决定当前模型实际采用的搜索策略。

    设计原则：
    - 用户若显式指定，就尊重显式指定
    - 默认 `auto` 不返回 `none`
    - 高成本模型优先 optuna
    - 小数据保守走 grid，中等数据走 halving，大数据走 optuna
    """

    explicit_method = str(config.get("search", {}).get("method", "auto")).lower()
    if explicit_method in {"grid", "halving_grid", "optuna"}:
        return explicit_method

    if domain == "torch":
        return "optuna"

    high_cost_models = {
        "xgboost",
        "lightgbm",
        "svm",
        "svr",
    }
    if model_name in high_cost_models:
        return "optuna"
    if n_rows <= 5000:
        return "grid"
    if n_rows <= 50000:
        return "halving_grid"
    return "optuna"


def resolve_scoring(task_type: str) -> str:
    """给搜索阶段提供一个稳定的默认 scoring。"""

    return "accuracy" if task_type == "classification" else "neg_root_mean_squared_error"


def build_cv_splitter(config: Dict[str, Any], task_type: str):
    """按任务类型构造 CV 切分器。"""

    n_splits = int(config.get("cv", {}).get("n_splits", 3))
    random_seed = int(config["project"]["random_seed"])
    if task_type == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)


def build_search_grid(task_type: str, model_name: str) -> Dict[str, List[Any]]:
    """为常见 sklearn 模型提供一套保守但实用的搜索网格。"""

    grids: Dict[str, Dict[str, List[Any]]] = {
        "classification": {
            "logistic_regression": {"C": [0.1, 1.0, 10.0]},
            "svm": {"C": [0.5, 1.0, 2.0], "gamma": ["scale", "auto"]},
            "random_forest": {
                "n_estimators": [120, 200],
                "max_depth": [None, 6, 12],
                "min_samples_leaf": [1, 2, 4],
            },
            "extra_trees": {
                "n_estimators": [120, 200],
                "max_depth": [None, 6, 12],
                "min_samples_leaf": [1, 2, 4],
            },
            "gradient_boosting": {
                "n_estimators": [100, 180],
                "learning_rate": [0.03, 0.1],
                "max_depth": [2, 3],
            },
            "hist_gradient_boosting": {
                "learning_rate": [0.03, 0.1],
                "max_depth": [None, 6, 12],
                "max_leaf_nodes": [15, 31],
            },
            "decision_tree": {
                "max_depth": [None, 4, 8, 12],
                "min_samples_leaf": [1, 2, 4],
            },
            "knn": {"n_neighbors": [3, 5, 9], "weights": ["uniform", "distance"]},
            "gaussian_nb": {"var_smoothing": [1e-9, 1e-8, 1e-7]},
            "bernoulli_nb": {"alpha": [0.1, 0.5, 1.0]},
            "multinomial_nb": {"alpha": [0.1, 0.5, 1.0]},
            "xgboost": {
                "n_estimators": [120, 220],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.03, 0.1],
            },
            "lightgbm": {
                "n_estimators": [120, 220],
                "num_leaves": [15, 31, 63],
                "learning_rate": [0.03, 0.1],
            },
        },
        "regression": {
            "linear_regression": {},
            "ridge": {"alpha": [0.1, 1.0, 10.0]},
            "lasso": {"alpha": [0.001, 0.01, 0.1]},
            "elasticnet": {"alpha": [0.001, 0.01, 0.1], "l1_ratio": [0.2, 0.5, 0.8]},
            "svr": {"C": [0.5, 1.0, 2.0], "gamma": ["scale", "auto"]},
            "random_forest": {
                "n_estimators": [120, 200],
                "max_depth": [None, 6, 12],
                "min_samples_leaf": [1, 2, 4],
            },
            "extra_trees": {
                "n_estimators": [120, 200],
                "max_depth": [None, 6, 12],
                "min_samples_leaf": [1, 2, 4],
            },
            "gradient_boosting": {
                "n_estimators": [100, 180],
                "learning_rate": [0.03, 0.1],
                "max_depth": [2, 3],
            },
            "hist_gradient_boosting": {
                "learning_rate": [0.03, 0.1],
                "max_depth": [None, 6, 12],
                "max_leaf_nodes": [15, 31],
            },
            "decision_tree": {
                "max_depth": [None, 4, 8, 12],
                "min_samples_leaf": [1, 2, 4],
            },
            "knn": {"n_neighbors": [3, 5, 9], "weights": ["uniform", "distance"]},
            "xgboost": {
                "n_estimators": [120, 220],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.03, 0.1],
            },
            "lightgbm": {
                "n_estimators": [120, 220],
                "num_leaves": [15, 31, 63],
                "learning_rate": [0.03, 0.1],
            },
        },
    }
    return grids.get(task_type, {}).get(model_name, {})


def suggest_model_params(trial: optuna.Trial, task_type: str, model_name: str) -> Dict[str, Any]:
    """为 Optuna 提供各模型的参数空间。"""

    if model_name == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-2, 10.0, log=True)}
    if model_name in {"svm", "svr"}:
        return {
            "C": trial.suggest_float("C", 0.1, 10.0, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
    if model_name in {"random_forest", "extra_trees"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, 260, step=20),
            "max_depth": trial.suggest_categorical("max_depth", [None, 4, 8, 12, 16]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
        }
    if model_name in {"gradient_boosting", "hist_gradient_boosting"}:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_categorical("max_depth", [None, 3, 6, 10]),
            "max_leaf_nodes": trial.suggest_categorical("max_leaf_nodes", [15, 31, 63]),
        } if model_name == "hist_gradient_boosting" else {
            "n_estimators": trial.suggest_int("n_estimators", 80, 220, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
        }
    if model_name == "decision_tree":
        return {
            "max_depth": trial.suggest_categorical("max_depth", [None, 4, 8, 12, 16]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
        }
    if model_name == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 15, step=2),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        }
    if model_name == "gaussian_nb":
        return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-10, 1e-7, log=True)}
    if model_name in {"bernoulli_nb", "multinomial_nb"}:
        return {"alpha": trial.suggest_float("alpha", 1e-2, 2.0, log=True)}
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, 260, step=20),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, 260, step=20),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, step=8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        }
    if model_name == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-2, 10.0, log=True)}
    if model_name == "lasso":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 1.0, log=True)}
    if model_name == "elasticnet":
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 1.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 0.9),
        }
    return {}


def run_search_for_pipeline(
    config: Dict[str, Any],
    task_type: str,
    model_name: str,
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    search_method: str,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """统一执行 sklearn 搜索。"""

    scoring = resolve_scoring(task_type)
    cv_splitter = build_cv_splitter(config, task_type)

    if search_method == "grid":
        raw_grid = build_search_grid(task_type, model_name)
        search_grid = {f"model__{key}": value for key, value in raw_grid.items()}
        if not search_grid:
            pipeline.fit(x_train, y_train)
            return pipeline, {"search_method": "grid", "search_note": "empty_grid_direct_fit"}
        searcher = GridSearchCV(
            estimator=pipeline,
            param_grid=search_grid,
            scoring=scoring,
            cv=cv_splitter,
            n_jobs=1,
        )
        searcher.fit(x_train, y_train)
        return searcher.best_estimator_, {
            "search_method": "grid",
            "best_params": searcher.best_params_,
            "best_score": float(searcher.best_score_),
        }

    if search_method == "halving_grid":
        raw_grid = build_search_grid(task_type, model_name)
        search_grid = {f"model__{key}": value for key, value in raw_grid.items()}
        if not search_grid:
            pipeline.fit(x_train, y_train)
            return pipeline, {"search_method": "halving_grid", "search_note": "empty_grid_direct_fit"}
        searcher = HalvingGridSearchCV(
            estimator=pipeline,
            param_grid=search_grid,
            scoring=scoring,
            cv=cv_splitter,
            n_jobs=1,
            factor=2,
        )
        searcher.fit(x_train, y_train)
        return searcher.best_estimator_, {
            "search_method": "halving_grid",
            "best_params": searcher.best_params_,
            "best_score": float(searcher.best_score_),
        }

    n_trials = int(config.get("search", {}).get("n_trials", 12))
    direction = "maximize"

    def objective(trial: optuna.Trial) -> float:
        params = suggest_model_params(trial, task_type, model_name)
        candidate = clone(pipeline)
        if params:
            candidate.set_params(**{f"model__{key}": value for key, value in params.items()})
        scores = cross_val_score(
            candidate,
            x_train,
            y_train,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=1,
        )
        return float(np.mean(scores))

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_pipeline = clone(pipeline)
    best_params = suggest_model_params(study.best_trial, task_type, model_name)
    if best_params:
        best_pipeline.set_params(**{f"model__{key}": value for key, value in best_params.items()})
    best_pipeline.fit(x_train, y_train)
    return best_pipeline, {
        "search_method": "optuna",
        "best_params": best_params,
        "best_score": float(study.best_value),
        "n_trials": n_trials,
    }


def build_xgboost_from_params(
    task_type: str,
    random_seed: int,
    params: Dict[str, Any],
):
    """根据搜索结果构造 xgboost 模型。"""

    if task_type == "classification":
        from xgboost import XGBClassifier

        return XGBClassifier(
            random_state=random_seed,
            eval_metric="mlogloss",
            n_jobs=1,
            **params,
        )

    from xgboost import XGBRegressor

    return XGBRegressor(random_state=random_seed, n_jobs=1, **params)


def tune_xgboost_with_optuna(
    config: Dict[str, Any],
    task_type: str,
    random_seed: int,
    preprocessor: ColumnTransformer,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> Dict[str, Any]:
    """
    使用 Optuna 搜索 xgboost 参数。

    当前只做一层务实实现：
    - 不追求覆盖所有参数
    - 优先搜索最影响效果的核心参数
    """

    n_trials = int(config.get("search", {}).get("n_trials", 12))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 260),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        model = build_xgboost_from_params(task_type, random_seed, params)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_valid)

        if task_type == "classification":
            return float(accuracy_score(y_valid, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_valid, y_pred)))
        return -rmse

    direction = "maximize" if task_type == "classification" else "maximize"
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {
        "best_params": study.best_params,
        "best_value": float(study.best_value),
        "n_trials": n_trials,
    }


def compute_metrics(
    task_type: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None
) -> Dict[str, float]:
    """根据任务类型计算核心指标。"""

    if task_type == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        }
        if y_prob is not None:
            unique_classes = np.unique(y_true)
            try:
                if len(unique_classes) == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
                else:
                    metrics["roc_auc_ovr"] = float(
                        roc_auc_score(y_true, y_prob, multi_class="ovr")
                    )
            except Exception:
                pass
        return metrics

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def split_data(
    config: Dict[str, Any], df: pd.DataFrame, feature_columns: List[str], target_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """按配置切分训练集与测试集。"""

    task_type = config["data"]["task_type"]
    stratify = (
        df[target_column]
        if task_type == "classification" and config["data"].get("stratify", True)
        else None
    )
    return train_test_split(
        df[feature_columns],
        df[target_column],
        test_size=config["data"].get("test_size", 0.2),
        random_state=config["project"]["random_seed"],
        stratify=stratify,
    )


def run_sklearn(
    config: Dict[str, Any],
    df: pd.DataFrame,
    paths: TaskPaths,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """执行 sklearn 主线实验。"""

    task_type = config["data"]["task_type"]
    target_column = config["data"]["target_column"]
    random_seed = config["project"]["random_seed"]
    numeric_columns, categorical_columns = infer_feature_columns(config, df)
    feature_columns = numeric_columns + categorical_columns

    if task_type == "classification":
        label_encoder = LabelEncoder()
        df = df.copy()
        df[target_column] = label_encoder.fit_transform(df[target_column].astype(str))

    x_train, x_test, y_train, y_test = split_data(config, df, feature_columns, target_column)
    registry = build_model_registry(task_type, random_seed)

    configured_models = config["models"]["sklearn"][task_type]
    summary_rows: List[Dict[str, Any]] = []
    imbalance_cfg = config.get("imbalance", {})
    imbalance_enabled = bool(imbalance_cfg.get("enabled", task_type == "classification"))

    if task_type == "classification" and imbalance_enabled and detect_imbalance(y_train):
        x_train, y_train = oversample_training_data(
            x_train=x_train,
            y_train=y_train,
            random_seed=random_seed,
        )

    for model_name in configured_models:
        if model_name not in registry or is_experiment_completed(state, "sklearn", model_name):
            continue

        state["current_stage"] = f"sklearn.{model_name}"
        save_run_state(paths, state)

        model_instance = clone(registry[model_name])
        if task_type == "classification" and imbalance_enabled:
            model_instance = apply_class_balance_to_model(
                model_name=model_name,
                model=model_instance,
                y_train=y_train,
            )
        search_method = resolve_search_method(
            config=config,
            model_name=model_name,
            task_type=task_type,
            n_rows=len(df),
            domain="sklearn",
        )

        preprocessor = build_preprocessor(
            config=config,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            model_name=model_name,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model_instance),
            ]
        )

        with mlflow.start_run(run_name=f"sklearn.{model_name}", nested=True):
            trained_pipeline, search_info = run_search_for_pipeline(
                config=config,
                task_type=task_type,
                model_name=model_name,
                pipeline=pipeline,
                x_train=x_train,
                y_train=y_train,
                search_method=search_method,
            )
            mlflow.log_param("search_method", search_info["search_method"])
            mlflow.log_param("imbalance_enabled", imbalance_enabled)
            mlflow.log_param("stratify_enabled", bool(config["data"].get("stratify", True)))
            if "n_trials" in search_info:
                mlflow.log_param("optuna_n_trials", search_info["n_trials"])
            if "best_params" in search_info:
                for key, value in search_info["best_params"].items():
                    mlflow.log_param(f"best_{key}", value)
            if "best_score" in search_info:
                mlflow.log_metric("search_best_score", search_info["best_score"])
            if "search_note" in search_info:
                mlflow.log_param("search_note", search_info["search_note"])

            y_pred = trained_pipeline.predict(x_test)
            y_prob = (
                trained_pipeline.predict_proba(x_test)
                if hasattr(trained_pipeline, "predict_proba")
                else None
            )

            metrics = compute_metrics(task_type, y_test.to_numpy(), y_pred, y_prob)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            prediction_df = x_test.copy()
            prediction_df["y_true"] = y_test.values
            prediction_df["y_pred"] = y_pred
            prediction_path = paths.predictions_dir / f"{model_name}_predictions.csv"
            save_dataframe(prediction_path, prediction_df)
            mlflow.log_artifact(str(prediction_path), artifact_path="predictions")

            metric_payload = {
                "task_id": config["task"]["task_id"],
                "model_name": model_name,
                "task_type": task_type,
                "metrics": metrics,
            }
            metrics_path = paths.metrics_dir / f"{model_name}_metrics.json"
            save_json(metrics_path, metric_payload)
            mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

            if task_type == "classification":
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                report_path = paths.metrics_dir / f"{model_name}_classification_report.json"
                save_json(report_path, report)
                mlflow.log_artifact(str(report_path), artifact_path="metrics")
                save_classification_plots(
                    y_true=y_test.to_numpy(),
                    y_pred=y_pred,
                    y_prob=y_prob,
                    model_name=model_name,
                    paths=paths,
                )
                mlflow.log_artifact(
                    str(paths.plots_dir / f"{model_name}_confusion_matrix.png"),
                    artifact_path="plots",
                )
                roc_path = paths.plots_dir / f"{model_name}_roc_curve.png"
                if roc_path.exists():
                    mlflow.log_artifact(str(roc_path), artifact_path="plots")

            summary_rows.append({"model_name": model_name, **metrics})
            mark_experiment_completed(state, "sklearn", model_name)
            save_run_state(paths, state)

    if summary_rows:
        primary_sort_key = list(summary_rows[0].keys())[1]
        summary_df = pd.DataFrame(summary_rows).sort_values(by=primary_sort_key, ascending=False)
        summary_path = paths.metrics_dir / "sklearn_summary.csv"
        save_dataframe(summary_path, summary_df)
        best_row = summary_df.iloc[0].to_dict()
        return {
            "summary_path": summary_path,
            "summary_rows": summary_rows,
            "best_model_name": best_row["model_name"],
            "primary_metric_name": primary_sort_key,
            "primary_metric_value": float(best_row[primary_sort_key]),
        }

    return {}

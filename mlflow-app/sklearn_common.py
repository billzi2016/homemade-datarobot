"""
sklearn 公共支持模块。

这里集中放置与 sklearn 链路通用、但不直接属于 run_sklearn 主流程的能力：
- 文件保存
- 预处理
- 模型注册
- 搜索策略
- 指标计算
- 数据切分与类别平衡
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from task_runtime import TaskPaths


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def save_dataframe(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def save_classification_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    model_name: str,
    paths: TaskPaths,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax_cm, colorbar=False)
    ax_cm.set_title(f"{model_name} Confusion Matrix")
    fig_cm.tight_layout()
    fig_cm.savefig(paths.plots_dir / f"{model_name}_confusion_matrix.png", dpi=160)
    plt.close(fig_cm)

    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == 2:
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_true, y_prob[:, 1], ax=ax_roc)
        ax_roc.set_title(f"{model_name} ROC Curve")
        fig_roc.tight_layout()
        fig_roc.savefig(paths.plots_dir / f"{model_name}_roc_curve.png", dpi=160)
        plt.close(fig_roc)


def infer_feature_columns(config: Dict[str, Any], df: pd.DataFrame) -> Tuple[List[str], List[str]]:
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
    preprocess_cfg = config.get("preprocess", {})
    numeric_imputer = preprocess_cfg.get("numeric_imputer", "median")
    categorical_imputer = preprocess_cfg.get("categorical_imputer", "most_frequent")
    scaler_name = preprocess_cfg.get("scaler", "standard")

    numeric_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy=numeric_imputer))]
    if model_name in {"bernoulli_nb", "multinomial_nb"}:
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
    from xgboost import XGBClassifier

    registry: Dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_seed),
        "svm": SVC(probability=True, random_state=random_seed),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=random_seed, n_jobs=1),
        "extra_trees": ExtraTreesClassifier(n_estimators=200, random_state=random_seed, n_jobs=1),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=random_seed, max_iter=40),
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
            num_threads=1,
        )
    except Exception:
        pass
    return registry


def regression_model_registry(random_seed: int) -> Dict[str, Any]:
    from xgboost import XGBRegressor

    registry: Dict[str, Any] = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(random_state=random_seed),
        "lasso": Lasso(random_state=random_seed),
        "elasticnet": ElasticNet(random_state=random_seed),
        "svr": SVR(),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=random_seed, n_jobs=1),
        "extra_trees": ExtraTreesRegressor(n_estimators=200, random_state=random_seed, n_jobs=1),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_seed),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=random_seed, max_iter=40),
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
            num_threads=1,
        )
    except Exception:
        pass
    return registry


def build_model_registry(task_type: str, random_seed: int) -> Dict[str, Any]:
    if task_type == "classification":
        return classification_model_registry(random_seed)
    if task_type == "regression":
        return regression_model_registry(random_seed)
    raise ValueError(f"不支持的 task_type: {task_type}")


def detect_imbalance(y: pd.Series | np.ndarray) -> bool:
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
    joined = x_train.copy()
    joined["__target__"] = y_train.values
    grouped = joined.groupby("__target__", group_keys=False)
    max_count = int(grouped.size().max())
    balanced_parts: List[pd.DataFrame] = []
    for _, group in grouped:
        balanced_parts.append(group.sample(n=max_count, replace=len(group) < max_count, random_state=random_seed))
    balanced_df = pd.concat(balanced_parts, axis=0).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    y_balanced = balanced_df.pop("__target__")
    return balanced_df, y_balanced


def apply_class_balance_to_model(model_name: str, model: Any, y_train: pd.Series) -> Any:
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


def resolve_search_method(config: Dict[str, Any], model_name: str, task_type: str, n_rows: int, domain: str) -> str:
    explicit_method = str(config.get("search", {}).get("method", "auto")).lower()
    if explicit_method in {"grid", "halving_grid", "optuna"}:
        return explicit_method
    if domain == "torch":
        return "optuna"
    if model_name in {"xgboost", "lightgbm", "svm", "svr"}:
        return "optuna"
    if n_rows <= 5000:
        return "grid"
    if n_rows <= 50000:
        return "halving_grid"
    return "optuna"


def resolve_scoring(task_type: str) -> str:
    return "accuracy" if task_type == "classification" else "neg_root_mean_squared_error"


def build_cv_splitter(config: Dict[str, Any], task_type: str):
    n_splits = int(config.get("cv", {}).get("n_splits", 3))
    random_seed = int(config["project"]["random_seed"])
    if task_type == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)


def build_search_grid(task_type: str, model_name: str) -> Dict[str, List[Any]]:
    grids: Dict[str, Dict[str, List[Any]]] = {
        "classification": {
            "logistic_regression": {"C": [0.1, 1.0, 10.0]},
            "svm": {"C": [0.5, 1.0, 2.0], "gamma": ["scale", "auto"]},
            "random_forest": {"n_estimators": [120, 200], "max_depth": [None, 6, 12], "min_samples_leaf": [1, 2, 4]},
            "extra_trees": {"n_estimators": [120, 200], "max_depth": [None, 6, 12], "min_samples_leaf": [1, 2, 4]},
            "gradient_boosting": {"n_estimators": [100, 180], "learning_rate": [0.03, 0.1], "max_depth": [2, 3]},
            "hist_gradient_boosting": {"learning_rate": [0.05, 0.1], "max_depth": [None, 6], "max_leaf_nodes": [15]},
            "decision_tree": {"max_depth": [None, 4, 8, 12], "min_samples_leaf": [1, 2, 4]},
            "knn": {"n_neighbors": [3, 5, 9], "weights": ["uniform", "distance"]},
            "gaussian_nb": {"var_smoothing": [1e-9, 1e-8, 1e-7]},
            "bernoulli_nb": {"alpha": [0.1, 0.5, 1.0]},
            "multinomial_nb": {"alpha": [0.1, 0.5, 1.0]},
            "xgboost": {"n_estimators": [120, 220], "max_depth": [3, 5, 7], "learning_rate": [0.03, 0.1]},
            "lightgbm": {"n_estimators": [120, 220], "num_leaves": [15, 31, 63], "learning_rate": [0.03, 0.1]},
        },
        "regression": {
            "linear_regression": {},
            "ridge": {"alpha": [0.1, 1.0, 10.0]},
            "lasso": {"alpha": [0.001, 0.01, 0.1]},
            "elasticnet": {"alpha": [0.001, 0.01, 0.1], "l1_ratio": [0.2, 0.5, 0.8]},
            "svr": {"C": [0.5, 1.0, 2.0], "gamma": ["scale", "auto"]},
            "random_forest": {"n_estimators": [120, 200], "max_depth": [None, 6, 12], "min_samples_leaf": [1, 2, 4]},
            "extra_trees": {"n_estimators": [120, 200], "max_depth": [None, 6, 12], "min_samples_leaf": [1, 2, 4]},
            "gradient_boosting": {"n_estimators": [100, 180], "learning_rate": [0.03, 0.1], "max_depth": [2, 3]},
            "hist_gradient_boosting": {"learning_rate": [0.05, 0.1], "max_depth": [None, 6], "max_leaf_nodes": [15]},
            "decision_tree": {"max_depth": [None, 4, 8, 12], "min_samples_leaf": [1, 2, 4]},
            "knn": {"n_neighbors": [3, 5, 9], "weights": ["uniform", "distance"]},
            "xgboost": {"n_estimators": [120, 220], "max_depth": [3, 5, 7], "learning_rate": [0.03, 0.1]},
            "lightgbm": {"n_estimators": [120, 220], "num_leaves": [15, 31, 63], "learning_rate": [0.03, 0.1]},
        },
    }
    return grids.get(task_type, {}).get(model_name, {})


def suggest_model_params(trial: optuna.Trial, task_type: str, model_name: str) -> Dict[str, Any]:
    if model_name == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-2, 10.0, log=True)}
    if model_name in {"svm", "svr"}:
        return {"C": trial.suggest_float("C", 0.1, 10.0, log=True), "gamma": trial.suggest_categorical("gamma", ["scale", "auto"])}
    if model_name in {"random_forest", "extra_trees"}:
        return {"n_estimators": trial.suggest_int("n_estimators", 80, 260, step=20), "max_depth": trial.suggest_categorical("max_depth", [None, 4, 8, 12, 16]), "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6)}
    if model_name in {"gradient_boosting", "hist_gradient_boosting"}:
        if model_name == "hist_gradient_boosting":
            return {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True), "max_depth": trial.suggest_categorical("max_depth", [None, 3, 6]), "max_leaf_nodes": trial.suggest_categorical("max_leaf_nodes", [15, 31])}
        return {"n_estimators": trial.suggest_int("n_estimators", 80, 220, step=20), "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True), "max_depth": trial.suggest_int("max_depth", 2, 5)}
    if model_name == "decision_tree":
        return {"max_depth": trial.suggest_categorical("max_depth", [None, 4, 8, 12, 16]), "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6)}
    if model_name == "knn":
        return {"n_neighbors": trial.suggest_int("n_neighbors", 3, 15, step=2), "weights": trial.suggest_categorical("weights", ["uniform", "distance"])}
    if model_name == "gaussian_nb":
        return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-10, 1e-7, log=True)}
    if model_name in {"bernoulli_nb", "multinomial_nb"}:
        return {"alpha": trial.suggest_float("alpha", 1e-2, 2.0, log=True)}
    if model_name == "xgboost":
        return {"n_estimators": trial.suggest_int("n_estimators", 80, 260, step=20), "max_depth": trial.suggest_int("max_depth", 3, 8), "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True), "subsample": trial.suggest_float("subsample", 0.7, 1.0), "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0), "min_child_weight": trial.suggest_int("min_child_weight", 1, 10)}
    if model_name == "lightgbm":
        return {"n_estimators": trial.suggest_int("n_estimators", 80, 260, step=20), "num_leaves": trial.suggest_int("num_leaves", 15, 127, step=8), "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True), "min_child_samples": trial.suggest_int("min_child_samples", 5, 30)}
    if model_name == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-2, 10.0, log=True)}
    if model_name == "lasso":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 1.0, log=True)}
    if model_name == "elasticnet":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 1.0, log=True), "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 0.9)}
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
    scoring = resolve_scoring(task_type)
    cv_splitter = build_cv_splitter(config, task_type)
    if search_method == "grid":
        raw_grid = build_search_grid(task_type, model_name)
        search_grid = {f"model__{key}": value for key, value in raw_grid.items()}
        if not search_grid:
            pipeline.fit(x_train, y_train)
            return pipeline, {"search_method": "grid", "search_note": "empty_grid_direct_fit"}
        searcher = GridSearchCV(estimator=pipeline, param_grid=search_grid, scoring=scoring, cv=cv_splitter, n_jobs=1)
        searcher.fit(x_train, y_train)
        return searcher.best_estimator_, {"search_method": "grid", "best_params": searcher.best_params_, "best_score": float(searcher.best_score_)}
    if search_method == "halving_grid":
        raw_grid = build_search_grid(task_type, model_name)
        search_grid = {f"model__{key}": value for key, value in raw_grid.items()}
        if not search_grid:
            pipeline.fit(x_train, y_train)
            return pipeline, {"search_method": "halving_grid", "search_note": "empty_grid_direct_fit"}
        searcher = HalvingGridSearchCV(estimator=pipeline, param_grid=search_grid, scoring=scoring, cv=cv_splitter, n_jobs=1, factor=2)
        searcher.fit(x_train, y_train)
        return searcher.best_estimator_, {"search_method": "halving_grid", "best_params": searcher.best_params_, "best_score": float(searcher.best_score_)}

    n_trials = int(config.get("search", {}).get("n_trials", 12))

    def objective(trial: optuna.Trial) -> float:
        params = suggest_model_params(trial, task_type, model_name)
        candidate = clone(pipeline)
        if params:
            candidate.set_params(**{f"model__{key}": value for key, value in params.items()})
        scores = cross_val_score(candidate, x_train, y_train, cv=cv_splitter, scoring=scoring, n_jobs=1)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_pipeline = clone(pipeline)
    best_params = suggest_model_params(study.best_trial, task_type, model_name)
    if best_params:
        best_pipeline.set_params(**{f"model__{key}": value for key, value in best_params.items()})
    best_pipeline.fit(x_train, y_train)
    return best_pipeline, {"search_method": "optuna", "best_params": best_params, "best_score": float(study.best_value), "n_trials": n_trials}


def compute_metrics(
    task_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> Dict[str, float]:
    if task_type == "classification":
        metrics = {"accuracy": float(accuracy_score(y_true, y_pred)), "f1_macro": float(f1_score(y_true, y_pred, average="macro"))}
        if y_prob is not None:
            unique_classes = np.unique(y_true)
            try:
                if len(unique_classes) == 2:
                    auc_value = float(roc_auc_score(y_true, y_prob[:, 1]))
                    metrics["roc_auc"] = auc_value
                    metrics["auc"] = auc_value
                else:
                    auc_value = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
                    metrics["roc_auc_ovr"] = auc_value
                    metrics["auc"] = auc_value
            except Exception:
                pass
        return metrics
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"rmse": rmse, "mae": float(mean_absolute_error(y_true, y_pred)), "r2": float(r2_score(y_true, y_pred))}


def split_data(
    config: Dict[str, Any],
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    task_type = config["data"]["task_type"]
    stratify = df[target_column] if task_type == "classification" and config["data"].get("stratify", True) else None
    return train_test_split(
        df[feature_columns],
        df[target_column],
        test_size=config["data"].get("test_size", 0.2),
        random_state=config["project"]["random_seed"],
        stratify=stratify,
    )

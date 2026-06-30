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
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

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
    config: Dict[str, Any], numeric_columns: List[str], categorical_columns: List[str]
) -> ColumnTransformer:
    """构造 sklearn 预处理器。"""

    preprocess_cfg = config.get("preprocess", {})
    numeric_imputer = preprocess_cfg.get("numeric_imputer", "median")
    categorical_imputer = preprocess_cfg.get("categorical_imputer", "most_frequent")
    scaler_name = preprocess_cfg.get("scaler", "standard")

    numeric_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy=numeric_imputer))]
    if scaler_name == "standard":
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy=categorical_imputer)),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
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

    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_seed),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=random_seed, n_jobs=1
        ),
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


def regression_model_registry(random_seed: int) -> Dict[str, Any]:
    """首版回归模型注册表。"""

    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200, random_state=random_seed, n_jobs=1
        ),
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


def build_model_registry(task_type: str, random_seed: int) -> Dict[str, Any]:
    """根据任务类型返回可用模型注册表。"""

    if task_type == "classification":
        return classification_model_registry(random_seed)
    if task_type == "regression":
        return regression_model_registry(random_seed)
    raise ValueError(f"不支持的 task_type: {task_type}")


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
    preprocessor = build_preprocessor(config, numeric_columns, categorical_columns)
    registry = build_model_registry(task_type, random_seed)

    configured_models = config["models"]["sklearn"][task_type]
    summary_rows: List[Dict[str, Any]] = []

    for model_name in configured_models:
        if model_name not in registry or is_experiment_completed(state, "sklearn", model_name):
            continue

        state["current_stage"] = f"sklearn.{model_name}"
        save_run_state(paths, state)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", clone(registry[model_name])),
            ]
        )

        with mlflow.start_run(run_name=f"sklearn.{model_name}", nested=True):
            pipeline.fit(x_train, y_train)
            y_pred = pipeline.predict(x_test)
            y_prob = pipeline.predict_proba(x_test) if hasattr(pipeline, "predict_proba") else None

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

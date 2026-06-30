#!/usr/bin/env python3
"""
task 运行入口。

这份脚本是当前测试阶段的最小可运行实现，目标只有一个：
把单个 task 目录里的配置、数据、analysis、sklearn、MLflow 记录串起来，
先跑通一条完整主链路。

设计说明：
1. 当前阶段不实现 task_id 自动发号，直接使用已有的 task_000001 / task_000002。
2. 一个 task 对应一个主 MLflow run；analysis / sklearn 各子项用 nested run 记录。
3. run_state.json 以“人类可读”为第一优先级，因此会明确记录已完成实验列表。
4. 为了控制写盘量，模型类只保留必要最终产物；analysis 类完整保留 2D 结果。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


@dataclass
class TaskPaths:
    """集中保存当前 task 会用到的路径，避免路径字符串散落在代码里。"""

    task_dir: Path
    config_path: Path
    run_state_path: Path
    checkpoints_dir: Path
    artifacts_dir: Path
    outputs_dir: Path
    predictions_dir: Path
    metrics_dir: Path
    models_dir: Path
    plots_dir: Path
    analysis_dir: Path
    mlruns_dir: Path
    mpl_config_dir: Path
    numba_cache_dir: Path


def build_task_paths(task_dir: Path) -> TaskPaths:
    """按照 PRD 里约定的目录结构，构造当前 task 运行所需的全部路径。"""

    outputs_dir = task_dir / "outputs"
    return TaskPaths(
        task_dir=task_dir,
        config_path=task_dir / "config.yaml",
        run_state_path=task_dir / "run_state.json",
        checkpoints_dir=task_dir / "checkpoints",
        artifacts_dir=task_dir / "artifacts",
        outputs_dir=outputs_dir,
        predictions_dir=outputs_dir / "predictions",
        metrics_dir=outputs_dir / "metrics",
        models_dir=outputs_dir / "models",
        plots_dir=outputs_dir / "plots",
        analysis_dir=outputs_dir / "analysis",
        mlruns_dir=task_dir / "mlruns",
        mpl_config_dir=task_dir / ".mplconfig",
        numba_cache_dir=task_dir / ".numba_cache",
    )


def ensure_task_dirs(paths: TaskPaths) -> None:
    """确保 task 目录下的核心输出路径存在。"""

    for path in [
        paths.checkpoints_dir,
        paths.artifacts_dir,
        paths.predictions_dir,
        paths.metrics_dir,
        paths.models_dir,
        paths.plots_dir,
        paths.analysis_dir,
        paths.mlruns_dir,
        paths.mpl_config_dir,
        paths.numba_cache_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def configure_runtime_env(paths: TaskPaths) -> None:
    """
    配置运行时环境变量。

    这里主要解决两个现实问题：
    1. matplotlib 在当前机器上默认缓存目录不可写。
    2. umap/numba 在当前机器上默认缓存目录也可能不可写。
    """

    os.environ["MPLCONFIGDIR"] = str(paths.mpl_config_dir)
    os.environ["NUMBA_CACHE_DIR"] = str(paths.numba_cache_dir)


def load_yaml(path: Path) -> Dict[str, Any]:
    """读取单个 task 的配置文件。"""

    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def init_run_state(task_id: str) -> Dict[str, Any]:
    """初始化 run_state.json 的内存结构。"""

    return {
        "task_id": task_id,
        "status": "created",
        "current_stage": "created",
        "last_completed_experiment": None,
        "completed_experiments": {
            "analysis": [],
            "sklearn": [],
            "torch": [],
        },
        "completed_steps": [],
        "pending_steps": [],
        "resume_supported": True,
        "resume_count": 0,
        "updated_at": pd.Timestamp.utcnow().isoformat(),
        "main_run_id": None,
    }


def load_or_create_run_state(paths: TaskPaths, task_id: str) -> Dict[str, Any]:
    """如果已有 run_state.json 就读取，否则创建新的状态结构。"""

    if paths.run_state_path.exists():
        with paths.run_state_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return init_run_state(task_id)


def save_run_state(paths: TaskPaths, state: Dict[str, Any]) -> None:
    """把当前 task 的状态落到 run_state.json。"""

    state["updated_at"] = pd.Timestamp.utcnow().isoformat()
    with paths.run_state_path.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)


def mark_step_completed(state: Dict[str, Any], step_name: str) -> None:
    """记录一个已完成步骤，避免重复写入。"""

    if step_name not in state["completed_steps"]:
        state["completed_steps"].append(step_name)


def mark_experiment_completed(
    state: Dict[str, Any], domain: str, experiment_name: str
) -> None:
    """
    记录一个已完成实验项。

    这里同时更新：
    - last_completed_experiment
    - completed_experiments 分组列表
    - completed_steps
    """

    if experiment_name not in state["completed_experiments"][domain]:
        state["completed_experiments"][domain].append(experiment_name)
    state["last_completed_experiment"] = f"{domain}.{experiment_name}"
    mark_step_completed(state, f"{domain}.{experiment_name}.completed")


def is_experiment_completed(
    state: Dict[str, Any], domain: str, experiment_name: str
) -> bool:
    """判断某个实验是否已在历史运行中完成，用于续跑跳过。"""

    return experiment_name in state["completed_experiments"].get(domain, [])


def read_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """按配置读取 CSV 数据。当前首版先只支持 CSV。"""

    data_cfg = config["data"]
    input_path = Path(data_cfg["input_path"])
    if input_path.suffix.lower() != ".csv":
        raise ValueError("当前首版实现只支持 CSV 输入。")
    return pd.read_csv(input_path)


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


def build_preprocessor(config: Dict[str, Any], numeric_columns: List[str], categorical_columns: List[str]) -> ColumnTransformer:
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
    """
    首版分类模型注册表。

    当前先保守支持三类：
    - logistic_regression
    - random_forest
    - xgboost
    """

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
    from sklearn.linear_model import LinearRegression
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


def compute_metrics(task_type: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> Dict[str, float]:
    """根据任务类型计算首版核心指标。"""

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


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """把字典以 JSON 形式落盘。"""

    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def save_dataframe(path: Path, df: pd.DataFrame) -> None:
    """统一保存 DataFrame。"""

    df.to_csv(path, index=False)


def run_analysis(
    config: Dict[str, Any],
    df: pd.DataFrame,
    target_series: pd.Series,
    feature_df: pd.DataFrame,
    paths: TaskPaths,
    state: Dict[str, Any],
) -> None:
    """
    执行 analysis 类实验。

    这里首版支持：
    - PCA
    - t-SNE
    - UMAP（若环境可用）
    """

    analysis_cfg = config.get("analysis", {})
    if not analysis_cfg.get("enabled", True):
        return

    # analysis 需要数值化输入，因此这里保守地只对原始特征做 one-hot，再做标准化。
    analysis_input = pd.get_dummies(feature_df, dummy_na=True)
    analysis_input = analysis_input.fillna(analysis_input.median(numeric_only=True)).fillna(0.0)
    analysis_input_values = StandardScaler().fit_transform(analysis_input)

    analysis_jobs: List[Tuple[str, bool]] = [
        ("pca", analysis_cfg.get("pca", {}).get("enabled", True)),
        ("tsne", analysis_cfg.get("tsne", {}).get("enabled", True)),
        ("umap", analysis_cfg.get("umap", {}).get("enabled", True)),
    ]

    for name, enabled in analysis_jobs:
        if not enabled or is_experiment_completed(state, "analysis", name):
            continue

        state["current_stage"] = f"analysis.{name}"
        save_run_state(paths, state)

        with mlflow.start_run(run_name=f"analysis.{name}", nested=True):
            if name == "pca":
                reducer = PCA(n_components=2, random_state=config["project"]["random_seed"])
                coords = reducer.fit_transform(analysis_input_values)
                mlflow.log_metric("explained_variance_sum", float(np.sum(reducer.explained_variance_ratio_)))
            elif name == "tsne":
                tsne_cfg = analysis_cfg.get("tsne", {})
                reducer = TSNE(
                    n_components=2,
                    perplexity=tsne_cfg.get("perplexity", 30),
                    init="pca",
                    learning_rate="auto",
                    random_state=config["project"]["random_seed"],
                )
                coords = reducer.fit_transform(analysis_input_values)
            else:
                try:
                    import umap  # type: ignore
                except Exception as exc:
                    mlflow.log_param("skipped_reason", f"umap_unavailable: {exc}")
                    continue
                umap_cfg = analysis_cfg.get("umap", {})
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=umap_cfg.get("n_neighbors", 15),
                    min_dist=umap_cfg.get("min_dist", 0.1),
                    random_state=config["project"]["random_seed"],
                )
                coords = reducer.fit_transform(analysis_input_values)

            result_df = pd.DataFrame(
                {
                    "x": coords[:, 0],
                    "y": coords[:, 1],
                    "target": target_series.astype(str).values,
                }
            )
            result_path = paths.analysis_dir / f"{name}_2d.csv"
            save_dataframe(result_path, result_df)
            mlflow.log_artifact(str(result_path), artifact_path="analysis")

            mark_experiment_completed(state, "analysis", name)
            save_run_state(paths, state)


def split_data(
    config: Dict[str, Any], df: pd.DataFrame, feature_columns: List[str], target_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """按配置切分训练集与测试集。"""

    task_type = config["data"]["task_type"]
    stratify = df[target_column] if task_type == "classification" and config["data"].get("stratify", True) else None
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
) -> None:
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
    else:
        label_encoder = None

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

            summary_rows.append({"model_name": model_name, **metrics})
            mark_experiment_completed(state, "sklearn", model_name)
            save_run_state(paths, state)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values(
            by=list(summary_rows[0].keys())[1], ascending=False
        )
        summary_path = paths.metrics_dir / "sklearn_summary.csv"
        save_dataframe(summary_path, summary_df)


def run_task(task_dir: Path) -> None:
    """单个 task 的总执行入口。"""

    paths = build_task_paths(task_dir)
    ensure_task_dirs(paths)
    configure_runtime_env(paths)

    config = load_yaml(paths.config_path)
    task_id = config["task"]["task_id"]
    state = load_or_create_run_state(paths, task_id)

    if state["status"] in {"failed", "terminated", "paused"}:
        state["resume_count"] = int(state.get("resume_count", 0)) + 1

    state["status"] = "running"
    state["current_stage"] = "bootstrap"
    save_run_state(paths, state)

    mlflow.set_tracking_uri(config.get("mlflow", {}).get("tracking_uri", str(paths.mlruns_dir)))
    mlflow.set_experiment(config.get("mlflow", {}).get("experiment_name", "homemade_datarobot"))

    existing_run_id = state.get("main_run_id")
    run_context = (
        mlflow.start_run(run_id=existing_run_id)
        if existing_run_id
        else mlflow.start_run(run_name=task_id)
    )

    with run_context as main_run:
        state["main_run_id"] = main_run.info.run_id
        save_run_state(paths, state)

        mlflow.set_tag("task_id", task_id)
        mlflow.set_tag("run_level", "task")
        mlflow.log_dict(config, "config_snapshot.yaml")

        df = read_dataset(config)
        mark_step_completed(state, "data_loaded")
        save_run_state(paths, state)

        target_column = config["data"]["target_column"]
        numeric_columns, categorical_columns = infer_feature_columns(config, df)
        feature_columns = numeric_columns + categorical_columns
        feature_df = df[feature_columns].copy()
        target_series = df[target_column].copy()

        mark_step_completed(state, "schema_checked")
        save_run_state(paths, state)

        run_analysis(config, df, target_series, feature_df, paths, state)
        run_sklearn(config, df, paths, state)

        state["status"] = "completed"
        state["current_stage"] = "completed"
        save_run_state(paths, state)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="运行单个 task 的 MLflow 实验链路。")
    parser.add_argument("task_dir", type=Path, help="例如 storage/task_000001")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_task(args.task_dir.resolve())

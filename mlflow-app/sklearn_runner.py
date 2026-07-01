"""
sklearn 执行入口。

当前文件只保留 sklearn 主流程编排，把大部分通用细节下沉到 sklearn_common.py，
避免继续维持近千行的单文件。
"""

from __future__ import annotations

import mlflow
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from sklearn_common import (
    apply_class_balance_to_model,
    build_model_registry,
    build_preprocessor,
    compute_metrics,
    detect_imbalance,
    infer_feature_columns,
    oversample_training_data,
    resolve_search_method,
    run_search_for_pipeline,
    save_classification_plots,
    save_dataframe,
    save_json,
    split_data,
)
from task_runtime import TaskPaths, is_experiment_completed, mark_experiment_completed, save_run_state


def run_sklearn(
    config: dict,
    df: pd.DataFrame,
    paths: TaskPaths,
    state: dict,
) -> dict:
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
    summary_rows = []
    imbalance_enabled = bool(config.get("imbalance", {}).get("enabled", task_type == "classification"))

    if task_type == "classification" and imbalance_enabled and detect_imbalance(y_train):
        x_train, y_train = oversample_training_data(x_train=x_train, y_train=y_train, random_seed=random_seed)

    for model_name in configured_models:
        if model_name not in registry or is_experiment_completed(state, "sklearn", model_name):
            continue

        state["current_stage"] = f"sklearn.{model_name}"
        save_run_state(paths, state)

        model_instance = clone(registry[model_name])
        if task_type == "classification" and imbalance_enabled:
            model_instance = apply_class_balance_to_model(model_name=model_name, model=model_instance, y_train=y_train)
        search_method = resolve_search_method(config=config, model_name=model_name, task_type=task_type, n_rows=len(df), domain="sklearn")

        preprocessor = build_preprocessor(
            config=config,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            model_name=model_name,
        )
        pipeline = __import__("sklearn.pipeline").pipeline.Pipeline(steps=[("preprocessor", preprocessor), ("model", model_instance)])

        with mlflow.start_run(run_name=model_name):
            mlflow.set_tag("task_id", config["task"]["task_id"])
            mlflow.set_tag("run_level", "item")
            mlflow.set_tag("item_name", model_name)
            mlflow.set_tag("item_kind", "model")
            mlflow.set_tag("model_family", "sklearn")
            mlflow.log_dict(config, "config_snapshot.yaml")
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
            y_prob = trained_pipeline.predict_proba(x_test) if hasattr(trained_pipeline, "predict_proba") else None
            metrics = compute_metrics(task_type, y_test.to_numpy(), y_pred, y_prob)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            prediction_df = x_test.copy()
            prediction_df["y_true"] = y_test.values
            prediction_df["y_pred"] = y_pred
            prediction_path = paths.predictions_dir / f"{model_name}_predictions.csv"
            save_dataframe(prediction_path, prediction_df)
            mlflow.log_artifact(str(prediction_path), artifact_path="predictions")

            metrics_path = paths.metrics_dir / f"{model_name}_metrics.json"
            save_json(
                metrics_path,
                {
                    "task_id": config["task"]["task_id"],
                    "model_name": model_name,
                    "task_type": task_type,
                    "metrics": metrics,
                },
            )
            mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

            if task_type == "classification":
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                report_path = paths.metrics_dir / f"{model_name}_classification_report.json"
                save_json(report_path, report)
                mlflow.log_artifact(str(report_path), artifact_path="metrics")
                save_classification_plots(y_true=y_test.to_numpy(), y_pred=y_pred, y_prob=y_prob, model_name=model_name, paths=paths)
                mlflow.log_artifact(str(paths.plots_dir / f"{model_name}_confusion_matrix.png"), artifact_path="plots")
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

"""
torch 执行入口。

当前文件只保留 torch 主流程编排，大块通用逻辑下沉到 torch_common.py，
避免继续维持数百行的单文件。
"""

from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd

from sklearn_common import detect_imbalance, resolve_search_method
from task_runtime import TaskPaths, is_experiment_completed, mark_experiment_completed, save_run_state
from torch_common import (
    build_tabular_matrix,
    build_torch_model,
    compute_torch_metrics,
    oversample_numpy_training_data,
    save_dataframe,
    save_json,
    save_torch_classification_plots,
    split_tensor_data,
    train_standard_torch_model,
    tune_torch_model_with_optuna,
)


def run_torch(config: dict, df: pd.DataFrame, paths: TaskPaths, state: dict) -> dict:
    configured_models = config["models"].get("torch", {}).get(config["data"]["task_type"], [])
    if not configured_models:
        return {}

    task_type = config["data"]["task_type"]
    x_df, target = build_tabular_matrix(config, df)
    if task_type == "classification":
        from sklearn.preprocessing import LabelEncoder

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(target.astype(str))
        output_dim = len(np.unique(y))
    else:
        y = target.to_numpy(dtype=np.float32)
        output_dim = 1

    x_train, x_test, y_train, y_test = split_tensor_data(config, x_df, y)
    summary_rows = []
    imbalance_enabled = bool(config.get("imbalance", {}).get("enabled", task_type == "classification"))
    if task_type == "classification" and imbalance_enabled and detect_imbalance(y_train):
        x_train, y_train = oversample_numpy_training_data(x_train=x_train, y_train=y_train, random_seed=int(config["project"]["random_seed"]))

    for model_name in configured_models:
        if is_experiment_completed(state, "torch", model_name):
            continue

        state["current_stage"] = f"torch.{model_name}"
        save_run_state(paths, state)

        with mlflow.start_run(run_name=model_name):
            mlflow.set_tag("task_id", config["task"]["task_id"])
            mlflow.set_tag("run_level", "item")
            mlflow.set_tag("item_name", model_name)
            mlflow.set_tag("item_kind", "model")
            mlflow.set_tag("model_family", "torch")
            mlflow.log_dict(config, "config_snapshot.yaml")
            training_cfg = dict(config.get("training", {}).get("torch", {}))
            search_method = resolve_search_method(config=config, model_name=model_name, task_type=task_type, n_rows=len(df), domain="torch")
            mlflow.log_param("search_method", search_method)
            mlflow.log_param("imbalance_enabled", imbalance_enabled)
            mlflow.log_param("stratify_enabled", bool(config["data"].get("stratify", True)))

            optuna_result = {}
            if search_method == "optuna":
                optuna_result = tune_torch_model_with_optuna(
                    model_name=model_name,
                    config=config,
                    x_train=x_train,
                    y_train=y_train,
                    x_valid=x_test,
                    y_valid=y_test,
                    task_type=task_type,
                )
                if model_name == "mlp":
                    training_cfg["hidden_dims"] = [optuna_result["best_params"]["hidden_dim_1"], optuna_result["best_params"]["hidden_dim_2"]]
                    training_cfg["dropout"] = optuna_result["best_params"]["dropout"]
                if "learning_rate" in optuna_result["best_params"]:
                    training_cfg["learning_rate"] = optuna_result["best_params"]["learning_rate"]
                if "batch_size" in optuna_result["best_params"]:
                    training_cfg["batch_size"] = optuna_result["best_params"]["batch_size"]
                mlflow.log_param("optuna_n_trials", optuna_result["n_trials"])
                for key, value in optuna_result["best_params"].items():
                    mlflow.log_param(f"best_{key}", value)
                mlflow.log_metric("optuna_best_value", optuna_result["best_value"])

            if model_name == "tabnet":
                from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor  # type: ignore

                tabnet_params = {"seed": int(config["project"]["random_seed"]), "verbose": 0}
                for key in ("n_d", "n_a", "n_steps", "gamma"):
                    if key in training_cfg:
                        tabnet_params[key] = training_cfg[key]
                    elif key in optuna_result.get("best_params", {}):
                        tabnet_params[key] = optuna_result["best_params"][key]
                model_cls = TabNetClassifier if task_type == "classification" else TabNetRegressor
                model = model_cls(**tabnet_params)
                model.fit(
                    x_train,
                    y_train if task_type == "classification" else y_train.reshape(-1, 1),
                    eval_set=[(x_test, y_test if task_type == "classification" else y_test.reshape(-1, 1))],
                    max_epochs=int(training_cfg.get("epochs", 30)),
                    patience=int(training_cfg.get("early_stopping", {}).get("patience", 5)),
                    batch_size=int(training_cfg.get("batch_size", 64)),
                    virtual_batch_size=min(int(training_cfg.get("batch_size", 64)), 32),
                )
                if task_type == "classification":
                    y_pred = model.predict(x_test)
                    y_prob = model.predict_proba(x_test)
                else:
                    y_pred = model.predict(x_test).reshape(-1)
                    y_prob = None
                metrics = compute_torch_metrics(task_type, y_test, y_pred, y_prob)
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                result = {"y_pred": y_pred, "y_prob": y_prob, "metrics": metrics}
            else:
                model = build_torch_model(model_name, input_dim=x_train.shape[1], output_dim=output_dim, training_cfg=training_cfg)
                result = train_standard_torch_model(
                    model_name=model_name,
                    model=model,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    task_type=task_type,
                    paths=paths,
                    config={**config, "training": {"torch": training_cfg}},
                )

            prediction_path = paths.predictions_dir / f"torch_{model_name}_predictions.csv"
            save_dataframe(prediction_path, pd.DataFrame({"y_true": y_test, "y_pred": result["y_pred"]}))
            mlflow.log_artifact(str(prediction_path), artifact_path="predictions")

            metrics_path = paths.metrics_dir / f"torch_{model_name}_metrics.json"
            save_json(metrics_path, {"task_id": config["task"]["task_id"], "model_name": model_name, "task_type": task_type, "metrics": result["metrics"]})
            mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

            if model_name != "tabnet":
                mlflow.log_artifact(str(paths.plots_dir / f"torch_{model_name}_loss_curve.png"), artifact_path="plots")
            if task_type == "classification":
                save_torch_classification_plots(y_true=y_test, y_pred=result["y_pred"], y_prob=result["y_prob"], model_name=model_name, paths=paths)
                mlflow.log_artifact(str(paths.plots_dir / f"torch_{model_name}_confusion_matrix.png"), artifact_path="plots")
                roc_path = paths.plots_dir / f"torch_{model_name}_roc_curve.png"
                if roc_path.exists():
                    mlflow.log_artifact(str(roc_path), artifact_path="plots")

            summary_rows.append({"model_name": model_name, **result["metrics"]})
            mark_experiment_completed(state, "torch", model_name)
            save_run_state(paths, state)

    if summary_rows:
        primary_sort_key = list(summary_rows[0].keys())[1]
        summary_df = pd.DataFrame(summary_rows).sort_values(by=primary_sort_key, ascending=False)
        summary_path = paths.metrics_dir / "torch_summary.csv"
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

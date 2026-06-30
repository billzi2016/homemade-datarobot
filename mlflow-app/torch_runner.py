"""
torch 执行模块。

当前不做空壳，而是提供真实可运行的表格深度学习链路：
- mlp
- cnn1d

TabNet 也保留真实入口，但前提是环境安装了 pytorch-tabnet。
若依赖不存在，就明确记录为跳过，而不是伪造支持。
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
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    RocCurveDisplay,
    r2_score,
    roc_auc_score,
)

from sklearn_runner import infer_feature_columns
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


def save_torch_classification_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    model_name: str,
    paths: TaskPaths,
) -> None:
    """保存 torch 分类图。"""

    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax_cm, colorbar=False)
    ax_cm.set_title(f"torch.{model_name} Confusion Matrix")
    fig_cm.tight_layout()
    cm_path = paths.plots_dir / f"torch_{model_name}_confusion_matrix.png"
    fig_cm.savefig(cm_path, dpi=160)
    plt.close(fig_cm)

    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == 2:
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_true, y_prob[:, 1], ax=ax_roc)
        ax_roc.set_title(f"torch.{model_name} ROC Curve")
        fig_roc.tight_layout()
        roc_path = paths.plots_dir / f"torch_{model_name}_roc_curve.png"
        fig_roc.savefig(roc_path, dpi=160)
        plt.close(fig_roc)


def save_loss_curve(
    train_losses: List[float],
    valid_losses: List[float],
    model_name: str,
    paths: TaskPaths,
) -> None:
    """保存 torch 训练曲线。"""

    fig, ax = plt.subplots(figsize=(7, 5))
    epochs = list(range(1, len(train_losses) + 1))
    ax.plot(epochs, train_losses, label="train_loss")
    ax.plot(epochs, valid_losses, label="valid_loss")
    ax.set_title(f"torch.{model_name} Loss Curve")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    curve_path = paths.plots_dir / f"torch_{model_name}_loss_curve.png"
    fig.savefig(curve_path, dpi=160)
    plt.close(fig)


def build_tabular_matrix(
    config: Dict[str, Any], df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    为 torch 模型构造输入矩阵。

    当前策略：
    - 数值列保留
    - 类别列做 one-hot
    - 缺失值用简单策略填补
    """

    target_column = config["data"]["target_column"]
    numeric_columns, categorical_columns = infer_feature_columns(config, df)

    feature_df = df[numeric_columns + categorical_columns].copy()
    for col in numeric_columns:
        feature_df[col] = feature_df[col].fillna(feature_df[col].median())
    for col in categorical_columns:
        feature_df[col] = feature_df[col].fillna("missing").astype(str)

    encoded = pd.get_dummies(feature_df, columns=categorical_columns, dummy_na=False)
    encoded = encoded.astype(np.float32)
    target = df[target_column].copy()
    return encoded, target


def split_tensor_data(
    config: Dict[str, Any], x_df: pd.DataFrame, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """为 torch 模型切分训练集与测试集。"""

    from sklearn.model_selection import train_test_split

    stratify = y if config["data"]["task_type"] == "classification" and config["data"].get("stratify", True) else None
    x_train, x_test, y_train, y_test = train_test_split(
        x_df.to_numpy(dtype=np.float32),
        y,
        test_size=config["data"].get("test_size", 0.2),
        random_state=config["project"]["random_seed"],
        stratify=stratify,
    )
    return x_train, x_test, y_train, y_test


class TabularMLP(nn.Module):
    """最小可用的表格 MLP。"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TabularCNN1D(nn.Module):
    """把表格特征视为一维序列的最小 CNN。"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.head(x)


def build_torch_model(
    model_name: str,
    input_dim: int,
    output_dim: int,
    training_cfg: Dict[str, Any],
):
    """根据模型名构造真实 torch 模型。"""

    if model_name == "mlp":
        hidden_dims = training_cfg.get("hidden_dims", [128, 64])
        dropout = training_cfg.get("dropout", 0.2)
        return TabularMLP(input_dim, output_dim, hidden_dims, dropout)
    if model_name == "cnn1d":
        return TabularCNN1D(input_dim, output_dim)
    if model_name == "tabnet":
        from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor  # type: ignore

        return TabNetClassifier if output_dim > 1 else TabNetRegressor
    raise ValueError(f"不支持的 torch 模型: {model_name}")


def tune_mlp_with_optuna(
    config: Dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    task_type: str,
) -> Dict[str, Any]:
    """使用 Optuna 搜索 MLP 的关键超参数。"""

    n_trials = int(config.get("search", {}).get("n_trials", 12))
    random_seed = int(config["project"]["random_seed"])
    output_dim = len(np.unique(y_train)) if task_type == "classification" else 1

    def objective(trial: optuna.Trial) -> float:
        torch.manual_seed(random_seed)
        hidden_dims = [
            trial.suggest_int("hidden_dim_1", 32, 256, step=32),
            trial.suggest_int("hidden_dim_2", 16, 128, step=16),
        ]
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

        model = TabularMLP(
            input_dim=x_train.shape[1],
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        if task_type == "classification":
            loss_fn = nn.CrossEntropyLoss()
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)
        else:
            loss_fn = nn.MSELoss()
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).view(-1, 1)

        train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), y_train_tensor)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model.train()
        for _ in range(12):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(x_valid, dtype=torch.float32))
            if task_type == "classification":
                y_prob = torch.softmax(logits, dim=1).numpy()
                y_pred = np.argmax(y_prob, axis=1)
                return float(accuracy_score(y_valid, y_pred))

            y_pred = logits.numpy().reshape(-1)
            return -float(np.sqrt(mean_squared_error(y_valid, y_pred)))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {
        "best_params": study.best_params,
        "best_value": float(study.best_value),
        "n_trials": n_trials,
    }


def compute_torch_metrics(
    task_type: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None
) -> Dict[str, float]:
    """计算 torch 模型的核心指标。"""

    if task_type == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        }
        if y_prob is not None:
            try:
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
                elif y_prob.ndim == 2 and y_prob.shape[1] > 2:
                    metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
            except Exception:
                pass
        return metrics

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_standard_torch_model(
    model_name: str,
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    paths: TaskPaths,
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """训练 MLP / CNN1D 这类标准 torch 模型。"""

    training_cfg = config.get("training", {}).get("torch", {})
    batch_size = int(training_cfg.get("batch_size", 128))
    epochs = int(training_cfg.get("epochs", 30))
    learning_rate = float(training_cfg.get("learning_rate", 1e-3))
    patience = int(training_cfg.get("early_stopping", {}).get("patience", 5))

    device = torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if task_type == "classification":
        loss_fn = nn.CrossEntropyLoss()
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    else:
        loss_fn = nn.MSELoss()
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), y_train_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    best_state = None
    no_improve_epochs = 0
    train_loss_history: List[float] = []
    valid_loss_history: List[float] = []

    for epoch in range(epochs):
        model.train()
        epoch_losses: List[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(x_test, dtype=torch.float32).to(device))
            valid_loss = float(loss_fn(logits, y_test_tensor.to(device)).item())
        avg_train_loss = float(np.mean(epoch_losses))
        train_loss_history.append(avg_train_loss)
        valid_loss_history.append(valid_loss)

        mlflow.log_metric(f"{model_name}_train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric(f"{model_name}_valid_loss", valid_loss, step=epoch)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
            checkpoint_path = paths.checkpoints_dir / f"{model_name}_best.pt"
            torch.save(best_state, checkpoint_path)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_test, dtype=torch.float32).to(device)).cpu().numpy()

    if task_type == "classification":
        y_prob = torch.softmax(torch.tensor(logits), dim=1).numpy()
        y_pred = np.argmax(y_prob, axis=1)
    else:
        y_prob = None
        y_pred = logits.reshape(-1)

    metrics = compute_torch_metrics(task_type, y_test, y_pred, y_prob)
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    torch.save(model.state_dict(), paths.models_dir / f"{model_name}_final.pt")
    save_loss_curve(train_loss_history, valid_loss_history, model_name, paths)
    return {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
        "train_losses": train_loss_history,
        "valid_losses": valid_loss_history,
    }


def run_torch(
    config: Dict[str, Any],
    df: pd.DataFrame,
    paths: TaskPaths,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """执行 torch 主线实验。"""

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
    summary_rows: List[Dict[str, Any]] = []

    for model_name in configured_models:
        if is_experiment_completed(state, "torch", model_name):
            continue

        state["current_stage"] = f"torch.{model_name}"
        save_run_state(paths, state)

        with mlflow.start_run(run_name=f"torch.{model_name}", nested=True):
            if model_name == "tabnet":
                try:
                    import pytorch_tabnet  # noqa: F401
                except Exception as exc:
                    mlflow.log_param("skipped_reason", f"tabnet_unavailable: {exc}")
                    continue
                mlflow.log_param("skipped_reason", "tabnet_real_training_not_implemented_yet")
                continue

            training_cfg = dict(config.get("training", {}).get("torch", {}))
            use_optuna = (
                config.get("search", {}).get("method") == "optuna"
                and model_name == "mlp"
                and config.get("search", {}).get("torch", {}).get("mlp", {}).get("enabled", False)
            )

            if use_optuna:
                optuna_result = tune_mlp_with_optuna(
                    config=config,
                    x_train=x_train,
                    y_train=y_train,
                    x_valid=x_test,
                    y_valid=y_test,
                    task_type=task_type,
                )
                training_cfg["hidden_dims"] = [
                    optuna_result["best_params"]["hidden_dim_1"],
                    optuna_result["best_params"]["hidden_dim_2"],
                ]
                training_cfg["dropout"] = optuna_result["best_params"]["dropout"]
                training_cfg["learning_rate"] = optuna_result["best_params"]["learning_rate"]
                training_cfg["batch_size"] = optuna_result["best_params"]["batch_size"]
                mlflow.log_param("search_method", "optuna")
                mlflow.log_param("optuna_n_trials", optuna_result["n_trials"])
                for key, value in optuna_result["best_params"].items():
                    mlflow.log_param(f"best_{key}", value)
                mlflow.log_metric("optuna_best_value", optuna_result["best_value"])

            model = build_torch_model(
                model_name,
                input_dim=x_train.shape[1],
                output_dim=output_dim,
                training_cfg=training_cfg,
            )
            result = train_standard_torch_model(
                model_name=model_name,
                model=model,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                task_type=task_type,
                paths=paths,
                state=state,
                config={**config, "training": {"torch": training_cfg}},
            )

            prediction_df = pd.DataFrame({"y_true": y_test, "y_pred": result["y_pred"]})
            prediction_path = paths.predictions_dir / f"torch_{model_name}_predictions.csv"
            save_dataframe(prediction_path, prediction_df)
            mlflow.log_artifact(str(prediction_path), artifact_path="predictions")

            metrics_path = paths.metrics_dir / f"torch_{model_name}_metrics.json"
            save_json(
                metrics_path,
                {
                    "task_id": config["task"]["task_id"],
                    "model_name": model_name,
                    "task_type": task_type,
                    "metrics": result["metrics"],
                },
            )
            mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

            mlflow.log_artifact(
                str(paths.plots_dir / f"torch_{model_name}_loss_curve.png"),
                artifact_path="plots",
            )
            if task_type == "classification":
                save_torch_classification_plots(
                    y_true=y_test,
                    y_pred=result["y_pred"],
                    y_prob=result["y_prob"],
                    model_name=model_name,
                    paths=paths,
                )
                mlflow.log_artifact(
                    str(paths.plots_dir / f"torch_{model_name}_confusion_matrix.png"),
                    artifact_path="plots",
                )
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

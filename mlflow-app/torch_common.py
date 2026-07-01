"""
torch 公共支持模块。

这里收口 torch 主链路的大块公共逻辑：
- 数据矩阵构造
- 模型定义
- 训练循环
- Optuna 搜索
- 指标与图
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn_common import infer_feature_columns
from task_runtime import TaskPaths


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def save_dataframe(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def save_torch_classification_plots(
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
    ax_cm.set_title(f"torch.{model_name} Confusion Matrix")
    fig_cm.tight_layout()
    fig_cm.savefig(paths.plots_dir / f"torch_{model_name}_confusion_matrix.png", dpi=160)
    plt.close(fig_cm)
    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == 2:
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_true, y_prob[:, 1], ax=ax_roc)
        ax_roc.set_title(f"torch.{model_name} ROC Curve")
        fig_roc.tight_layout()
        fig_roc.savefig(paths.plots_dir / f"torch_{model_name}_roc_curve.png", dpi=160)
        plt.close(fig_roc)


def save_loss_curve(train_losses: List[float], valid_losses: List[float], model_name: str, paths: TaskPaths) -> None:
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
    fig.savefig(paths.plots_dir / f"torch_{model_name}_loss_curve.png", dpi=160)
    plt.close(fig)


def build_tabular_matrix(config: Dict[str, Any], df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    target_column = config["data"]["target_column"]
    numeric_columns, categorical_columns = infer_feature_columns(config, df)
    feature_df = df[numeric_columns + categorical_columns].copy()
    for col in numeric_columns:
        feature_df[col] = feature_df[col].fillna(feature_df[col].median())
    for col in categorical_columns:
        feature_df[col] = feature_df[col].fillna("missing").astype(str)
    encoded = pd.get_dummies(feature_df, columns=categorical_columns, dummy_na=False).astype(np.float32)
    return encoded, df[target_column].copy()


def split_tensor_data(config: Dict[str, Any], x_df: pd.DataFrame, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    stratify = y if config["data"]["task_type"] == "classification" and config["data"].get("stratify", True) else None
    return train_test_split(
        x_df.to_numpy(dtype=np.float32),
        y,
        test_size=config["data"].get("test_size", 0.2),
        random_state=config["project"]["random_seed"],
        stratify=stratify,
    )


def oversample_numpy_training_data(x_train: np.ndarray, y_train: np.ndarray, random_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    classes, counts = np.unique(y_train, return_counts=True)
    max_count = int(counts.max())
    sampled_indices: List[np.ndarray] = []
    for class_value in classes:
        class_indices = np.where(y_train == class_value)[0]
        sampled_indices.append(rng.choice(class_indices, size=max_count, replace=True))
    final_indices = np.concatenate(sampled_indices)
    rng.shuffle(final_indices)
    return x_train[final_indices], y_train[final_indices]


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TabularCNN1D(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 8, 64), nn.ReLU(), nn.Linear(64, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x.unsqueeze(1)))


def build_torch_model(model_name: str, input_dim: int, output_dim: int, training_cfg: Dict[str, Any]):
    if model_name == "mlp":
        return TabularMLP(input_dim, output_dim, training_cfg.get("hidden_dims", [128, 64]), training_cfg.get("dropout", 0.2))
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
    n_trials = int(config.get("search", {}).get("n_trials", 12))
    random_seed = int(config["project"]["random_seed"])
    output_dim = len(np.unique(y_train)) if task_type == "classification" else 1

    def objective(trial: optuna.Trial) -> float:
        torch.manual_seed(random_seed)
        hidden_dims = [trial.suggest_int("hidden_dim_1", 32, 256, step=32), trial.suggest_int("hidden_dim_2", 16, 128, step=16)]
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        model = TabularMLP(input_dim=x_train.shape[1], output_dim=output_dim, hidden_dims=hidden_dims, dropout=dropout)
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
                loss = loss_fn(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(x_valid, dtype=torch.float32))
            if task_type == "classification":
                y_prob = torch.softmax(logits, dim=1).numpy()
                return float(accuracy_score(y_valid, np.argmax(y_prob, axis=1)))
            return -float(np.sqrt(mean_squared_error(y_valid, logits.numpy().reshape(-1))))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"best_params": study.best_params, "best_value": float(study.best_value), "n_trials": n_trials}


def tune_torch_model_with_optuna(
    model_name: str,
    config: Dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    task_type: str,
) -> Dict[str, Any]:
    if model_name == "mlp":
        return tune_mlp_with_optuna(config, x_train, y_train, x_valid, y_valid, task_type)

    n_trials = int(config.get("search", {}).get("n_trials", 12))
    random_seed = int(config["project"]["random_seed"])
    output_dim = len(np.unique(y_train)) if task_type == "classification" else 1

    def objective(trial: optuna.Trial) -> float:
        torch.manual_seed(random_seed)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        if model_name == "cnn1d":
            model = TabularCNN1D(input_dim=x_train.shape[1], output_dim=output_dim)
            result = train_standard_torch_model(
                model_name=model_name,
                model=model,
                x_train=x_train,
                y_train=y_train,
                x_test=x_valid,
                y_test=y_valid,
                task_type=task_type,
                paths=TaskPaths.__new__(TaskPaths),
                config={"training": {"torch": {"learning_rate": learning_rate, "batch_size": batch_size, "epochs": 12}}},
                enable_mlflow_logging=False,
            )
            return float(result["metrics"]["accuracy"] if task_type == "classification" else -result["metrics"]["rmse"])
        if model_name == "tabnet":
            from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor  # type: ignore

            model_cls = TabNetClassifier if task_type == "classification" else TabNetRegressor
            model = model_cls(
                n_d=trial.suggest_int("n_d", 8, 32, step=8),
                n_a=trial.suggest_int("n_a", 8, 32, step=8),
                n_steps=trial.suggest_int("n_steps", 3, 7),
                gamma=trial.suggest_float("gamma", 1.0, 1.8),
                seed=random_seed,
                verbose=0,
            )
            model.fit(
                x_train,
                y_train if task_type == "classification" else y_train.reshape(-1, 1),
                eval_set=[(x_valid, y_valid if task_type == "classification" else y_valid.reshape(-1, 1))],
                max_epochs=30,
                patience=5,
                batch_size=batch_size,
                virtual_batch_size=min(batch_size, 32),
            )
            if task_type == "classification":
                return float(accuracy_score(y_valid, model.predict(x_valid)))
            return -float(np.sqrt(mean_squared_error(y_valid, model.predict(x_valid).reshape(-1))))
        raise ValueError(f"不支持的 torch Optuna 模型: {model_name}")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"best_params": study.best_params, "best_value": float(study.best_value), "n_trials": n_trials}


def compute_torch_metrics(task_type: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> Dict[str, float]:
    if task_type == "classification":
        metrics = {"accuracy": float(accuracy_score(y_true, y_pred)), "f1_macro": float(f1_score(y_true, y_pred, average="macro"))}
        if y_prob is not None:
            try:
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    auc_value = float(roc_auc_score(y_true, y_prob[:, 1]))
                    metrics["roc_auc"] = auc_value
                    metrics["auc"] = auc_value
                elif y_prob.ndim == 2 and y_prob.shape[1] > 2:
                    auc_value = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
                    metrics["roc_auc_ovr"] = auc_value
                    metrics["auc"] = auc_value
            except Exception:
                pass
        return metrics
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"rmse": rmse, "mae": float(mean_absolute_error(y_true, y_pred)), "r2": float(r2_score(y_true, y_pred))}


def train_standard_torch_model(
    model_name: str,
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    paths: TaskPaths,
    config: Dict[str, Any],
    enable_mlflow_logging: bool = True,
) -> Dict[str, Any]:
    if enable_mlflow_logging:
        import mlflow

    training_cfg = config.get("training", {}).get("torch", {})
    batch_size = int(training_cfg.get("batch_size", 128))
    epochs = int(training_cfg.get("epochs", 30))
    learning_rate = float(training_cfg.get("learning_rate", 1e-3))
    patience = int(training_cfg.get("early_stopping", {}).get("patience", 5))

    device = torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if task_type == "classification":
        class_weights = None
        if config.get("imbalance", {}).get("enabled", True):
            class_counts = np.bincount(y_train)
            class_weights = class_counts.sum() / np.maximum(class_counts, 1)
            class_weights = class_weights / class_weights.sum() * len(class_weights)
        loss_fn = nn.CrossEntropyLoss(weight=None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32))
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
            loss = loss_fn(model(batch_x), batch_y)
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
        if enable_mlflow_logging:
            mlflow.log_metric(f"{model_name}_train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric(f"{model_name}_valid_loss", valid_loss, step=epoch)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
            if hasattr(paths, "checkpoints_dir"):
                torch.save(best_state, paths.checkpoints_dir / f"{model_name}_best.pt")
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
    if enable_mlflow_logging:
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
    if hasattr(paths, "models_dir"):
        torch.save(model.state_dict(), paths.models_dir / f"{model_name}_final.pt")
    if hasattr(paths, "plots_dir"):
        save_loss_curve(train_loss_history, valid_loss_history, model_name, paths)
    return {"y_pred": y_pred, "y_prob": y_prob, "metrics": metrics, "train_losses": train_loss_history, "valid_losses": valid_loss_history}

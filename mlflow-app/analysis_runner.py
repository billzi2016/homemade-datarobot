"""
analysis 执行模块。

当前模块只负责真正的数据降维与结果落盘，不再直接创建 MLflow run。
这样做的原因很明确：
1. pca / tsne / umap 不应该混进模型排行榜主列表；
2. 它们更适合作为一个统一的 data_visualization 汇总 run 的 artifacts；
3. 但 analysis 原始 2D 结果和散点图仍然必须完整保留。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import mlflow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from task_runtime import (
    TaskPaths,
    is_experiment_completed,
    mark_experiment_completed,
    save_run_state,
)


def save_dataframe(path, df: pd.DataFrame) -> None:
    """统一保存 DataFrame。"""

    df.to_csv(path, index=False)


def save_analysis_scatter_plot(result_df: pd.DataFrame, plot_path, title: str) -> None:
    """
    保存 analysis 2D 散点图。

    这里不用特别花哨的图形库，原因很简单：
    当前阶段优先保证稳定、可落盘、可进入 MLflow。
    """

    fig, ax = plt.subplots(figsize=(8, 6))
    unique_targets = sorted(result_df["target"].astype(str).unique())
    for target_name in unique_targets:
        part = result_df[result_df["target"].astype(str) == target_name]
        ax.scatter(part["x"], part["y"], label=target_name, alpha=0.75, s=28)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def run_analysis(
    config: Dict[str, Any],
    target_series: pd.Series,
    feature_df: pd.DataFrame,
    paths: TaskPaths,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    执行 analysis 类实验。

    这里首版支持：
    - PCA
    - t-SNE
    - UMAP（若环境可用）
    """

    analysis_cfg = config.get("analysis", {})
    if not analysis_cfg.get("enabled", True):
        return {}

    # analysis 需要数值化输入，因此这里保守地只对原始特征做 one-hot，再做标准化。
    analysis_input = pd.get_dummies(feature_df, dummy_na=True)
    analysis_input = analysis_input.fillna(analysis_input.median(numeric_only=True)).fillna(0.0)
    analysis_input_values = StandardScaler().fit_transform(analysis_input)

    analysis_jobs: List[Tuple[str, bool]] = [
        ("pca", analysis_cfg.get("pca", {}).get("enabled", True)),
        ("tsne", analysis_cfg.get("tsne", {}).get("enabled", True)),
        ("umap", analysis_cfg.get("umap", {}).get("enabled", True)),
    ]

    visualization_rows: List[Dict[str, Any]] = []

    for name, enabled in analysis_jobs:
        if not enabled or is_experiment_completed(state, "analysis", name):
            continue

        state["current_stage"] = f"analysis.{name}"
        save_run_state(paths, state)

        explained_variance_sum = None
        method_params: Dict[str, Any] = {"method": name, "n_components": 2}
        if name == "pca":
            reducer = PCA(n_components=2, random_state=config["project"]["random_seed"])
            coords = reducer.fit_transform(analysis_input_values)
            explained_variance_sum = float(np.sum(reducer.explained_variance_ratio_))
        elif name == "tsne":
            tsne_cfg = analysis_cfg.get("tsne", {})
            method_params["perplexity"] = tsne_cfg.get("perplexity", 30)
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
            except Exception:
                continue
            umap_cfg = analysis_cfg.get("umap", {})
            method_params["n_neighbors"] = umap_cfg.get("n_neighbors", 15)
            method_params["min_dist"] = umap_cfg.get("min_dist", 0.1)
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=umap_cfg.get("n_neighbors", 15),
                min_dist=umap_cfg.get("min_dist", 0.1),
                random_state=config["project"]["random_seed"],
            )
            coords = reducer.fit_transform(analysis_input_values)

        result_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "target": target_series.astype(str).values})
        result_path = paths.analysis_dir / f"{name}_2d.csv"
        save_dataframe(result_path, result_df)
        plot_path = paths.plots_dir / f"{name}_scatter.png"
        save_analysis_scatter_plot(result_df, plot_path, title=f"{name.upper()} 2D Scatter")

        row = {
            "name": name,
            "result_path": str(result_path),
            "plot_path": str(plot_path),
            "point_count": int(len(result_df)),
            "target_class_count": int(result_df["target"].nunique()),
            **method_params,
        }
        if explained_variance_sum is not None:
            row["explained_variance_sum"] = explained_variance_sum
        visualization_rows.append(row)

        mark_experiment_completed(state, "analysis", name)
        save_run_state(paths, state)

    if visualization_rows:
        index_df = pd.DataFrame(visualization_rows)
        index_path = paths.analysis_dir / "visualization_index.csv"
        save_dataframe(index_path, index_df)
        return {
            "index_path": index_path,
            "rows": visualization_rows,
            "sample_count": int(len(feature_df)),
            "feature_count": int(feature_df.shape[1]),
            "analysis_input_feature_count": int(analysis_input.shape[1]),
        }
    return {}

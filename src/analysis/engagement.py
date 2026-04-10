"""
Task 2 – User Engagement Analysis
Covers: top-10 per metric, k-means (k=3), elbow method, cluster stats,
top-10 per app, top-3 apps visualization.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from loguru import logger
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
import config


class EngagementAnalysis:
    """All Task-2 analyses."""

    METRICS = ["sessions_frequency", "total_duration_ms", "total_traffic_bytes"]

    def __init__(self, engagement_df: pd.DataFrame, app_traffic_df: pd.DataFrame) -> None:
        self.eng = engagement_df.copy()
        self.app = app_traffic_df.copy()
        self._reports = config.REPORTS_DIR
        self._reports.mkdir(exist_ok=True)
        self._cluster_col = "engagement_cluster"
        self._scaler = MinMaxScaler()

    # ── 2.1  Top-10 per metric ────────────────────────────────────────────────

    def top10_per_metric(self) -> dict[str, pd.DataFrame]:
        """Return top-10 users per engagement metric."""
        result = {}
        for metric in self.METRICS:
            if metric in self.eng.columns:
                result[metric] = (
                    self.eng[[config.USER_ID_COL, metric]]
                    .sort_values(metric, ascending=False)
                    .head(10)
                    .reset_index(drop=True)
                )
        return result

    # ── 2.2  K-Means (k=3) ───────────────────────────────────────────────────

    def run_kmeans(self, k: int = config.ENGAGEMENT_K) -> pd.DataFrame:
        """Normalize metrics and run k-means clustering."""
        features = self._get_feature_matrix()
        scaled = self._scaler.fit_transform(features)

        km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
        labels = km.fit_predict(scaled)
        self.eng[self._cluster_col] = labels
        self._kmeans_model = km
        self._scaled_features = scaled
        logger.success(f"K-Means (k={k}) applied. Cluster sizes: "
                       f"{pd.Series(labels).value_counts().to_dict()}")
        return self.eng

    # ── 2.3  Cluster Statistics ───────────────────────────────────────────────

    def cluster_statistics(self) -> pd.DataFrame:
        """Min, max, avg, total (non-normalized) metrics per cluster."""
        if self._cluster_col not in self.eng.columns:
            self.run_kmeans()
        stats = (
            self.eng.groupby(self._cluster_col)[self.METRICS]
            .agg(["min", "max", "mean", "sum"])
        )
        stats.columns = ["_".join(c) for c in stats.columns]
        return stats.reset_index()

    def plot_clusters(self, save: bool = True) -> plt.Figure:
        """Scatter plot of engagement clusters (frequency vs traffic)."""
        if self._cluster_col not in self.eng.columns:
            self.run_kmeans()
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        pairs = [
            ("sessions_frequency", "total_traffic_bytes"),
            ("total_duration_ms",  "total_traffic_bytes"),
            ("sessions_frequency", "total_duration_ms"),
        ]
        colors = ["#e74c3c", "#2ecc71", "#3498db"]
        for ax, (x, y) in zip(axes, pairs):
            if x not in self.eng.columns or y not in self.eng.columns:
                continue
            for cluster in sorted(self.eng[self._cluster_col].unique()):
                mask = self.eng[self._cluster_col] == cluster
                ax.scatter(
                    self.eng.loc[mask, x],
                    self.eng.loc[mask, y],
                    s=10, alpha=0.5,
                    color=colors[cluster % len(colors)],
                    label=f"Cluster {cluster}",
                )
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(f"{x} vs {y}")
            ax.legend()
        plt.suptitle("Engagement Clusters (k=3)", fontsize=14)
        plt.tight_layout()
        if save:
            path = self._reports / "engagement_clusters.png"
            fig.savefig(path, dpi=120)
            logger.info(f"Saved → {path}")
        return fig

    # ── 2.4  Top-10 per application ───────────────────────────────────────────

    def top10_per_app(self) -> dict[str, pd.DataFrame]:
        """Top-10 most engaged users per application (by total bytes)."""
        result = {}
        app_cols = [c for c in self.app.columns
                    if c.endswith("_total_bytes") and c != config.USER_ID_COL]
        for col in app_cols:
            app_name = col.replace("_total_bytes", "")
            result[app_name] = (
                self.app[[config.USER_ID_COL, col]]
                .sort_values(col, ascending=False)
                .head(10)
                .reset_index(drop=True)
            )
        return result

    # ── 2.5  Top-3 apps chart ─────────────────────────────────────────────────

    def plot_top3_apps(self, save: bool = True) -> plt.Figure:
        """Bar chart of total traffic for the top-3 most used applications."""
        app_cols = [c for c in self.app.columns
                    if c.endswith("_total_bytes")]
        totals = {
            col.replace("_total_bytes", ""): self.app[col].sum()
            for col in app_cols
        }
        sorted_apps = sorted(totals.items(), key=lambda x: x[1], reverse=True)[:3]
        apps, values = zip(*sorted_apps)
        values_gb = [v / 1e9 for v in values]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(apps, values_gb, color=["#e74c3c", "#3498db", "#2ecc71"],
                      edgecolor="black")
        ax.bar_label(bars, fmt="%.2f GB", padding=3)
        ax.set_title("Top 3 Most Used Applications by Total Traffic")
        ax.set_ylabel("Total Traffic (GB)")
        ax.set_xlabel("Application")
        plt.tight_layout()
        if save:
            path = self._reports / "top3_apps.png"
            fig.savefig(path, dpi=120)
            logger.info(f"Saved → {path}")
        return fig

    # ── 2.6  Elbow Method ─────────────────────────────────────────────────────

    def elbow_method(self, max_k: int = config.ELBOW_MAX_K, save: bool = True) -> dict:
        """Compute inertia for k=1..max_k and plot the elbow curve."""
        features = self._get_feature_matrix()
        scaled = self._scaler.fit_transform(features)

        inertias = []
        k_range = range(1, max_k + 1)
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
            km.fit(scaled)
            inertias.append(km.inertia_)

        # Simple elbow detection (kneedle approximation)
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        optimal_k = int(np.argmax(diffs2) + 2)  # +2: 0-indexed + first diff shift

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(list(k_range), inertias, "bo-", linewidth=2, markersize=8)
        ax.axvline(x=optimal_k, color="red", linestyle="--",
                   label=f"Optimal k={optimal_k}")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Inertia (Within-cluster SSE)")
        ax.set_title("Elbow Method – Optimal k for Engagement Clustering")
        ax.legend()
        plt.tight_layout()
        if save:
            path = self._reports / "engagement_elbow.png"
            fig.savefig(path, dpi=120)
            logger.info(f"Saved → {path}")
        plt.close(fig)

        return {"inertias": inertias, "optimal_k": optimal_k}

    # ── 2.3b  Cluster distribution (user share & traffic share) ──────────────

    def cluster_distribution(self) -> pd.DataFrame:
        """Return per-cluster user count, traffic share, and session share."""
        if self._cluster_col not in self.eng.columns:
            self.run_kmeans()
        available = [m for m in self.METRICS if m in self.eng.columns]
        grp = self.eng.groupby(self._cluster_col)
        dist = pd.DataFrame({
            "user_count": grp.size(),
        })
        for metric in available:
            dist[f"{metric}_total"] = grp[metric].sum()
        dist = dist.reset_index()
        dist["user_pct"] = dist["user_count"] / dist["user_count"].sum() * 100
        for metric in available:
            col = f"{metric}_total"
            dist[f"{metric}_pct"] = dist[col] / dist[col].sum() * 100
        return dist

    # ── 2.3c  Classify clusters ───────────────────────────────────────────────

    def classify_clusters(self) -> dict[int, str]:
        """Return a mapping {cluster_id: label} of Low / Medium / High engagement."""
        if self._cluster_col not in self.eng.columns:
            self.run_kmeans()
        # Rank clusters by their composite mean (average of all three normalised metrics)
        available = [m for m in self.METRICS if m in self.eng.columns]
        cluster_means = (
            self.eng.groupby(self._cluster_col)[available].mean().mean(axis=1)
        )
        ranked = cluster_means.sort_values()
        labels = ["Low", "Medium", "High"]
        return {int(cluster): labels[rank] for rank, cluster in enumerate(ranked.index)}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_feature_matrix(self) -> np.ndarray:
        available = [m for m in self.METRICS if m in self.eng.columns]
        return self.eng[available].fillna(0).values

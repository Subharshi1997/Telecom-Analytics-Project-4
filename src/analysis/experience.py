"""
Task 3 – User Experience Analysis
Covers: TCP / RTT / Throughput top-10/bottom-10/most-frequent,
distribution plots per handset, k-means (k=3) on experience metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from loguru import logger
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
import config


class ExperienceAnalysis:
    """All Task-3 analyses."""

    NUMERIC_EXP_COLS = [
        "avg_tcp_retransmission", "avg_rtt_ms", "avg_throughput_kbps"
    ]

    def __init__(self, experience_df: pd.DataFrame) -> None:
        self.exp = experience_df.copy()
        self._reports = config.REPORTS_DIR
        self._reports.mkdir(exist_ok=True)
        self._cluster_col = "experience_cluster"

    # ── 3.2  Top / Bottom / Most Frequent ────────────────────────────────────

    def top_bottom_frequent(self, col: str, n: int = 10) -> dict[str, pd.Series]:
        """Return top-N, bottom-N, most-frequent-N values for a column."""
        if col not in self.exp.columns:
            raise KeyError(f"Column '{col}' not found in experience data.")
        series = self.exp[col].dropna()
        return {
            "top":        series.nlargest(n),
            "bottom":     series.nsmallest(n),
            "most_freq":  series.value_counts().head(n),
        }

    def experience_top_bottom_summary(self) -> dict[str, dict]:
        """Run top_bottom_frequent for TCP, RTT, and Throughput."""
        mapping = {
            "TCP Retransmission": "avg_tcp_retransmission",
            "RTT (ms)":           "avg_rtt_ms",
            "Throughput (kbps)":  "avg_throughput_kbps",
        }
        return {
            label: self.top_bottom_frequent(col)
            for label, col in mapping.items()
            if col in self.exp.columns
        }

    # ── 3.3a  Throughput per Handset ──────────────────────────────────────────

    def throughput_per_handset(self, top_n: int = 10, save: bool = True) -> pd.DataFrame:
        """Average throughput per handset type — top N (best performing)."""
        if "avg_throughput_kbps" not in self.exp.columns:
            raise KeyError("avg_throughput_kbps not in experience data")
        summary = (
            self.exp.groupby(config.HANDSET_TYPE_COL)["avg_throughput_kbps"]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(summary[config.HANDSET_TYPE_COL], summary["avg_throughput_kbps"],
                color="steelblue", edgecolor="black")
        ax.invert_yaxis()
        ax.set_xlabel("Avg Throughput (kbps)")
        ax.set_title(f"Average Throughput per Handset Type (Top {top_n})")
        plt.tight_layout()
        if save:
            path = self._reports / "throughput_per_handset.png"
            fig.savefig(path, dpi=120)
            logger.info(f"Saved → {path}")
        plt.close(fig)
        return summary

    def throughput_bottom_handsets(self, bottom_n: int = 10) -> pd.DataFrame:
        """Average throughput per handset type — bottom N (worst performing)."""
        if "avg_throughput_kbps" not in self.exp.columns:
            raise KeyError("avg_throughput_kbps not in experience data")
        return (
            self.exp.groupby(config.HANDSET_TYPE_COL)["avg_throughput_kbps"]
            .mean()
            .sort_values(ascending=True)
            .head(bottom_n)
            .reset_index()
        )

    # ── 3.3b  TCP Retransmission per Handset ──────────────────────────────────

    def tcp_per_handset(self, top_n: int = 10, save: bool = True) -> pd.DataFrame:
        """Average TCP retransmission per handset type — top N (worst = most retrans)."""
        if "avg_tcp_retransmission" not in self.exp.columns:
            raise KeyError("avg_tcp_retransmission not in experience data")
        summary = (
            self.exp.groupby(config.HANDSET_TYPE_COL)["avg_tcp_retransmission"]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(summary[config.HANDSET_TYPE_COL], summary["avg_tcp_retransmission"],
                color="coral", edgecolor="black")
        ax.invert_yaxis()
        ax.set_xlabel("Avg TCP Retransmission (Bytes)")
        ax.set_title(f"Average TCP Retransmission per Handset Type (Top {top_n})")
        plt.tight_layout()
        if save:
            path = self._reports / "tcp_per_handset.png"
            fig.savefig(path, dpi=120)
            logger.info(f"Saved → {path}")
        plt.close(fig)
        return summary

    def tcp_best_handsets(self, top_n: int = 10) -> pd.DataFrame:
        """Average TCP retransmission per handset type — bottom N (best = least retrans)."""
        if "avg_tcp_retransmission" not in self.exp.columns:
            raise KeyError("avg_tcp_retransmission not in experience data")
        return (
            self.exp.groupby(config.HANDSET_TYPE_COL)["avg_tcp_retransmission"]
            .mean()
            .sort_values(ascending=True)
            .head(top_n)
            .reset_index()
        )

    # ── 3.4  K-Means (k=3) ───────────────────────────────────────────────────

    def run_kmeans(self, k: int = config.EXPERIENCE_K) -> pd.DataFrame:
        """Normalize metrics and cluster users into k experience groups."""
        features = self._get_feature_matrix()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
        labels = km.fit_predict(scaled)
        self.exp[self._cluster_col] = labels
        self._kmeans_model = km
        self._scaler = scaler
        self._scaled_features = scaled
        logger.success(f"Experience K-Means (k={k}) applied. "
                       f"Cluster sizes: {pd.Series(labels).value_counts().to_dict()}")
        return self.exp

    def classify_clusters(self) -> dict[int, str]:
        """
        Return {cluster_id: label} using data-driven ranking on TCP retransmission.
        Lowest TCP  → 'Good'
        Middle TCP  → 'Average'
        Highest TCP → 'Poor'
        TCP is the primary quality signal: high retransmission = packet loss = bad experience.
        """
        if self._cluster_col not in self.exp.columns:
            self.run_kmeans()
        tcp_means = self.exp.groupby(self._cluster_col)["avg_tcp_retransmission"].mean()
        ranked = tcp_means.sort_values(ascending=True)
        labels = ["Good", "Average", "Poor"]
        return {int(cluster): labels[rank] for rank, cluster in enumerate(ranked.index)}

    def cluster_summary(self) -> pd.DataFrame:
        """Mean experience metrics per cluster."""
        if self._cluster_col not in self.exp.columns:
            self.run_kmeans()
        available = [c for c in self.NUMERIC_EXP_COLS if c in self.exp.columns]
        return (
            self.exp.groupby(self._cluster_col)[available]
            .mean()
            .reset_index()
        )

    def cluster_statistics(self) -> pd.DataFrame:
        """Min, max, mean, total (non-normalized) metrics per cluster."""
        if self._cluster_col not in self.exp.columns:
            self.run_kmeans()
        available = [c for c in self.NUMERIC_EXP_COLS if c in self.exp.columns]
        stats = (
            self.exp.groupby(self._cluster_col)[available]
            .agg(["min", "max", "mean", "sum"])
        )
        stats.columns = ["_".join(c) for c in stats.columns]
        return stats.reset_index()

    def cluster_distribution(self) -> pd.DataFrame:
        """Per-cluster user count, percentage share, and mean metrics."""
        if self._cluster_col not in self.exp.columns:
            self.run_kmeans()
        available = [c for c in self.NUMERIC_EXP_COLS if c in self.exp.columns]
        grp = self.exp.groupby(self._cluster_col)
        dist = pd.DataFrame({"user_count": grp.size()})
        for metric in available:
            dist[f"{metric}_mean"] = grp[metric].mean()
        dist = dist.reset_index()
        dist["user_pct"] = dist["user_count"] / dist["user_count"].sum() * 100
        return dist

    def plot_experience_clusters(self, save: bool = True) -> plt.Figure:
        """Parallel coordinates / box plots of experience metrics by cluster."""
        if self._cluster_col not in self.exp.columns:
            self.run_kmeans()
        available = [c for c in self.NUMERIC_EXP_COLS if c in self.exp.columns]
        fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
        if len(available) == 1:
            axes = [axes]
        for ax, col in zip(axes, available):
            self.exp.boxplot(column=col, by=self._cluster_col, ax=ax)
            ax.set_title(col)
            ax.set_xlabel("Cluster")
        plt.suptitle("Experience Clusters – Metric Distributions", fontsize=13)
        plt.tight_layout()
        if save:
            path = self._reports / "experience_clusters.png"
            fig.savefig(path, dpi=120)
            logger.info(f"Saved → {path}")
        return fig

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_feature_matrix(self) -> np.ndarray:
        available = [c for c in self.NUMERIC_EXP_COLS if c in self.exp.columns]
        return self.exp[available].fillna(0).values

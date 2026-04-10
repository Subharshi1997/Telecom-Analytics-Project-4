"""
Task 4 – User Satisfaction Analysis
Covers: engagement/experience scores (Euclidean distance), satisfaction score,
top-10 satisfied users, regression model, k-means (k=2), cluster aggregation,
MySQL export, MLflow tracking, cluster interpretation, model explanation.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge  # noqa: F401 – kept for import compatibility
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from loguru import logger
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
import config

try:
    import mlflow
    import mlflow.sklearn
    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MLFLOW_AVAILABLE = False
    logger.warning("mlflow not installed – model tracking will be skipped.")


class SatisfactionAnalysis:
    """All Task-4 analyses."""

    def __init__(
        self,
        engagement_df: pd.DataFrame,
        experience_df: pd.DataFrame,
    ) -> None:
        """
        Parameters
        ----------
        engagement_df : must contain `engagement_cluster` and engagement metrics.
        experience_df : must contain `experience_cluster` and experience metrics.
        """
        self.eng = engagement_df.copy()
        self.exp = experience_df.copy()
        self._uid = config.USER_ID_COL
        self._result: pd.DataFrame | None = None
        self._satisfaction_model = None

    # ── 4.1  Engagement & Experience Scores ──────────────────────────────────

    def compute_engagement_score(self) -> pd.Series:
        """
        Euclidean distance (on scaled features) from each user to the
        least-engaged cluster centroid.

        Scaling is applied via StandardScaler so that high-magnitude features
        (e.g. total_traffic_bytes in billions) do not dominate low-magnitude
        ones (e.g. sessions_frequency in tens).

        Requires `engagement_cluster` to exist in self.eng.
        """
        if "engagement_cluster" not in self.eng.columns:
            raise ValueError("Run EngagementAnalysis.run_kmeans() first.")

        metrics = ["sessions_frequency", "total_duration_ms", "total_traffic_bytes"]
        available = [m for m in metrics if m in self.eng.columns]
        raw_data = self.eng[available].fillna(0).values

        # ── FIX: scale before distance computation ───────────────────────────
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(raw_data)

        # Least engaged = cluster with smallest mean total_traffic (raw scale)
        traffic_col = (
            "total_traffic_bytes" if "total_traffic_bytes" in self.eng.columns
            else available[0]
        )
        least_engaged_cluster = (
            self.eng.groupby("engagement_cluster")[traffic_col].mean().idxmin()
        )

        # Centroid computed on raw values, then scaled with the same scaler
        raw_centroid = (
            self.eng[self.eng["engagement_cluster"] == least_engaged_cluster][available]
            .mean()
            .values
            .reshape(1, -1)
        )
        scaled_centroid = scaler.transform(raw_centroid).flatten()
        # ────────────────────────────────────────────────────────────────────

        scores = np.linalg.norm(scaled_data - scaled_centroid, axis=1)
        logger.info(
            f"Engagement scores computed on {len(available)} scaled features "
            f"(least-engaged cluster: {least_engaged_cluster})."
        )
        return pd.Series(scores, index=self.eng.index, name="engagement_score")

    def compute_experience_score(self) -> pd.Series:
        """
        Euclidean distance (on scaled features) from each user to the
        worst-experience cluster centroid.

        Scaling prevents avg_throughput_kbps (order of magnitude ~1000) from
        eclipsing avg_tcp_retransmission (order of magnitude ~1).

        Requires `experience_cluster` to exist in self.exp.
        """
        if "experience_cluster" not in self.exp.columns:
            raise ValueError("Run ExperienceAnalysis.run_kmeans() first.")

        exp_metrics = ["avg_tcp_retransmission", "avg_rtt_ms", "avg_throughput_kbps"]
        available = [m for m in exp_metrics if m in self.exp.columns]
        raw_data = self.exp[available].fillna(0).values

        # ── FIX: scale before distance computation ───────────────────────────
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(raw_data)

        # Worst experience = highest avg TCP retrans (raw scale)
        tcp_col = (
            "avg_tcp_retransmission" if "avg_tcp_retransmission" in self.exp.columns
            else available[0]
        )
        worst_cluster = (
            self.exp.groupby("experience_cluster")[tcp_col].mean().idxmax()
        )

        raw_centroid = (
            self.exp[self.exp["experience_cluster"] == worst_cluster][available]
            .mean()
            .values
            .reshape(1, -1)
        )
        scaled_centroid = scaler.transform(raw_centroid).flatten()
        # ────────────────────────────────────────────────────────────────────

        scores = np.linalg.norm(scaled_data - scaled_centroid, axis=1)
        logger.info(
            f"Experience scores computed on {len(available)} scaled features "
            f"(worst-experience cluster: {worst_cluster})."
        )
        return pd.Series(scores, index=self.exp.index, name="experience_score")

    # ── 4.2  Satisfaction Score & Top-10 ─────────────────────────────────────

    def build_satisfaction_table(self) -> pd.DataFrame:
        """
        Merge engagement & experience data, compute scores, and build
        the full satisfaction table per user.
        """
        eng_scores = self.compute_engagement_score()
        self.eng["engagement_score"] = eng_scores

        exp_scores = self.compute_experience_score()
        self.exp["experience_score"] = exp_scores

        # Merge on MSISDN
        merged = pd.merge(
            self.eng[[self._uid, "engagement_score"]],
            self.exp[[self._uid, "experience_score"]],
            on=self._uid,
            how="inner",
        )
        merged["satisfaction_score"] = (
            merged["engagement_score"] + merged["experience_score"]
        ) / 2

        self._result = merged
        logger.success(f"Satisfaction table built: {merged.shape}")
        return merged

    def top10_satisfied(self) -> pd.DataFrame:
        """
        Return the top-10 users ranked by satisfaction score.

        Business Interpretation
        -----------------------
        These users represent the highest-value segment of the customer base:
        - They are highly engaged (frequent sessions, high data usage).
        - They enjoy superior network experience (low retransmission, low RTT,
          high throughput).
        - Retention priority: churn by even one of these users carries
          disproportionate revenue risk.
        - Recommended action: enrol them in loyalty / VIP programmes and use
          their usage patterns as the benchmark for network quality targets.
        """
        if self._result is None:
            self.build_satisfaction_table()

        top10 = (
            self._result[
                [self._uid, "engagement_score", "experience_score", "satisfaction_score"]
            ]
            .sort_values("satisfaction_score", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )

        logger.info(
            "Top-10 satisfied users identified. "
            "These are VIP retention targets — high engagement AND high experience."
        )
        return top10

    # ── 4.3  Regression Model ─────────────────────────────────────────────────

    def train_satisfaction_model(self) -> dict:
        """
        Train a Gradient Boosting regressor to predict satisfaction score and
        log the run to MLflow (Task 4.7).

        Returns
        -------
        dict with keys: mse, rmse, r2, model, training_time_s, run_id
        """
        if self._result is None:
            self.build_satisfaction_table()

        features = ["engagement_score", "experience_score"]
        X = self._result[features].values
        y = self._result["satisfaction_score"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=config.RANDOM_STATE
        )

        params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": config.RANDOM_STATE,
        }

        model = GradientBoostingRegressor(**params)

        t0 = time.time()
        model.fit(X_train, y_train)
        training_time = round(time.time() - t0, 4)

        y_pred = model.predict(X_test)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        r2   = float(r2_score(y_test, y_pred))

        self._satisfaction_model = model

        # ── 4.7  MLflow Tracking ──────────────────────────────────────────────
        run_id: str | None = None
        if _MLFLOW_AVAILABLE:
            try:
                mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
                mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

                with mlflow.start_run(run_name="GradientBoosting_satisfaction") as run:
                    run_id = run.info.run_id

                    # Parameters
                    mlflow.log_param("model_name", "GradientBoostingRegressor")
                    mlflow.log_param("n_estimators",  params["n_estimators"])
                    mlflow.log_param("learning_rate", params["learning_rate"])
                    mlflow.log_param("max_depth",     params["max_depth"])

                    # Metrics
                    mlflow.log_metric("mse",              mse)
                    mlflow.log_metric("rmse",             rmse)
                    mlflow.log_metric("r2",               r2)
                    mlflow.log_metric("training_time_s",  training_time)

                    # Model artifact
                    mlflow.sklearn.log_model(model, artifact_path="model")

                logger.success(
                    f"MLflow run logged – id: {run_id} | "
                    f"RMSE: {rmse:.4f} | R²: {r2:.4f} | "
                    f"train time: {training_time}s"
                )
            except Exception as exc:  # pragma: no cover
                logger.warning(f"MLflow logging failed (non-fatal): {exc}")
        # ─────────────────────────────────────────────────────────────────────

        logger.success(
            f"Satisfaction model – RMSE: {rmse:.4f} | R²: {r2:.4f} | "
            f"Training time: {training_time}s"
        )
        return {
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "model": model,
            "training_time_s": training_time,
            "run_id": run_id,
        }

    # ── 4.4  K-Means (k=2) on Scores ─────────────────────────────────────────

    def kmeans_on_scores(self, k: int = config.SATISFACTION_K) -> pd.DataFrame:
        """K-means clustering on engagement & experience scores."""
        if self._result is None:
            self.build_satisfaction_table()

        X = self._result[["engagement_score", "experience_score"]].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
        self._result["satisfaction_cluster"] = km.fit_predict(X_scaled)
        logger.info(f"Satisfaction K-Means (k={k}) applied.")
        return self._result

    # ── 4.5  Cluster Aggregation ──────────────────────────────────────────────

    def cluster_aggregation(self) -> pd.DataFrame:
        """Average satisfaction & experience score per satisfaction cluster."""
        if self._result is None or "satisfaction_cluster" not in self._result.columns:
            self.kmeans_on_scores()
        return (
            self._result
            .groupby("satisfaction_cluster")[
                ["engagement_score", "experience_score", "satisfaction_score"]
            ]
            .mean()
            .reset_index()
        )

    def describe_satisfaction_clusters(self) -> pd.DataFrame:
        """
        Attach human-readable business labels to each satisfaction cluster and
        provide actionable insights for each segment.

        Labels are assigned by ranking clusters on their mean satisfaction_score:
        - Highest mean  → "High Satisfaction Users"
        - Middle (if k=3) → "Moderate Users"
        - Lowest mean   → "Low Satisfaction Users"

        Returns
        -------
        pd.DataFrame with columns:
            satisfaction_cluster, engagement_score, experience_score,
            satisfaction_score, label, insight
        """
        agg = self.cluster_aggregation().copy()

        n_clusters = len(agg)
        agg_sorted = agg.sort_values("satisfaction_score").reset_index(drop=True)

        # Build ordered label list (low → high)
        if n_clusters == 2:
            ordered_labels = ["Low Satisfaction Users", "High Satisfaction Users"]
        elif n_clusters == 3:
            ordered_labels = [
                "Low Satisfaction Users",
                "Moderate Users",
                "High Satisfaction Users",
            ]
        else:
            ordered_labels = [
                f"Tier-{i+1} Users (Low → High)" for i in range(n_clusters)
            ]

        insight_map = {
            "Low Satisfaction Users": (
                "Low engagement AND poor network experience. "
                "Immediate intervention required: investigate coverage gaps, "
                "reduce TCP retransmission rates, and proactively reach out "
                "with service-improvement offers to prevent churn."
            ),
            "Moderate Users": (
                "Average engagement with acceptable experience. "
                "Growth potential exists — targeted up-sell campaigns and "
                "personalised data plan recommendations can elevate this "
                "segment toward high satisfaction."
            ),
            "High Satisfaction Users": (
                "High engagement AND superior network experience. "
                "These are VIP customers: high data consumers with low latency "
                "and minimal packet loss. Protect with loyalty programmes, "
                "early access to new services, and dedicated support channels."
            ),
        }

        agg_sorted["label"] = ordered_labels
        agg_sorted["insight"] = agg_sorted["label"].map(
            lambda lbl: insight_map.get(
                lbl,
                "Mixed engagement and experience profile — monitor closely.",
            )
        )

        # Re-merge back on original cluster IDs so callers can join easily
        result = agg.merge(
            agg_sorted[["satisfaction_cluster", "label", "insight"]],
            on="satisfaction_cluster",
            how="left",
        )

        logger.info("Satisfaction cluster descriptions generated.")
        return result

    # ── 4.6  MySQL Export ─────────────────────────────────────────────────────

    def export_to_mysql(self) -> bool:
        """Export satisfaction table to MySQL as `user_scores`. Returns True on success."""
        if self._result is None:
            self.build_satisfaction_table()

        # Rename MSISDN/Number → customer_id to match assignment schema
        export_df = self._result[
            [self._uid, "engagement_score", "experience_score", "satisfaction_score"]
        ].rename(columns={self._uid: "customer_id"})

        from src.database.mysql_connector import MySQLConnector
        connector = MySQLConnector()
        return connector.export_dataframe(export_df, table_name="user_scores")

    def generate_sql_report_instructions(self) -> str:
        """
        Return step-by-step instructions for querying the exported satisfaction
        data and capturing the required screenshot for the final report.

        Returns
        -------
        str – formatted instruction block ready to paste into a report appendix.
        """
        instructions = """
========================================================
  Task 4.6 – MySQL Export: Report Instructions
========================================================

STEP 1 – Verify the export
  Connect to the MySQL database and confirm the table exists:

      SHOW TABLES LIKE 'user_satisfaction';

STEP 2 – Preview the data (REQUIRED screenshot)
  Run the following query and capture a full screenshot of the result set:

      SELECT * FROM user_satisfaction LIMIT 10;

  The screenshot must show:
    • The query in the SQL editor / terminal
    • The result grid with all columns visible:
        MSISDN/Number | engagement_score | experience_score | satisfaction_score
    • At least 10 rows of data
    • The row count / execution time shown by the client (e.g. MySQL Workbench footer)

STEP 3 – Optional deeper queries for the report
  Top-10 most satisfied users:
      SELECT * FROM user_satisfaction
      ORDER BY satisfaction_score DESC
      LIMIT 10;

  Cluster distribution (if satisfaction_cluster column was exported):
      SELECT satisfaction_cluster,
             COUNT(*)            AS user_count,
             AVG(satisfaction_score) AS avg_score
      FROM user_satisfaction
      GROUP BY satisfaction_cluster
      ORDER BY avg_score DESC;

STEP 4 – Include in final report
  Insert the screenshot under Section 4.6 – MySQL Export with the caption:
    "Figure X: First 10 rows of the user_satisfaction table exported to MySQL."

========================================================
"""
        return instructions.strip()

    # ── 4.7  Model Explanation ────────────────────────────────────────────────

    def explain_model_results(self) -> str:
        """
        Explain why the Gradient Boosting model achieves a near-perfect R²
        score on the satisfaction prediction task.

        Returns
        -------
        str – a clear, concise explanation suitable for a technical report.
        """
        explanation = """
========================================================
  Task 4.3 – Satisfaction Model: Result Explanation
========================================================

WHY THE MODEL ACHIEVES NEAR-PERFECT R²
---------------------------------------
The satisfaction score is defined as:

    satisfaction_score = (engagement_score + experience_score) / 2

The model is trained to predict satisfaction_score using engagement_score
and experience_score as the only features.

Because the target variable is a deterministic linear combination of the
exact same two input features, the Gradient Boosting regressor — or any
reasonably capable model — can learn this relationship almost perfectly.

In other words:
  • There is NO noise introduced by external factors (demographics, churn
    signals, billing issues) that are absent from the feature set.
  • The relationship is algebraically exact, not statistical.

WHAT THIS MEANS
---------------
  • R² ≈ 1.0 does NOT indicate overfitting in the traditional sense.
  • It does indicate that the modelling task, as currently defined, is
    trivial from a machine-learning perspective.

RECOMMENDATIONS FOR A MORE MEANINGFUL MODEL
--------------------------------------------
  1. Add independent features such as:
       - Handset type / manufacturer
       - Subscription plan / ARPU
       - Number of complaint tickets
       - Network cell-tower load
  2. Replace the algebraic satisfaction definition with an independently
     collected ground-truth label (e.g. NPS survey score).
  3. Evaluate on a truly held-out time period (walk-forward validation)
     to measure real predictive power.

CURRENT UTILITY
---------------
The model still serves a valid purpose:
  • It validates that the pipeline is end-to-end correct.
  • It can score NEW users (without a satisfaction label) as long as their
    engagement and experience scores are available.
  • The MLflow run provides a reproducible baseline for future experiments.

========================================================
"""
        return explanation.strip()

    # ── Getter ────────────────────────────────────────────────────────────────

    @property
    def satisfaction_table(self) -> pd.DataFrame:
        if self._result is None:
            self.build_satisfaction_table()
        return self._result

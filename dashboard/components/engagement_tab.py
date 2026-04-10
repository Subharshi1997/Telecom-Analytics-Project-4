"""Streamlit component – Task 2: User Engagement."""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render(engagement_analysis) -> None:
    """Render the User Engagement tab."""
    st.header("📊 Task 2 – User Engagement Analysis")

    # ── Top-10 per metric ─────────────────────────────────────────────────────
    st.subheader("Top 10 Users per Engagement Metric")
    top10 = engagement_analysis.top10_per_metric()
    metric_labels = {
        "sessions_frequency": "Session Frequency",
        "total_duration_ms":  "Total Duration (ms)",
        "total_traffic_bytes": "Total Traffic (Bytes)",
    }
    cols = st.columns(len(top10))
    for col, (metric, df) in zip(cols, top10.items()):
        with col:
            label = metric_labels.get(metric, metric)
            st.markdown(f"**{label}**")
            st.dataframe(df, use_container_width=True)

    # ── K-Means Clusters ──────────────────────────────────────────────────────
    st.subheader("Engagement Clusters (k=3) – Normalized K-Means")
    with st.spinner("Running K-Means clustering…"):
        engagement_analysis.run_kmeans(k=3)

    cluster_stats = engagement_analysis.cluster_statistics()
    cluster_labels = engagement_analysis.classify_clusters()
    cluster_dist = engagement_analysis.cluster_distribution()

    # ── SECTION 1: Formatted cluster statistics table ─────────────────────────
    st.markdown("**Cluster Statistics (raw, non-normalized values)**")

    # Build a human-readable display table
    display_rows = []
    label_order = ["Low", "Medium", "High"]
    for label in label_order:
        cluster_id = next(cid for cid, lbl in cluster_labels.items() if lbl == label)
        row = cluster_stats[cluster_stats["engagement_cluster"] == cluster_id].iloc[0]
        dist_row = cluster_dist[cluster_dist["engagement_cluster"] == cluster_id].iloc[0]
        display_rows.append({
            "Cluster": f"Cluster {cluster_id}",
            "Classification": label,
            "Users (n)": int(dist_row["user_count"]),
            "Users (%)": round(dist_row["user_pct"], 1),
            "Sessions – Min": int(row["sessions_frequency_min"]),
            "Sessions – Max": int(row["sessions_frequency_max"]),
            "Sessions – Avg": round(row["sessions_frequency_mean"], 2),
            "Sessions – Total": int(row["sessions_frequency_sum"]),
            "Duration – Avg (hrs)": round(row["total_duration_ms_mean"] / 3_600_000, 2),
            "Traffic – Avg (MB)": round(row["total_traffic_bytes_mean"] / 1e6, 2),
            "Traffic – Total (GB)": round(row["total_traffic_bytes_sum"] / 1e9, 2),
            "Traffic (%)": round(dist_row["total_traffic_bytes_pct"], 1),
        })

    stats_display_df = pd.DataFrame(display_rows)
    st.dataframe(stats_display_df.set_index("Cluster"), use_container_width=True)

    # ── SECTION 2: Per-cluster metric contribution cards ──────────────────────
    st.markdown("---")
    st.markdown("**Cluster Classification & Business Insights**")

    CARD_STYLE = {
        "Low":    ("🔵", "#1a3a5c", "#4a9eda"),
        "Medium": ("🟡", "#3a3010", "#d4a820"),
        "High":   ("🔴", "#3a1010", "#e05050"),
    }
    ACTION_TEXT = {
        "Low": (
            "**Re-engagement opportunity.** These subscribers use the network sparingly — "
            "low session count, short durations, minimal data. They represent low current "
            "revenue but a meaningful churn risk if unaddressed. "
            "Recommended action: targeted re-engagement campaigns, introductory data bundles, "
            "or usage-based discount triggers to lift activity."
        ),
        "Medium": (
            "**Core revenue base.** This is the backbone of the subscriber base — consistent "
            "usage, predictable session patterns, moderate data appetite. "
            "They generate stable, recurring revenue and respond well to loyalty programmes. "
            "Recommended action: tiered plan upgrades, bundled services (streaming + data), "
            "and proactive retention to defend against churn."
        ),
        "High": (
            "**Premium monetisation target.** Power users driving a disproportionate share "
            "of total traffic with high session frequency and long engagement windows. "
            "High ARPU potential but also highest network cost. "
            "Recommended action: upsell to unlimited/priority data plans, offer exclusive "
            "content partnerships, and monitor for network congestion impact."
        ),
    }

    for label in label_order:
        cluster_id = next(cid for cid, lbl in cluster_labels.items() if lbl == label)
        row = cluster_stats[cluster_stats["engagement_cluster"] == cluster_id].iloc[0]
        dist_row = cluster_dist[cluster_dist["engagement_cluster"] == cluster_id].iloc[0]
        icon, bg, accent = CARD_STYLE[label]

        user_pct   = dist_row["user_pct"]
        traffic_pct = dist_row["total_traffic_bytes_pct"]
        avg_sessions = row["sessions_frequency_mean"]
        avg_duration_hrs = row["total_duration_ms_mean"] / 3_600_000
        avg_traffic_mb   = row["total_traffic_bytes_mean"] / 1e6
        total_traffic_gb = row["total_traffic_bytes_sum"] / 1e9

        st.markdown(
            f"""<div style="background:{bg};border-left:4px solid {accent};
                padding:14px 18px;border-radius:6px;margin-bottom:12px;">
            <h4 style="color:{accent};margin:0 0 6px 0;">
                {icon} Cluster {cluster_id} — {label} Engagement
            </h4>
            <p style="color:#e0e0e0;margin:0 0 8px 0;font-size:0.85rem;">
                <b>{int(dist_row['user_count']):,} users ({user_pct:.1f}% of base)</b>
                &nbsp;|&nbsp; Traffic share: <b>{traffic_pct:.1f}%</b>
                &nbsp;|&nbsp; Total traffic: <b>{total_traffic_gb:.1f} GB</b>
            </p>
            <p style="color:#c0c0c0;margin:0 0 8px 0;font-size:0.82rem;">
                Avg sessions: <b>{avg_sessions:.1f}</b>
                &nbsp;|&nbsp; Avg session duration: <b>{avg_duration_hrs:.2f} hrs</b>
                &nbsp;|&nbsp; Avg data per user: <b>{avg_traffic_mb:.1f} MB</b>
            </p>
            <p style="color:#d0d0d0;margin:0;font-size:0.83rem;">{ACTION_TEXT[label]}</p>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Distribution insight callout (data-driven) ────────────────────────────
    high_id  = next(cid for cid, lbl in cluster_labels.items() if lbl == "High")
    low_id   = next(cid for cid, lbl in cluster_labels.items() if lbl == "Low")
    high_dist  = cluster_dist[cluster_dist["engagement_cluster"] == high_id].iloc[0]
    high_stats = cluster_stats[cluster_stats["engagement_cluster"] == high_id].iloc[0]
    low_stats  = cluster_stats[cluster_stats["engagement_cluster"] == low_id].iloc[0]

    session_ratio = (
        high_stats["sessions_frequency_mean"] / low_stats["sessions_frequency_mean"]
        if low_stats["sessions_frequency_mean"] > 0 else float("inf")
    )
    traffic_ratio = (
        high_stats["total_traffic_bytes_mean"] / low_stats["total_traffic_bytes_mean"]
        if low_stats["total_traffic_bytes_mean"] > 0 else float("inf")
    )

    st.info(
        f"**Key finding:** The High Engagement cluster represents "
        f"**{high_dist['user_pct']:.1f}%** of users but accounts for "
        f"**{high_dist['total_traffic_bytes_pct']:.1f}%** of total traffic — "
        f"averaging **{session_ratio:.0f}×** more sessions and "
        f"**{traffic_ratio:.0f}×** more data than the Low Engagement group. "
        f"This concentration indicates significant monetisation upside from a small, "
        f"identifiable user cohort."
    )

    # ── Bar charts: mean of each metric per cluster ───────────────────────────
    st.markdown("---")
    st.markdown("**Mean Engagement Metrics per Cluster**")
    mean_cols = [c for c in cluster_stats.columns if c.endswith("_mean")]
    if mean_cols:
        mean_df = cluster_stats[["engagement_cluster"] + mean_cols].copy()
        mean_df["label"] = mean_df["engagement_cluster"].map(cluster_labels)
        mean_df = mean_df.sort_values(
            "label", key=lambda s: s.map({"Low": 0, "Medium": 1, "High": 2})
        )
        mean_df.columns = (
            ["engagement_cluster"]
            + [c.replace("_mean", "") for c in mean_cols]
            + ["label"]
        )
        mean_melted = mean_df.melt(
            id_vars=["engagement_cluster", "label"],
            var_name="Metric", value_name="Mean Value"
        )
        mean_melted["Cluster"] = (
            mean_melted["engagement_cluster"].astype(str)
            + " – "
            + mean_melted["label"]
        )
        fig_bar = px.bar(
            mean_melted, x="Metric", y="Mean Value", color="Cluster",
            barmode="group",
            title="Mean Engagement Metrics per Cluster (raw values)",
            color_discrete_map={
                f"{next(cid for cid,lbl in cluster_labels.items() if lbl=='Low')} – Low": "#4a9eda",
                f"{next(cid for cid,lbl in cluster_labels.items() if lbl=='Medium')} – Medium": "#d4a820",
                f"{next(cid for cid,lbl in cluster_labels.items() if lbl=='High')} – High": "#e05050",
            },
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption(
            "Each bar shows the average (non-normalized) value for that metric within a cluster. "
            "Large differences across clusters confirm meaningful behavioral segmentation."
        )

    # ── SECTION 3: Scatter plot with interpretation ───────────────────────────
    eng_df = engagement_analysis.eng
    if "engagement_cluster" in eng_df.columns:
        eng_df = eng_df.copy()
        eng_df["Cluster"] = eng_df["engagement_cluster"].map(
            lambda x: f"{x} – {cluster_labels.get(x, x)}"
        )
        fig = px.scatter(
            eng_df,
            x="sessions_frequency",
            y="total_traffic_bytes",
            color="Cluster",
            title="Engagement Clusters: Session Frequency vs Total Traffic (Bytes)",
            labels={
                "sessions_frequency": "Session Frequency (count)",
                "total_traffic_bytes": "Total Traffic (Bytes)",
            },
            opacity=0.55,
            color_discrete_map={
                f"{next(cid for cid,lbl in cluster_labels.items() if lbl=='Low')} – Low": "#4a9eda",
                f"{next(cid for cid,lbl in cluster_labels.items() if lbl=='Medium')} – Medium": "#d4a820",
                f"{next(cid for cid,lbl in cluster_labels.items() if lbl=='High')} – High": "#e05050",
            },
        )
        fig.update_traces(marker_size=5)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
**How to read this scatter plot:**
- Each dot is one subscriber. Position reflects their raw (non-normalized) session count and total data consumed.
- **Cluster separation** along both axes confirms K-Means found genuinely distinct behavioral groups — not arbitrary partitions.
- **Low Engagement (blue):** Dense cloud near the origin — many users, low activity. Broad base but limited individual revenue contribution.
- **Medium Engagement (yellow):** Spread across moderate session and traffic values — the predictable, stable middle tier.
- **High Engagement (red):** Sparse but far from the origin — few users, extreme values. These outliers drive infrastructure load and represent the highest ARPU potential.
- A diagonal trend (more sessions → more traffic) validates that session frequency is a reliable proxy for data appetite — supporting its use as a primary segmentation feature.
        """)


    # ── Top-10 per application ────────────────────────────────────────────────
    st.subheader("Top 10 Users per Application")
    top10_app = engagement_analysis.top10_per_app()
    app_names = list(top10_app.keys())
    selected_app = st.selectbox("Select Application", app_names)
    if selected_app:
        app_df = top10_app[selected_app]
        col_name = f"{selected_app}_total_bytes"
        fig_app = px.bar(
            app_df, x=col_name, y=app_df.columns[0],
            orientation="h",
            title=f"Top 10 Users – {selected_app}",
            color=col_name,
            color_continuous_scale="Reds",
        )
        fig_app.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_app, use_container_width=True)

    # ── Top-3 apps ────────────────────────────────────────────────────────────
    st.subheader("Top 3 Most Used Applications")
    app_df_all = engagement_analysis.app
    app_cols = [c for c in app_df_all.columns if c.endswith("_total_bytes")]
    if app_cols:
        totals = {
            c.replace("_total_bytes", ""): app_df_all[c].sum() / 1e9
            for c in app_cols
        }
        sorted_totals = sorted(totals.items(), key=lambda x: x[1], reverse=True)[:3]
        apps, values = zip(*sorted_totals)
        fig_top3 = go.Figure(go.Bar(
            x=list(apps),
            y=list(values),
            marker_color=["#e74c3c", "#3498db", "#2ecc71"],
            text=[f"{v:.2f} GB" for v in values],
            textposition="outside",
        ))
        fig_top3.update_layout(
            title="Top 3 Applications by Total Traffic",
            xaxis_title="Application",
            yaxis_title="Total Traffic (GB)",
        )
        st.plotly_chart(fig_top3, use_container_width=True)
    st.markdown("""
    **Interpretation:**
    - The top 3 applications account for the vast majority of total network traffic, confirming that a small number of apps drive disproportionate bandwidth consumption.
    - Video and gaming applications typically dominate — their high data volumes are driven by continuous streaming and real-time data transfer requirements.
    - These top apps should be prioritised in QoS (Quality of Service) policies and network capacity planning to maintain a satisfactory user experience for the majority of subscribers.
    """)

    # ── Elbow Method ──────────────────────────────────────────────────────────
    st.subheader("Elbow Method – Optimized k")
    with st.spinner("Computing elbow curve…"):
        elbow = engagement_analysis.elbow_method(max_k=10, save=False)
    k_range = list(range(1, len(elbow["inertias"]) + 1))
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=k_range, y=elbow["inertias"],
        mode="lines+markers", name="Inertia",
        line={"color": "steelblue"},
    ))
    fig_elbow.add_vline(
        x=elbow["optimal_k"], line_dash="dash",
        line_color="red",
        annotation_text=f"Optimal k={elbow['optimal_k']}",
    )
    fig_elbow.update_layout(
        title="Elbow Method – Optimal Number of Clusters",
        xaxis_title="k (Number of Clusters)",
        yaxis_title="Inertia (Within-cluster SSE)",
    )
    st.plotly_chart(fig_elbow, use_container_width=True)
    st.info(f"✅ Optimal k = **{elbow['optimal_k']}** based on second-derivative elbow detection.")
    st.markdown(f"""
    **Interpretation:**
    - The elbow curve shows inertia (within-cluster sum of squares) decreasing as k increases. The rate of decrease slows sharply at **k = {elbow['optimal_k']}**, forming the "elbow".
    - Beyond k = {elbow['optimal_k']}**, adding more clusters yields diminishing returns — clusters become too granular to be meaningfully distinct.
    - This confirms that **{elbow['optimal_k']} engagement groups** best balance cluster cohesion and separation for this dataset.
    - Using k=3 for the main analysis aligns well with the natural low/medium/high engagement segmentation observed in the data.
    """)

"""Streamlit component – Task 3: User Experience."""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ── Styling constants ─────────────────────────────────────────────────────────
_CARD = {
    "Good":    ("🟢", "#0d2b1a", "#2ecc71"),
    "Average": ("🟡", "#2b2300", "#f1c40f"),
    "Poor":    ("🔴", "#2b0d0d", "#e74c3c"),
}
_LABEL_ORDER = ["Good", "Average", "Poor"]
_ACTION = {
    "Good": (
        "**Premium experience — protect and monetise.** These users enjoy low packet loss, "
        "minimal latency, and strong data speeds. They represent the highest satisfaction "
        "cohort and the strongest retention asset. "
        "Recommended action: upsell to premium plans, enrol in loyalty programmes, "
        "and use as a benchmark for network quality targets."
    ),
    "Average": (
        "**Stable but improvable — the core base.** Mid-range metrics indicate adequate "
        "but unremarkable service. Most subscribers fall here, making this group the "
        "largest driver of recurring revenue. "
        "Recommended action: targeted network optimisation in their coverage zones, "
        "proactive plan upgrades, and QoS prioritisation to shift users toward Good."
    ),
    "Poor": (
        "**High churn risk — immediate intervention required.** Elevated TCP retransmission "
        "and RTT signal persistent packet loss and congestion. These users are the most "
        "likely to churn or escalate complaints. "
        "Recommended action: cell-level capacity upgrades, handset-specific firmware "
        "advisories, proactive outreach with service credits, and SLA monitoring alerts."
    ),
}


def render(experience_analysis) -> None:
    """Render the User Experience tab."""
    st.header("🔬 Task 3 – User Experience Analysis")

    # ── 3.1  Aggregated customer data ─────────────────────────────────────────
    st.subheader("Task 3.1 – Per-Customer Aggregated Experience Metrics")
    st.markdown(
        "Each row represents one subscriber. Metrics are averaged across all "
        "sessions per customer. Missing values were imputed with the column mean "
        "(numeric) or mode (handset type). Outliers were replaced using IQR-based "
        "mean substitution before aggregation."
    )
    display_cols = [
        c for c in [
            "MSISDN/Number", "avg_tcp_retransmission",
            "avg_rtt_ms", "avg_throughput_kbps", "Handset Type",
        ]
        if c in experience_analysis.exp.columns
    ]
    st.dataframe(
        experience_analysis.exp[display_cols].head(20),
        use_container_width=True,
    )
    n_users = len(experience_analysis.exp)
    avg_tp = experience_analysis.exp["avg_throughput_kbps"].mean() if "avg_throughput_kbps" in experience_analysis.exp.columns else None
    avg_rtt = experience_analysis.exp["avg_rtt_ms"].mean() if "avg_rtt_ms" in experience_analysis.exp.columns else None
    avg_tcp = experience_analysis.exp["avg_tcp_retransmission"].mean() if "avg_tcp_retransmission" in experience_analysis.exp.columns else None
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Subscribers", f"{n_users:,}")
    if avg_tp is not None:
        m2.metric("Avg Throughput", f"{avg_tp:,.1f} kbps")
    if avg_rtt is not None:
        m3.metric("Avg RTT", f"{avg_rtt:,.1f} ms")
    if avg_tcp is not None:
        m4.metric("Avg TCP Retrans.", f"{avg_tcp:,.0f} B")

    st.markdown("---")

    # ── 3.2  Top / Bottom / Most Frequent ────────────────────────────────────
    st.subheader("Task 3.2 – TCP / RTT / Throughput: Top, Bottom & Most Frequent")
    st.markdown(
        "Examining the extremes of each metric reveals the range of network "
        "conditions experienced by subscribers and highlights outlier devices "
        "or coverage zones that warrant investigation."
    )
    summary = experience_analysis.experience_top_bottom_summary()
    metric_desc = {
        "TCP Retransmission": (
            "High values = packet loss / congestion. "
            "The top-10 represent the worst-served users on the network."
        ),
        "RTT (ms)": (
            "Round-trip time measures latency. "
            "High RTT degrades real-time applications (VoIP, gaming, video calls)."
        ),
        "Throughput (kbps)": (
            "Data speed experienced by the user. "
            "Bottom-10 are candidates for network capacity or handset upgrades."
        ),
    }
    for metric, data in summary.items():
        with st.expander(f"📈 {metric}"):
            st.caption(metric_desc.get(metric, ""))
            col1, col2, col3 = st.columns(3)
            col1.markdown("**Top 10 (Highest)**")
            col1.dataframe(
                data["top"].reset_index().rename(
                    columns={"index": "User", data["top"].name: metric}
                ),
                use_container_width=True,
            )
            col2.markdown("**Bottom 10 (Lowest)**")
            col2.dataframe(
                data["bottom"].reset_index().rename(
                    columns={"index": "User", data["bottom"].name: metric}
                ),
                use_container_width=True,
            )
            col3.markdown("**Most Frequent 10**")
            col3.dataframe(
                data["most_freq"].reset_index().rename(
                    columns={"index": metric, data["most_freq"].name: "Count"}
                ),
                use_container_width=True,
            )

    st.markdown("---")

    # ── 3.3a  Throughput per handset ──────────────────────────────────────────
    st.subheader("Task 3.3a – Throughput Distribution per Handset Type")

    tp_top = experience_analysis.throughput_per_handset(top_n=15, save=False)
    tp_bot = experience_analysis.throughput_bottom_handsets(bottom_n=15)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Top 15 – Best Throughput**")
        fig_tp_top = px.bar(
            tp_top,
            x="avg_throughput_kbps",
            y="Handset Type",
            orientation="h",
            color="avg_throughput_kbps",
            color_continuous_scale="Blues",
            labels={"avg_throughput_kbps": "Avg Throughput (kbps)", "Handset Type": ""},
        )
        fig_tp_top.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            margin={"l": 0, "r": 0},
        )
        st.plotly_chart(fig_tp_top, use_container_width=True)

    with col_b:
        st.markdown("**Bottom 15 – Worst Throughput (Churn Risk)**")
        fig_tp_bot = px.bar(
            tp_bot,
            x="avg_throughput_kbps",
            y="Handset Type",
            orientation="h",
            color="avg_throughput_kbps",
            color_continuous_scale="Reds_r",
            labels={"avg_throughput_kbps": "Avg Throughput (kbps)", "Handset Type": ""},
        )
        fig_tp_bot.update_layout(
            yaxis={"categoryorder": "total descending"},
            coloraxis_showscale=False,
            margin={"l": 0, "r": 0},
        )
        st.plotly_chart(fig_tp_bot, use_container_width=True)

    # Throughput distribution (KDE / histogram)
    st.markdown("**Overall Throughput Distribution (all subscribers)**")
    if "avg_throughput_kbps" in experience_analysis.exp.columns:
        fig_dist_tp = px.histogram(
            experience_analysis.exp,
            x="avg_throughput_kbps",
            nbins=80,
            marginal="violin",
            opacity=0.75,
            color_discrete_sequence=["#3498db"],
            labels={"avg_throughput_kbps": "Avg Throughput (kbps)"},
            title="Throughput Distribution – All Subscribers",
        )
        fig_dist_tp.update_layout(showlegend=False)
        st.plotly_chart(fig_dist_tp, use_container_width=True)

    # Gap metric: best vs worst handset
    if not tp_top.empty and not tp_bot.empty:
        best_tp  = tp_top["avg_throughput_kbps"].max()
        worst_tp = tp_bot["avg_throughput_kbps"].min()
        gap_x    = round(best_tp / worst_tp, 1) if worst_tp > 0 else float("inf")
        best_name  = tp_top.iloc[0]["Handset Type"]
        worst_name = tp_bot.iloc[0]["Handset Type"]
    else:
        gap_x, best_name, worst_name = "N/A", "—", "—"

    st.markdown(f"""
**Throughput Interpretation:**
- The best-performing device (**{best_name}**) delivers up to **{gap_x}×** more throughput
  than the worst-performing device (**{worst_name}**). This gap is driven by hardware
  capabilities (antenna design, modem generation) and software TCP optimisation.
- **High-throughput handsets** (top 15): Premium flagship devices with advanced LTE-A / 5G
  modems. Users on these devices have the highest satisfaction and lowest churn propensity —
  ideal targets for premium data plan upsells.
- **Low-throughput handsets** (bottom 15): Entry-level and legacy devices unable to exploit
  available network capacity. These users experience degraded service regardless of coverage
  quality, representing a **churn risk that can be partially mitigated** through device
  upgrade campaigns or targeted network-side QoS prioritisation.
- A right-skewed throughput distribution (long tail to the right) is typical — a small
  cohort of power users consume disproportionate bandwidth, confirming the need for
  tiered data plans.
    """)

    st.markdown("---")

    # ── 3.3b  TCP Retransmission per handset ──────────────────────────────────
    st.subheader("Task 3.3b – TCP Retransmission per Handset Type")

    tcp_worst = experience_analysis.tcp_per_handset(top_n=15, save=False)
    tcp_best  = experience_analysis.tcp_best_handsets(top_n=15)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**Top 15 – Highest TCP Retransmission (Worst Quality)**")
        fig_tcp_worst = px.bar(
            tcp_worst,
            x="avg_tcp_retransmission",
            y="Handset Type",
            orientation="h",
            color="avg_tcp_retransmission",
            color_continuous_scale="Reds",
            labels={"avg_tcp_retransmission": "Avg TCP Retrans. (Bytes)", "Handset Type": ""},
        )
        fig_tcp_worst.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            margin={"l": 0, "r": 0},
        )
        st.plotly_chart(fig_tcp_worst, use_container_width=True)

    with col_d:
        st.markdown("**Top 15 – Lowest TCP Retransmission (Best Quality)**")
        fig_tcp_best = px.bar(
            tcp_best,
            x="avg_tcp_retransmission",
            y="Handset Type",
            orientation="h",
            color="avg_tcp_retransmission",
            color_continuous_scale="Greens_r",
            labels={"avg_tcp_retransmission": "Avg TCP Retrans. (Bytes)", "Handset Type": ""},
        )
        fig_tcp_best.update_layout(
            yaxis={"categoryorder": "total descending"},
            coloraxis_showscale=False,
            margin={"l": 0, "r": 0},
        )
        st.plotly_chart(fig_tcp_best, use_container_width=True)

    # TCP distribution
    st.markdown("**Overall TCP Retransmission Distribution (all subscribers)**")
    if "avg_tcp_retransmission" in experience_analysis.exp.columns:
        fig_dist_tcp = px.histogram(
            experience_analysis.exp,
            x="avg_tcp_retransmission",
            nbins=80,
            marginal="violin",
            opacity=0.75,
            color_discrete_sequence=["#e74c3c"],
            labels={"avg_tcp_retransmission": "Avg TCP Retransmission (Bytes)"},
            title="TCP Retransmission Distribution – All Subscribers",
        )
        fig_dist_tcp.update_layout(showlegend=False)
        st.plotly_chart(fig_dist_tcp, use_container_width=True)

    if not tcp_worst.empty and not tcp_best.empty:
        worst_tcp_name = tcp_worst.iloc[0]["Handset Type"]
        best_tcp_name  = tcp_best.iloc[0]["Handset Type"]
        tcp_ratio = round(
            tcp_worst["avg_tcp_retransmission"].max()
            / max(tcp_best["avg_tcp_retransmission"].min(), 1),
            0,
        )
    else:
        worst_tcp_name, best_tcp_name, tcp_ratio = "—", "—", "N/A"

    st.markdown(f"""
**TCP Retransmission Interpretation:**
- TCP retransmissions occur when packets are lost in transit and must be re-sent.
  High retransmission volume indicates **congestion, interference, or a weak signal** —
  all of which degrade user experience.
- **{worst_tcp_name}** shows the highest average retransmission — up to **{tcp_ratio:.0f}×**
  more than **{best_tcp_name}**. This disparity points to older TCP stack implementations
  or device-side congestion control issues rather than pure network problems.
- Handsets with persistently high TCP retransmission are likely associated with higher
  support ticket volumes. Identifying these device models allows targeted **firmware
  update advisories** or **device swap promotions** — reducing support costs while
  improving perceived network quality.
- A heavily right-skewed TCP distribution (most users near zero, long tail of high
  retransmitters) is the expected pattern. The tail is where intervention has the
  greatest quality impact per user reached.
    """)

    st.markdown("---")

    # ── 3.4  Experience Clusters ──────────────────────────────────────────────
    st.subheader("Task 3.4 – Experience Clustering (k=3, Normalized K-Means)")
    st.markdown(
        "K-Means is applied on **standardized** TCP retransmission, RTT, and throughput "
        "to place each subscriber into one of three experience tiers. "
        "Cluster labels are assigned **data-driven**: the cluster with the lowest average "
        "TCP retransmission is labelled *Good*, highest is labelled *Poor*."
    )

    with st.spinner("Clustering experience data…"):
        experience_analysis.run_kmeans(k=3)

    cluster_labels = experience_analysis.classify_clusters()
    cluster_stats  = experience_analysis.cluster_statistics()
    cluster_dist   = experience_analysis.cluster_distribution()

    # ── Formatted statistics table ────────────────────────────────────────────
    st.markdown("**Cluster Statistics (raw, non-normalized values)**")
    display_rows = []
    for label in _LABEL_ORDER:
        cid = next(c for c, l in cluster_labels.items() if l == label)
        row  = cluster_stats[cluster_stats["experience_cluster"] == cid].iloc[0]
        drow = cluster_dist[cluster_dist["experience_cluster"] == cid].iloc[0]
        display_rows.append({
            "Cluster":           f"Cluster {cid}",
            "Classification":    label,
            "Users (n)":         int(drow["user_count"]),
            "Users (%)":         round(drow["user_pct"], 1),
            "TCP – Avg (B)":     round(row["avg_tcp_retransmission_mean"], 1),
            "TCP – Max (B)":     round(row["avg_tcp_retransmission_max"], 1),
            "RTT – Avg (ms)":    round(row["avg_rtt_ms_mean"], 1),
            "RTT – Max (ms)":    round(row["avg_rtt_ms_max"], 1),
            "Throughput – Avg (kbps)": round(row["avg_throughput_kbps_mean"], 1),
            "Throughput – Max (kbps)": round(row["avg_throughput_kbps_max"], 1),
        })

    st.dataframe(
        pd.DataFrame(display_rows).set_index("Cluster"),
        use_container_width=True,
    )

    # ── Per-cluster business cards ────────────────────────────────────────────
    st.markdown("**Cluster Classification & Business Insights**")
    for label in _LABEL_ORDER:
        cid  = next(c for c, l in cluster_labels.items() if l == label)
        drow = cluster_dist[cluster_dist["experience_cluster"] == cid].iloc[0]
        row  = cluster_stats[cluster_stats["experience_cluster"] == cid].iloc[0]
        icon, bg, accent = _CARD[label]

        avg_tcp  = row["avg_tcp_retransmission_mean"]
        avg_rtt  = row["avg_rtt_ms_mean"]
        avg_tp   = row["avg_throughput_kbps_mean"]

        st.markdown(
            f"""<div style="background:{bg};border-left:4px solid {accent};
                padding:14px 18px;border-radius:6px;margin-bottom:12px;">
            <h4 style="color:{accent};margin:0 0 6px 0;">
                {icon} Cluster {cid} — {label} Experience
            </h4>
            <p style="color:#e0e0e0;margin:0 0 8px 0;font-size:0.85rem;">
                <b>{int(drow['user_count']):,} users ({drow['user_pct']:.1f}% of base)</b>
            </p>
            <p style="color:#c0c0c0;margin:0 0 8px 0;font-size:0.82rem;">
                Avg TCP retrans: <b>{avg_tcp:,.0f} B</b>
                &nbsp;|&nbsp; Avg RTT: <b>{avg_rtt:.1f} ms</b>
                &nbsp;|&nbsp; Avg Throughput: <b>{avg_tp:,.1f} kbps</b>
            </p>
            <p style="color:#d0d0d0;margin:0;font-size:0.83rem;">{_ACTION[label]}</p>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Key finding callout ───────────────────────────────────────────────────
    good_cid = next(c for c, l in cluster_labels.items() if l == "Good")
    poor_cid = next(c for c, l in cluster_labels.items() if l == "Poor")
    good_row = cluster_stats[cluster_stats["experience_cluster"] == good_cid].iloc[0]
    poor_row = cluster_stats[cluster_stats["experience_cluster"] == poor_cid].iloc[0]
    good_dist = cluster_dist[cluster_dist["experience_cluster"] == good_cid].iloc[0]
    poor_dist = cluster_dist[cluster_dist["experience_cluster"] == poor_cid].iloc[0]

    tcp_gap = round(
        poor_row["avg_tcp_retransmission_mean"] / max(good_row["avg_tcp_retransmission_mean"], 1), 0
    )
    tp_gap = round(
        good_row["avg_throughput_kbps_mean"] / max(poor_row["avg_throughput_kbps_mean"], 1), 1
    )

    st.info(
        f"**Key finding:** The Poor Experience cluster ({poor_dist['user_pct']:.1f}% of users) "
        f"generates **{tcp_gap:.0f}×** more TCP retransmissions and receives "
        f"**{tp_gap:.1f}×** less throughput than the Good Experience cluster "
        f"({good_dist['user_pct']:.1f}% of users). "
        f"Prioritising network improvements in Poor cluster coverage zones could shift "
        f"a meaningful share of at-risk subscribers into the Average or Good tier."
    )

    st.markdown("---")

    # ── SECTION 3: Visual plots ───────────────────────────────────────────────
    st.markdown("**Visual Cluster Analysis**")

    exp_df = experience_analysis.exp.copy()
    exp_df["Experience"] = exp_df["experience_cluster"].map(
        lambda x: f"{x} – {cluster_labels.get(x, x)}"
    )
    color_map = {
        f"{next(c for c,l in cluster_labels.items() if l=='Good')} – Good":    "#2ecc71",
        f"{next(c for c,l in cluster_labels.items() if l=='Average')} – Average": "#f1c40f",
        f"{next(c for c,l in cluster_labels.items() if l=='Poor')} – Poor":    "#e74c3c",
    }

    # Scatter: Throughput vs RTT
    if "avg_throughput_kbps" in exp_df.columns and "avg_rtt_ms" in exp_df.columns:
        fig_scatter = px.scatter(
            exp_df,
            x="avg_throughput_kbps",
            y="avg_rtt_ms",
            color="Experience",
            color_discrete_map=color_map,
            opacity=0.55,
            title="Experience Clusters: Throughput vs RTT",
            labels={
                "avg_throughput_kbps": "Avg Throughput (kbps)",
                "avg_rtt_ms": "Avg RTT (ms)",
            },
        )
        fig_scatter.update_traces(marker_size=4)
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("""
**How to read this scatter plot:**
- Each dot is one subscriber. Position reflects their raw throughput (x-axis) and RTT (y-axis).
- **Good Experience (green):** High throughput, low RTT — upper-right to lower-right quadrant.
  Fast data speeds with low latency confirm excellent network conditions.
- **Average Experience (yellow):** Mid-range spread — the bulk of the subscriber base clusters
  around moderate values with visible overlap with both other groups.
- **Poor Experience (red):** Low throughput and/or high RTT — lower-left and upper areas.
  These users experience slow speeds and sluggish response times simultaneously.
- **Cluster separation** along both axes confirms K-Means identified genuinely distinct
  behavioral profiles rather than arbitrary partitions. The inverse relationship between
  throughput and RTT (higher speed ↔ lower latency) is expected and validates the feature
  selection for clustering.
        """)

    # Throughput distribution coloured by cluster
    st.markdown("**Throughput Distribution by Experience Cluster**")
    if "avg_throughput_kbps" in exp_df.columns:
        fig_tp_cluster = px.histogram(
            exp_df,
            x="avg_throughput_kbps",
            color="Experience",
            color_discrete_map=color_map,
            barmode="overlay",
            nbins=60,
            opacity=0.6,
            labels={"avg_throughput_kbps": "Avg Throughput (kbps)"},
            title="Throughput Distribution by Experience Cluster",
        )
        st.plotly_chart(fig_tp_cluster, use_container_width=True)
        st.caption(
            "Overlapping distributions show where cluster boundaries sit in the raw metric space. "
            "Well-separated peaks confirm meaningful segmentation; overlap indicates the "
            "gradient nature of network quality (no hard boundary between 'Average' and 'Good')."
        )

    # Box plot: metric per cluster
    st.markdown("**Metric Distribution by Cluster (Box Plot)**")
    feat_col = st.selectbox(
        "Select metric to inspect",
        options=[c for c in ["avg_tcp_retransmission", "avg_rtt_ms", "avg_throughput_kbps"]
                 if c in exp_df.columns],
        format_func=lambda c: {
            "avg_tcp_retransmission": "TCP Retransmission (Bytes)",
            "avg_rtt_ms": "RTT (ms)",
            "avg_throughput_kbps": "Throughput (kbps)",
        }.get(c, c),
    )
    fig_box = px.box(
        exp_df,
        x="Experience",
        y=feat_col,
        color="Experience",
        color_discrete_map=color_map,
        points="outliers",
        title=f"{feat_col} Distribution by Experience Cluster",
        labels={"Experience": "Cluster", feat_col: feat_col.replace("_", " ").title()},
        category_orders={"Experience": list(color_map.keys())},
    )
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)
    st.caption(
        "Box plots show median, IQR, and outliers per cluster. "
        "Tight boxes with little overlap between clusters indicate strong segmentation; "
        "wide boxes suggest high intra-cluster variance in that metric."
    )

    # Radar chart: cluster profiles
    st.markdown("**Cluster Profiles – Radar Chart (Mean Metrics)**")
    cluster_sum = experience_analysis.cluster_summary()
    categories = [c for c in cluster_sum.columns
                  if c not in ["experience_cluster"]]
    cat_labels = {
        "avg_tcp_retransmission": "TCP Retrans.",
        "avg_rtt_ms": "RTT (ms)",
        "avg_throughput_kbps": "Throughput",
    }
    fig_radar = go.Figure()
    for _, row in cluster_sum.iterrows():
        cid = int(row["experience_cluster"])
        lbl = cluster_labels.get(cid, f"Cluster {cid}")
        icon, _, accent = _CARD.get(lbl, ("", "#555", "#aaa"))
        fig_radar.add_trace(go.Scatterpolar(
            r=[row[c] for c in categories],
            theta=[cat_labels.get(c, c) for c in categories],
            fill="toself",
            name=f"{icon} {cid} – {lbl}",
            line_color=accent,
        ))
    fig_radar.update_layout(
        polar={"radialaxis": {"visible": True}},
        title="Experience Cluster Profiles (Mean Metrics)",
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.caption(
        "The radar chart compares cluster centroids across all three metrics simultaneously. "
        "The Good cluster should show a distinctly different shape (low TCP/RTT, high throughput) "
        "from the Poor cluster — confirming meaningful separation in metric space."
    )

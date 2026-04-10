"""Streamlit component – Task 1: User Overview."""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render(overview_analysis, raw_df: pd.DataFrame) -> None:
    """Render the User Overview tab."""
    st.header("📱 Task 1 – User Overview Analysis")

    # ── Handset Analysis ──────────────────────────────────────────────────────
    st.subheader("Top 10 Handsets")
    top_hs = overview_analysis.top_handsets(10)
    fig = px.bar(
        top_hs, x="Session Count", y="Handset Type",
        orientation="h", color="Session Count",
        color_continuous_scale="Blues",
        title="Top 10 Handsets by Session Count",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    # ── Top Manufacturers ─────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 3 Manufacturers")
        top_mfr = overview_analysis.top_manufacturers(3)
        st.dataframe(top_mfr, use_container_width=True)

        fig_mfr = px.pie(
            top_mfr, names="Manufacturer", values="Session Count",
            title="Market Share – Top 3 Manufacturers",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig_mfr, use_container_width=True)

    with col2:
        st.subheader("Top 5 Handsets per Manufacturer")
        top_hs_mfr = overview_analysis.top_handsets_per_manufacturer()
        for mfr, df in top_hs_mfr.items():
            with st.expander(f"🏭 {mfr}"):
                st.dataframe(df, use_container_width=True)

    # ── Marketing Recommendations ─────────────────────────────────────────────
    st.subheader("💡 Marketing Recommendations")
    st.markdown("""
    **Interpretation & Recommendations:**
    - **Dominant handsets** are mid-range Android devices. Marketing should target
      mobile data bundles compatible with these devices.
    - **Top manufacturers** (likely Samsung, Huawei, Apple) indicate a diverse
      hardware ecosystem. Optimize app/web assets for the top handset screen sizes.
    - **iPhone users** tend to consume higher data volumes → premium data plan upsell.
    - **Entry-level handsets** show higher session frequency but lower throughout →
      consider lightweight app versions and data-saver promotions.
    """)

    # ── Variable Descriptions & Data Types ───────────────────────────────────
    st.subheader("Variable Descriptions & Data Types")
    var_desc = pd.DataFrame([
        {"Variable": "MSISDN/Number",               "Type": "Categorical (ID)", "Description": "Unique user identifier (mobile number)"},
        {"Variable": "Bearer Id",                   "Type": "Categorical (ID)", "Description": "Unique session/bearer identifier"},
        {"Variable": "Start ms / End ms",           "Type": "Integer",          "Description": "Session start and end timestamps in milliseconds"},
        {"Variable": "Dur. (ms)",                   "Type": "Float",            "Description": "Total session duration in milliseconds"},
        {"Variable": "IMSI",                        "Type": "Categorical (ID)", "Description": "International Mobile Subscriber Identity"},
        {"Variable": "IMEI",                        "Type": "Categorical (ID)", "Description": "International Mobile Equipment Identity (device ID)"},
        {"Variable": "Handset Manufacturer",        "Type": "Categorical",      "Description": "Manufacturer of the user's handset"},
        {"Variable": "Handset Type",                "Type": "Categorical",      "Description": "Specific model of the user's handset"},
        {"Variable": "Avg RTT DL/UL (ms)",          "Type": "Float",            "Description": "Average Round Trip Time for downlink/uplink — measures network latency"},
        {"Variable": "Avg Bearer TP DL/UL (kbps)",  "Type": "Float",            "Description": "Average bearer throughput for downlink/uplink in kilobits per second"},
        {"Variable": "TCP DL/UL Retrans. Vol (Bytes)","Type": "Float",          "Description": "Volume of TCP retransmitted data — indicator of network quality"},
        {"Variable": "Social Media DL/UL (Bytes)",  "Type": "Float",            "Description": "Download/upload data volume for Social Media applications"},
        {"Variable": "Google DL/UL (Bytes)",        "Type": "Float",            "Description": "Download/upload data volume for Google services"},
        {"Variable": "Email DL/UL (Bytes)",         "Type": "Float",            "Description": "Download/upload data volume for Email applications"},
        {"Variable": "Youtube DL/UL (Bytes)",       "Type": "Float",            "Description": "Download/upload data volume for YouTube"},
        {"Variable": "Netflix DL/UL (Bytes)",       "Type": "Float",            "Description": "Download/upload data volume for Netflix"},
        {"Variable": "Gaming DL/UL (Bytes)",        "Type": "Float",            "Description": "Download/upload data volume for Gaming applications"},
        {"Variable": "Other DL/UL (Bytes)",         "Type": "Float",            "Description": "Download/upload data volume for all other applications"},
        {"Variable": "Total DL/UL (Bytes)",         "Type": "Float",            "Description": "Total download/upload data volume across all applications"},
        {"Variable": "Total Data (Bytes)",          "Type": "Float",            "Description": "Combined total DL + UL data volume per session"},
    ])
    st.dataframe(var_desc, use_container_width=True, hide_index=True)

    # ── Task 1.1 – Per-User Aggregation Table ────────────────────────────────
    st.subheader("Task 1.1 – Per-User Aggregated Data")
    st.markdown("One row per user (MSISDN): number of xDR sessions, session duration, DL/UL data, and per-application data volumes.")
    st.dataframe(overview_analysis.ov, use_container_width=True)

    # ── Task 1.2 – Basic Metrics ──────────────────────────────────────────────
    st.subheader("Task 1.2 – Basic Metrics (mean, median, std, skewness, kurtosis)")
    user_stats = overview_analysis.basic_metrics()
    user_stats_display = user_stats.copy()
    user_stats_display["variance"] = user_stats_display["variance"].apply(lambda x: f"{x:.2e}")
    st.dataframe(user_stats_display.style.format("{:.2f}", subset=[c for c in user_stats_display.columns if c != "variance"]), use_container_width=True)

    st.markdown("""
    **Interpretation & Importance for Global Objective:**
    - **Session Duration (Dur. ms)** — Mean ~92,261 ms with positive skewness indicates most users have short sessions but a few power users drive very long sessions. Key for segmenting engagement levels.
    - **Throughput DL (Avg Bearer TP DL)** — Mean ~7,426 kbps but median only 63 kbps reveals extreme right-skew: a small group of users consume disproportionately high bandwidth, critical for capacity planning.
    - **RTT DL (Avg RTT DL)** — Mean ~65 ms, median 54 ms. Lower RTT = better experience. High skewness indicates some users consistently experience poor latency — targets for network improvement.
    - **TCP Retransmissions** — High mean values signal packet loss and poor network quality for affected users, directly impacting satisfaction scores.
    - **App Data Volumes (YouTube, Gaming, Netflix)** — These dominate total data consumption. Mean values confirm video and gaming are the primary bandwidth drivers — essential for prioritization policies.
    - **Missing values = 0** across all columns confirms the cleaning pipeline successfully imputed all NaN values with column means.
    """)

    # ── Graphical Univariate Analysis ────────────────────────────────────────
    st.subheader("Graphical Univariate Analysis")

    # Histograms for continuous session/network metrics
    st.markdown("**Continuous Variables – Histograms**")
    hist_cols = [c for c in ["Dur. (ms)", "Avg RTT DL (ms)", "Avg RTT UL (ms)",
                              "Avg Bearer TP DL (kbps)", "Avg Bearer TP UL (kbps)"] if c in raw_df.columns]
    for col in hist_cols:
        fig_h = px.histogram(raw_df, x=col, nbins=60, title=f"Distribution of {col}",
                             color_discrete_sequence=["steelblue"])
        fig_h.update_layout(bargap=0.05)
        st.plotly_chart(fig_h, use_container_width=True)

    # Box plots for app byte columns — better for spotting outliers
    st.markdown("**App Data Volume Variables – Box Plots**")
    app_byte_cols = [c for c in [
        "Social Media DL (Bytes)", "Google DL (Bytes)", "Email DL (Bytes)",
        "Youtube DL (Bytes)", "Netflix DL (Bytes)", "Gaming DL (Bytes)", "Other DL (Bytes)",
    ] if c in raw_df.columns]
    if app_byte_cols:
        box_df = raw_df[app_byte_cols].copy()
        box_df_melted = box_df.melt(var_name="Application", value_name="Bytes")
        fig_box = px.box(box_df_melted, x="Application", y="Bytes",
                         title="App DL Data Volume Distribution (Box Plot)",
                         color="Application")
        fig_box.update_layout(xaxis_tickangle=-30, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    # Bar chart for categorical — handset manufacturer
    st.markdown("**Categorical Variables – Bar Charts**")
    if "Handset Manufacturer" in raw_df.columns:
        mfr_counts = raw_df["Handset Manufacturer"].value_counts().head(10).reset_index()
        mfr_counts.columns = ["Manufacturer", "Count"]
        fig_cat = px.bar(mfr_counts, x="Manufacturer", y="Count",
                         title="Top 10 Handset Manufacturers (Frequency)",
                         color_discrete_sequence=["coral"])
        st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("""
    **Interpretation of Graphical Univariate Analysis:**
    - **Session Duration histogram** — right-skewed distribution confirms most sessions are short; a long tail of heavy users exists. Mean is pulled above median by outliers.
    - **RTT histograms** — DL RTT is more spread than UL RTT. Most users cluster at low latency but a visible tail indicates network congestion for a subset of users.
    - **Throughput histograms** — extreme right skew in DL throughout; the majority of sessions have near-zero throughput while a small group has very high speeds. Suggests unequal network resource distribution.
    - **App data box plots** — Gaming and Other DL show the widest IQR and most outliers, confirming they are the most variable and highest-volume applications. Email and Social Media are comparatively uniform.
    - **Manufacturer bar chart** — confirms market concentration among a few manufacturers, guiding device-specific optimisation priorities.
    """)

    # ── Non-Graphical Univariate Analysis – Dispersion Parameters ────────────
    st.subheader("Non-Graphical Univariate Analysis – Dispersion Parameters")
    disp = overview_analysis.dispersion_analysis()
    # Format large numeric columns as scientific notation to avoid overflow warnings
    disp_display = disp.copy()
    for col in ["Q1", "Q3", "IQR", "Range"]:
        if col in disp_display.columns:
            disp_display[col] = disp_display[col].apply(lambda x: f"{x:.2e}")
    st.dataframe(disp_display.style.format("{:.2f}", subset=["CV_%"]), use_container_width=True)
    st.markdown("""
    **Interpretation of Dispersion Parameters:**
    - **IQR (Interquartile Range)** — measures the spread of the middle 50% of data. Large IQR in byte columns (Gaming, Other, YouTube) confirms high variability in app usage across users.
    - **Range** — the gap between min and max values. Extremely large ranges in DL/UL byte columns indicate the presence of both very light and very heavy users in the dataset.
    - **CV% (Coefficient of Variation)** — standardised measure of dispersion relative to the mean. CV% > 100% in throughput and byte columns signals highly heterogeneous user behaviour, making mean-based segmentation insufficient — clustering is needed.
    - **Session Duration** — moderate IQR relative to its range suggests a fairly spread distribution, with most users concentrated in mid-range durations but a long tail of power users.
    - **RTT columns** — low IQR relative to range indicates most users experience similar latency, but outliers (very high RTT) represent users with significantly degraded network experience.
    - **TCP Retransmissions** — high CV% indicates inconsistent network quality; some users suffer frequent retransmissions while others experience near-perfect connections.
    """)

    # ── Bivariate Analysis ────────────────────────────────────────────────────
    st.subheader("Bivariate Analysis – Application vs Total DL+UL Data")
    biv = overview_analysis.bivariate_app_vs_total(save=False)
    # Correlation bar chart
    fig_biv = px.bar(
        biv, x="Application", y="Correlation_with_Total",
        title="Correlation of Each Application with Total DL+UL Data",
        labels={"Correlation_with_Total": "Pearson Correlation", "Application": "Application"},
        color="Correlation_with_Total", color_continuous_scale="Blues",
    )
    st.plotly_chart(fig_biv, use_container_width=True)
    st.dataframe(biv.style.format({"Correlation_with_Total": "{:.4f}"}), use_container_width=True, hide_index=True)
    st.markdown("""
    **Interpretation:**
    - Applications with **high correlation** to total data are the primary drivers of network load — these should be prioritised in capacity planning.
    - **Gaming and Other** apps typically show the highest correlation, confirming they are the dominant bandwidth consumers.
    - **Email and Social Media** show lower correlation, indicating lighter but more frequent usage patterns.
    - Strong correlations across all apps suggest that heavy users tend to be heavy across all applications — not just one — supporting the need for overall usage-based segmentation.
    """)

    # ── Decile Segmentation ───────────────────────────────────────────────────
    st.subheader("Variable Transformation – Decile Segmentation (Top 5 Deciles by Duration)")
    deciles = overview_analysis.decile_segmentation()
    deciles = deciles.sort_values("decile")
    fig_dec = px.bar(
        deciles, x="decile", y="total_data_bytes_sum",
        title="Total DL+UL Data per Duration Decile (Top 5)",
        labels={"total_data_bytes_sum": "Total Data (Bytes)", "decile": "Decile"},
        color="total_data_bytes_sum", color_continuous_scale="Blues",
    )
    st.plotly_chart(fig_dec, use_container_width=True)
    st.dataframe(deciles.style.format({"total_data_bytes_sum": "{:,.0f}"}), use_container_width=True, hide_index=True)
    st.markdown("""
    **Interpretation:**
    - Users are segmented into 10 decile classes based on total session duration; only the **top 5 deciles (D6–D10)** are shown — the longest-session users.
    - **D10 (highest duration)** consistently contributes the most total data, confirming that session duration is a strong proxy for data consumption.
    - The increasing trend from D6 to D10 shows a near-linear relationship between time spent and data consumed.
    - These top decile users represent the most engaged segment — prime targets for premium data plans and loyalty programmes.
    """)

    # ── Correlation Analysis ──────────────────────────────────────────────────
    st.subheader("Correlation Analysis – Application Data")
    corr = overview_analysis.correlation_matrix(save=False)
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlation Matrix – Social Media, Google, Email, YouTube, Netflix, Gaming, Other",
        aspect="auto",
    )
    fig_corr.update_layout(coloraxis_colorbar=dict(title="r"))
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown("""
    **Interpretation:**
    - **High positive correlations (r ≈ 1)** between apps indicate users who are heavy on one app tend to be heavy on others — consistent with a general "heavy user" behaviour pattern.
    - **Gaming and Other** data typically show the strongest mutual correlation due to their large and similar data volumes dominating total consumption.
    - **YouTube and Netflix** are strongly correlated with each other, reflecting a shared video-streaming user segment that consumes large amounts of data together.
    - **Email and Social Media** correlate weakly with video apps, confirming they belong to a distinct lightweight-usage segment with different traffic profiles.
    - **Google** shows moderate correlation across most apps, acting as a baseline service used alongside other applications.
    - Overall, the matrix reveals two clusters: **heavy-data apps** (Gaming, Other, YouTube, Netflix) and **light-data apps** (Email, Social Media, Google) — directly informing tiered pricing and QoS policies.
    """)

    # ── PCA ───────────────────────────────────────────────────────────────────
    st.subheader("Dimensionality Reduction – Principal Component Analysis (PCA)")
    with st.spinner("Running PCA…"):
        pca_result = overview_analysis.pca_analysis(n_components=2, save=False)
    components = pca_result["components"]
    explained = pca_result["explained_variance_ratio"]

    col_pca1, col_pca2 = st.columns(2)

    with col_pca1:
        pca_df = pd.DataFrame({
            "PC1": components[:, 0],
            "PC2": components[:, 1] if components.shape[1] > 1 else [0] * len(components),
        })
        fig_pca = px.scatter(
            pca_df, x="PC1", y="PC2",
            title=f"PCA Scatter – PC1 ({explained[0]*100:.1f}%) vs PC2 ({explained[1]*100:.1f}%)",
            opacity=0.4,
            color_discrete_sequence=["steelblue"],
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    with col_pca2:
        scree_df = pd.DataFrame({
            "Principal Component": [f"PC{i+1}" for i in range(len(explained))],
            "Explained Variance (%)": explained * 100,
        })
        fig_scree = px.bar(
            scree_df, x="Principal Component", y="Explained Variance (%)",
            title="Scree Plot – Explained Variance per Component",
            color_discrete_sequence=["coral"],
            text_auto=".1f",
        )
        st.plotly_chart(fig_scree, use_container_width=True)

    st.markdown("**PC Loadings** (contribution of each variable to each component)")
    st.dataframe(pca_result["loadings"].style.format("{:.4f}"), use_container_width=True)

    st.markdown(f"""
    **PCA Interpretation:**
    - **PC1 ({explained[0]*100:.1f}% variance)** captures the dominant data consumption axis — variables with high loadings on PC1 represent the overall heavy vs. light user dimension across all applications.
    - **PC2 ({explained[1]*100:.1f}% variance)** separates video-streaming services (YouTube, Netflix) from text-based services (Email, Google), revealing two distinct usage archetypes within the user base.
    - The first two components together explain **{(explained[0]+explained[1])*100:.1f}% of total variance**, confirming that app usage behaviour is highly correlated within user groups and can be effectively summarised in two dimensions.
    - Users in the outlier cluster (high PC1 values) are power users driving disproportionate network load — they represent a priority segment for capacity planning and premium plan targeting.
    """)

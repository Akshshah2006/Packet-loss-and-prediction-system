import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Packet Loss Detection & Prediction",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 AI-Based Packet Loss Detection and Prediction System")
st.caption("Machine Learning-Based Network Performance Analysis")

OUTPUT_DIR = "."

# --------------------------------------------------
# LOAD TRAINED MODELS AND RESULTS
# --------------------------------------------------

@st.cache_resource
def load_models():
    detection = joblib.load(f"{OUTPUT_DIR}/detection_model.pkl")
    prediction = joblib.load(f"{OUTPUT_DIR}/prediction_model.pkl")
    regression = joblib.load(f"{OUTPUT_DIR}/regression_model.pkl")

    with open(f"{OUTPUT_DIR}/config.json", "r") as f:
        config = json.load(f)

    return detection, prediction, regression, config


@st.cache_data
def load_data():
    df = pd.read_csv(f"{OUTPUT_DIR}/network_dataset.csv")

    detection_results = pd.read_csv(
        f"{OUTPUT_DIR}/detection_results.csv", index_col=0
    )

    prediction_results = pd.read_csv(
        f"{OUTPUT_DIR}/prediction_results.csv", index_col=0
    )

    regression_results = pd.read_csv(
        f"{OUTPUT_DIR}/regression_results.csv", index_col=0
    )

    unseen_results = pd.read_csv(
        f"{OUTPUT_DIR}/unseen_link_results.csv", index_col=0
    )

    return (
        df,
        detection_results,
        prediction_results,
        regression_results,
        unseen_results
    )


try:
    detection_model, prediction_model, regression_model, config = load_models()

    (
        df,
        detection_results,
        prediction_results,
        regression_results,
        unseen_results
    ) = load_data()

except Exception as e:
    st.error(
        "Unable to load project files. Make sure the "
        "'packet_loss_project' folder is in the same directory as app.py."
    )
    st.exception(e)
    st.stop()


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

H = config["horizon_seconds"]
LOSS_THRESHOLD = config["loss_threshold_pct"]
PRED_THRESHOLD = config["alert_threshold"]

DETECT_FEATURES = config["detect_features"]
PRED_FEATURES = config["prediction_features"]

LINKS = sorted(df["link_name"].unique())

# --------------------------------------------------
# FEATURE ENGINEERING
# Same logic as the Colab notebook
# --------------------------------------------------

HIST_BASE = [
    "utilization_pct",
    "queue_length",
    "rtt_ms",
    "offered_load_mbps",
    "loss_pct",
    "retransmissions"
]


def build_history_features(g):
    g = g.sort_values("time").copy()

    for c in HIST_BASE:
        for k in (1, 2, 3, 5):
            g[f"{c}_lag{k}"] = g[c].shift(k)

    g["util_trend3"] = (
        g["utilization_pct"] - g["utilization_pct"].shift(3)
    )

    g["queue_trend3"] = (
        g["queue_length"] - g["queue_length"].shift(3)
    )

    g["load_trend3"] = (
        g["offered_load_mbps"] -
        g["offered_load_mbps"].shift(3)
    )

    g["queue_roll5_mean"] = g["queue_length"].rolling(5).mean()
    g["queue_roll5_max"] = g["queue_length"].rolling(5).max()
    g["queue_roll5_std"] = g["queue_length"].rolling(5).std()

    g["load_roll5_mean"] = (
        g["offered_load_mbps"].rolling(5).mean()
    )

    g["util_roll5_max"] = (
        g["utilization_pct"].rolling(5).max()
    )

    g["rtt_roll5_std"] = g["rtt_ms"].rolling(5).std()

    g["loss_roll10_mean"] = (
        g["loss_pct"].rolling(10).mean()
    )

    g["queue_roll10_mean"] = (
        g["queue_length"].rolling(10).mean()
    )

    return g


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Dashboard Section",
    [
        "Overview",
        "Network Analysis",
        "Packet Loss Detection",
        "Packet Loss Prediction",
        "Model Performance",
        "Dataset Explorer"
    ]
)

st.sidebar.divider()

st.sidebar.write("**Project Information**")
st.sidebar.write("AI-Based Packet Loss Detection and Prediction")
st.sidebar.write(f"Prediction horizon: {H} seconds")
st.sidebar.write(f"Monitored links: {len(LINKS)}")


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.header("System Overview")

    total_records = len(df)
    total_links = df["link_name"].nunique()
    loss_events = int(df["current_loss"].sum())
    loss_percentage = df["current_loss"].mean() * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Measurements", f"{total_records:,}")
    c2.metric("Monitored Links", total_links)
    c3.metric("Packet Loss Events", f"{loss_events:,}")
    c4.metric("Loss Event Frequency", f"{loss_percentage:.2f}%")

    st.divider()

    st.subheader("Network Topology")

    topology = """
    graph Network {
        rankdir=TB;

        PC0 -- Switch0;
        PC1 -- Switch0;
        Switch0 -- Router0;

        PC2 -- Switch1;
        PC3 -- Switch1;
        Switch1 -- Router1;

        PC4 -- Switch2;
        PC5 -- Switch2;
        Switch2 -- Router2;

        Router0 -- Router1;
        Router1 -- Router2;
        Router1 -- Server0;

        node [shape=box, style=filled, fillcolor=lightblue];
        PC0; PC1; PC2; PC3; PC4; PC5;

        node [fillcolor=lightgoldenrod];
        Switch0; Switch1; Switch2;

        node [fillcolor=lightcoral];
        Router0; Router1; Router2;

        node [fillcolor=lightgreen];
        Server0;
    }
    """

    st.graphviz_chart(topology, use_container_width=True)

    st.divider()

    st.subheader("Packet Loss Distribution by Network Link")

    link_loss = (
        df.groupby("link_name")["current_loss"]
        .mean()
        .mul(100)
        .reset_index(name="Loss Event Frequency (%)")
    )

    fig = px.bar(
        link_loss,
        x="link_name",
        y="Loss Event Frequency (%)",
        color="Loss Event Frequency (%)",
        title="Packet Loss Frequency Across Network Links"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Network Traffic Parameters")

    parameter = st.selectbox(
        "Select parameter",
        [
            "utilization_pct",
            "queue_length",
            "rtt_ms",
            "jitter_ms",
            "throughput_mbps",
            "loss_pct"
        ]
    )

    selected_link = st.selectbox("Select network link", LINKS)

    temp = df[df["link_name"] == selected_link].sort_values("time")

    fig = px.line(
        temp.head(1000),
        x="time",
        y=parameter,
        title=f"{parameter} over time - {selected_link}"
    )

    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# NETWORK ANALYSIS
# --------------------------------------------------

elif page == "Network Analysis":

    st.header("Network Traffic Analysis")

    selected_link = st.selectbox(
        "Select Network Link",
        LINKS
    )

    link_df = df[df["link_name"] == selected_link].sort_values("time")

    st.subheader(f"Analysis: {selected_link}")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Average Utilization",
        f"{link_df['utilization_pct'].mean():.2f}%"
    )

    c2.metric(
        "Average Queue Length",
        f"{link_df['queue_length'].mean():.2f}"
    )

    c3.metric(
        "Average RTT",
        f"{link_df['rtt_ms'].mean():.2f} ms"
    )

    c4.metric(
        "Average Packet Loss",
        f"{link_df['loss_pct'].mean():.2f}%"
    )

    st.divider()

    st.subheader("Traffic Behaviour")

    chart_parameter = st.selectbox(
        "Choose network parameter",
        [
            "offered_load_mbps",
            "utilization_pct",
            "queue_length",
            "rtt_ms",
            "jitter_ms",
            "throughput_mbps",
            "loss_pct",
            "retransmissions"
        ]
    )

    fig = px.line(
        link_df.head(2000),
        x="time",
        y=chart_parameter,
        title=f"{chart_parameter} - {selected_link}"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Network Parameter Correlation")

    corr_cols = [
        "utilization_pct",
        "queue_length",
        "rtt_ms",
        "jitter_ms",
        "throughput_mbps",
        "retransmissions",
        "loss_pct"
    ]

    corr = link_df[corr_cols].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Network Parameter Correlation"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Raw Link Measurements")

    st.dataframe(
        link_df.head(100),
        use_container_width=True
    )


# --------------------------------------------------
# PACKET LOSS DETECTION
# --------------------------------------------------

elif page == "Packet Loss Detection":

    st.header("AI-Based Packet Loss Detection")

    st.write(
        "This module evaluates current network measurements "
        "to identify packet loss conditions."
    )

    selected_link = st.selectbox("Select network link", LINKS)

    link_df = df[df["link_name"] == selected_link].sort_values("time")

    selected_time = st.select_slider(
        "Select measurement time (seconds)",
        options=link_df["time"].astype(int).tolist(),
        value=int(link_df["time"].iloc[-1])
    )

    row = link_df[link_df["time"] == selected_time].iloc[0]

    input_data = pd.DataFrame(
        [[row[col] for col in DETECT_FEATURES]],
        columns=DETECT_FEATURES
    )

    probability = detection_model.predict_proba(input_data)[0, 1]
    detected = probability >= 0.5

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric("Detection Probability", f"{probability * 100:.2f}%")

    c2.metric(
        "Actual Packet Loss",
        f"{row['loss_pct']:.2f}%"
    )

    c3.metric(
        "Network Utilization",
        f"{row['utilization_pct']:.2f}%"
    )

    if detected:
        st.error("⚠️ AI MODEL: PACKET LOSS DETECTED")
    else:
        st.success("✅ AI MODEL: NO PACKET LOSS DETECTED")

    st.caption(
        "The detection model estimates the probability of a current "
        "loss event. The actual simulated loss measurement is displayed separately."
    )

    st.subheader("Input Network Parameters")

    st.dataframe(
        input_data.T.rename(columns={0: "Current Value"}),
        use_container_width=True
    )


# --------------------------------------------------
# PACKET LOSS PREDICTION
# --------------------------------------------------

elif page == "Packet Loss Prediction":

    st.header("AI-Based Future Packet Loss Prediction")

    st.write(
        f"The prediction model estimates packet loss risk "
        f"over the next {H} seconds using recent network history."
    )

    selected_link = st.selectbox("Select network link", LINKS)

    link_df = df[df["link_name"] == selected_link].sort_values("time")

    selected_time = st.select_slider(
        "Select prediction time (seconds)",
        options=link_df["time"].astype(int).tolist(),
        value=int(link_df["time"].iloc[-1])
    )

    history = link_df[link_df["time"] <= selected_time].copy()

    if len(history) < 11:
        st.warning(
            "At least 11 consecutive measurements are required "
            "for the prediction model."
        )

    else:

        featured = build_history_features(history)
        latest = featured.iloc[[-1]]

        if latest[PRED_FEATURES].isna().values.any():

            st.warning(
                "Not enough valid historical measurements to generate a prediction."
            )

        else:

            pred_input = latest[PRED_FEATURES]

            risk = prediction_model.predict_proba(pred_input)[0, 1]

            predicted_rate = max(
                0,
                float(regression_model.predict(pred_input)[0])
            )

            prediction = risk >= PRED_THRESHOLD

            st.divider()

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Future Packet Loss Risk",
                f"{risk * 100:.2f}%"
            )

            c2.metric(
                "Predicted Loss Rate",
                f"{predicted_rate:.2f}%"
            )

            c3.metric(
                "Prediction Horizon",
                f"{H} seconds"
            )

            if prediction:
                st.error("⚠️ EARLY WARNING: PACKET LOSS LIKELY")
            else:
                st.success("✅ LOW PREDICTED PACKET LOSS RISK")

            st.progress(float(np.clip(risk, 0, 1)))

            st.caption(
                f"Alert threshold: {PRED_THRESHOLD * 100:.1f}%"
            )

            st.subheader("Historical Network Conditions")

            history_chart = history.tail(100)

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=history_chart["time"],
                    y=history_chart["utilization_pct"],
                    name="Utilization (%)"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=history_chart["time"],
                    y=history_chart["loss_pct"],
                    name="Packet Loss (%)"
                )
            )

            fig.update_layout(
                title="Network Utilization and Packet Loss History",
                xaxis_title="Time (seconds)",
                yaxis_title="Percentage"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Prediction Input Features")

            st.dataframe(
                pred_input.T.rename(columns={0: "Value"}),
                use_container_width=True
            )


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.header("Machine Learning Model Performance")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Detection",
            "Prediction",
            "Regression",
            "Unseen Link Testing"
        ]
    )

    with tab1:

        st.subheader("Packet Loss Detection Results")

        st.write(f"Selected Model: **{config['detection_model']}**")

        st.dataframe(
            detection_results,
            use_container_width=True
        )

        st.bar_chart(detection_results)

    with tab2:

        st.subheader("Future Packet Loss Prediction Results")

        st.write(f"Selected Model: **{config['prediction_model']}**")

        st.dataframe(
            prediction_results,
            use_container_width=True
        )

        st.bar_chart(prediction_results)

    with tab3:

        st.subheader("Packet Loss Rate Regression Results")

        st.write(f"Selected Model: **{config['regression_model']}**")

        st.dataframe(
            regression_results,
            use_container_width=True
        )

        st.bar_chart(regression_results)

    with tab4:

        st.subheader("Generalization to Unseen Network Links")

        st.write(
            "Models are evaluated on links excluded from their training data."
        )

        st.dataframe(
            unseen_results,
            use_container_width=True
        )

        st.bar_chart(unseen_results)


# --------------------------------------------------
# DATASET EXPLORER
# --------------------------------------------------

elif page == "Dataset Explorer":

    st.header("Network Dataset Explorer")

    st.write(
        "Explore the measurements used to train and evaluate "
        "the packet loss detection and prediction models."
    )

    selected_link = st.selectbox(
        "Filter by network link",
        ["All Links"] + LINKS
    )

    if selected_link == "All Links":
        filtered = df.copy()
    else:
        filtered = df[df["link_name"] == selected_link].copy()

    c1, c2, c3 = st.columns(3)

    c1.metric("Records", f"{len(filtered):,}")
    c2.metric("Features", len(filtered.columns))
    c3.metric(
        "Packet Loss Events",
        f"{int(filtered['current_loss'].sum()):,}"
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        filtered.head(500),
        use_container_width=True
    )

    st.subheader("Dataset Statistics")

    st.dataframe(
        filtered.describe(),
        use_container_width=True
    )

    st.subheader("Download Dataset")

    csv = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Filtered CSV",
        data=csv,
        file_name="network_dataset.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AI-Based Packet Loss Detection and Prediction System | "
    "Cisco Packet Tracer + Python + Machine Learning + Streamlit"
)

st.caption(
    "Note: The default Colab dataset is simulated and topology-aware. "
    "Dashboard outputs reflect the loaded dataset and trained models, "
    "not live Cisco Packet Tracer measurements."
)

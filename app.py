import streamlit as st
import pandas as pd
import numpy as np
import joblib
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Graph-Based Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Graph-Based Bitcoin Fraud Detection")
st.write(
    "A hybrid machine learning system for detecting "
    "illicit Bitcoin transactions using the Elliptic dataset."
)

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "elliptic_bitcoin_dataset"

MODEL_PATH = BASE_DIR / "expanded_xgb_model.pkl"
COLUMNS_PATH = BASE_DIR / "feature_columns.pkl"

FEATURES_PATH = DATASET_DIR / "elliptic_txs_features.csv"
CLASSES_PATH = DATASET_DIR / "elliptic_txs_classes.csv"
EDGES_PATH = DATASET_DIR / "elliptic_txs_edgelist.csv"


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(COLUMNS_PATH)

    return model, feature_columns


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():

    # Load transaction features
    features = pd.read_csv(
        FEATURES_PATH,
        header=None
    )

    features.columns = (
        ["txId", "time_step"] +
        [f"feature_{i}" for i in range(1, 166)]
    )

    features["txId"] = (
        features["txId"].astype(str).str.strip()
    )

    # Load transaction classes
    classes = pd.read_csv(
        CLASSES_PATH,
        header=None,
        names=["txId", "class"]
    )

    # Remove accidental header row if present
    classes = classes[
        classes["class"].astype(str).str.lower() != "class"
    ].copy()

    classes["txId"] = (
        classes["txId"].astype(str).str.strip()
    )

    classes["class"] = (
        classes["class"].astype(str).str.strip()
    )

    # Load graph features
    graph_features_path = BASE_DIR / "graph_features.pkl"

    graph_features = joblib.load(graph_features_path)

    graph_features["txId"] = (
        graph_features["txId"].astype(str).str.strip()
    )

    # Merge transaction and graph features
    data = features.merge(
        graph_features,
        on="txId",
        how="left",
        validate="one_to_one"
    )

    # Merge class labels
    data = data.merge(
        classes,
        on="txId",
        how="left",
        validate="one_to_one"
    )

    # Load transaction edges
    edges = pd.read_csv(
        EDGES_PATH,
        header=None,
        names=["txId1", "txId2"]
    )

    edges["txId1"] = (
        edges["txId1"].astype(str).str.strip()
    )

    edges["txId2"] = (
        edges["txId2"].astype(str).str.strip()
    )

    # Remove accidental header row
    edges = edges[
        (edges["txId1"] != "txId1") &
        (edges["txId2"] != "txId2")
    ].copy()

    return data, edges


# --------------------------------------------------
# LOAD APPLICATION DATA
# --------------------------------------------------

try:

    model, feature_columns = load_model()

    data, edges = load_data()

except FileNotFoundError as error:

    st.error(f"Required file not found: {error}")

    st.info(
        "Check that the model files, graph_features.pkl "
        "and elliptic_bitcoin_dataset folder are present."
    )

    st.stop()

except Exception as error:

    st.error(f"Error loading project files: {error}")
    st.stop()


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a section",
    [
        "Dashboard Overview",
        "Transaction Prediction",
        "Transaction Network",
        "Model Analysis"
    ]
)


# --------------------------------------------------
# PAGE 1: DASHBOARD OVERVIEW
# --------------------------------------------------

if page == "Dashboard Overview":

    st.header("📊 Dashboard Overview")

    total_transactions = len(data)

    illicit_count = (
        data["class"] == "1"
    ).sum()

    licit_count = (
        data["class"] == "2"
    ).sum()

    unknown_count = (
        data["class"] == "unknown"
    ).sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "Illicit Transactions",
        f"{illicit_count:,}"
    )

    col3.metric(
        "Licit Transactions",
        f"{licit_count:,}"
    )

    col4.metric(
        "Unknown Transactions",
        f"{unknown_count:,}"
    )

    st.divider()

    st.subheader("Transaction Class Distribution")

    class_counts = pd.DataFrame({
        "Class": [
            "Illicit",
            "Licit",
            "Unknown"
        ],
        "Count": [
            illicit_count,
            licit_count,
            unknown_count
        ]
    })

    fig = px.pie(
        class_counts,
        names="Class",
        values="Count",
        title="Distribution of Bitcoin Transactions",
        hole=0.4
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Transactions Across Time Steps")

    time_counts = (
        data.groupby("time_step")
        .size()
        .reset_index(name="Transaction Count")
    )

    fig = px.line(
        time_counts,
        x="time_step",
        y="Transaction Count",
        markers=True,
        title="Transaction Count by Time Step"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Sample Transaction Data")

    st.dataframe(
        data[
            ["txId", "time_step", "class",
             "in_degree", "out_degree", "pagerank"]
        ].head(20),
        use_container_width=True
    )


# --------------------------------------------------
# PAGE 2: TRANSACTION PREDICTION
# --------------------------------------------------

elif page == "Transaction Prediction":

    st.header("🔎 Transaction Prediction")

    st.write(
        "Enter a transaction ID to predict whether "
        "the transaction is licit or illicit."
    )

    tx_id = st.text_input(
        "Enter Bitcoin Transaction ID"
    )

    if st.button("Predict Transaction"):

        tx_id = tx_id.strip()

        transaction = data[
            data["txId"] == tx_id
        ]

        if transaction.empty:

            st.error("Transaction ID not found.")

        else:

            transaction = transaction.iloc[[0]]

            # Prepare model input
            X = transaction[feature_columns]

            # Predict class
            prediction = model.predict(X)[0]

            # Get prediction probability
            probability = model.predict_proba(X)[0][1]

            st.divider()

            st.subheader("Prediction Result")

            if prediction == 1:

                st.error("⚠️ Predicted Class: ILLICIT")

            else:

                st.success("✅ Predicted Class: LICIT")

            col1, col2 = st.columns(2)

            col1.metric(
                "Illicit Probability",
                f"{probability * 100:.2f}%"
            )

            col2.metric(
                "Time Step",
                int(transaction["time_step"].iloc[0])
            )

            st.subheader("Transaction Graph Features")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "In-Degree",
                int(transaction["in_degree"].iloc[0])
            )

            col2.metric(
                "Out-Degree",
                int(transaction["out_degree"].iloc[0])
            )

            col3.metric(
                "PageRank",
                f"{transaction['pagerank'].iloc[0]:.6f}"
            )

            original_class = transaction["class"].iloc[0]

            if original_class == "1":
                st.write("Original dataset label: Illicit")

            elif original_class == "2":
                st.write("Original dataset label: Licit")

            else:
                st.write("Original dataset label: Unknown")

            st.caption(
                "The original label is the dataset's reference "
                "label, not the model's prediction."
            )


# --------------------------------------------------
# PAGE 3: TRANSACTION NETWORK
# --------------------------------------------------

elif page == "Transaction Network":

    st.header("🌐 Bitcoin Transaction Network")

    st.write(
        "Explore transactions connected to a selected "
        "Bitcoin transaction."
    )

    tx_id = st.text_input(
        "Enter Transaction ID",
        key="network_tx_id"
    )

    if st.button("Visualize Network"):

        selected_id = tx_id.strip()

        if not selected_id:

            st.warning("Please enter a transaction ID.")

        elif selected_id not in set(data["txId"]):

            st.error("Transaction ID not found.")

        else:

            with st.spinner("Building transaction network..."):

                G = nx.from_pandas_edgelist(
                    edges,
                    source="txId1",
                    target="txId2",
                    create_using=nx.DiGraph()
                )

                # Include all transaction nodes
                G.add_nodes_from(data["txId"])

                # Find directly connected transactions
                neighbors = (
                    list(G.predecessors(selected_id)) +
                    list(G.successors(selected_id))
                )

                # Remove duplicates and limit display
                neighbors = list(
                    dict.fromkeys(neighbors)
                )[:20]

                if not neighbors:

                    st.info(
                        "No connected transactions found."
                    )

                else:

                    nodes = [selected_id] + neighbors

                    subgraph = G.subgraph(nodes).copy()

                    label_map = (
                        data.set_index("txId")["class"].to_dict()
                    )

                    node_colors = []

                    for node in subgraph.nodes():

                        if node == selected_id:

                            node_colors.append("orange")

                        elif label_map.get(node) == "1":

                            node_colors.append("red")

                        elif label_map.get(node) == "2":

                            node_colors.append("green")

                        else:

                            node_colors.append("gray")

                    # Calculate node positions
                    pos = nx.spring_layout(
                        subgraph,
                        seed=42
                    )

                    # Build edge coordinates
                    edge_x = []
                    edge_y = []

                    for source, target in subgraph.edges():

                        x0, y0 = pos[source]
                        x1, y1 = pos[target]

                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])

                    edge_trace = go.Scatter(
                        x=edge_x,
                        y=edge_y,
                        mode="lines",
                        line=dict(
                            width=1,
                            color="#888"
                        ),
                        hoverinfo="none"
                    )

                    # Build node coordinates
                    node_x = []
                    node_y = []
                    node_text = []

                    for node in subgraph.nodes():

                        x, y = pos[node]

                        node_x.append(x)
                        node_y.append(y)

                        node_text.append(
                            f"Transaction: {node}<br>"
                            f"Original class: "
                            f"{label_map.get(node, 'Unknown')}"
                        )

                    node_trace = go.Scatter(
                        x=node_x,
                        y=node_y,
                        mode="markers",
                        marker=dict(
                            size=18,
                            color=node_colors,
                            line=dict(
                                width=1,
                                color="white"
                            )
                        ),
                        text=node_text,
                        hoverinfo="text"
                    )

                    fig = go.Figure(
                        data=[
                            edge_trace,
                            node_trace
                        ]
                    )

                    fig.update_layout(
                        title="Connected Transaction Network",
                        showlegend=False,
                        height=600,
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        margin=dict(
                            l=20,
                            r=20,
                            t=50,
                            b=20
                        )
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    st.write(
                        "**Selected Transaction:**",
                        selected_id
                    )

                    st.write(
                        "**Connected Transactions:**",
                        len(neighbors)
                    )

                    st.caption(
                        "Orange = selected transaction | "
                        "Red = originally labelled illicit | "
                        "Green = originally labelled licit | "
                        "Gray = unknown"
                    )

                    st.warning(
                        "Node colors show original dataset labels, "
                        "not model predictions. This visualization "
                        "shows direct connections only."
                    )


# --------------------------------------------------
# PAGE 4: MODEL ANALYSIS
# --------------------------------------------------

elif page == "Model Analysis":

    st.header("📈 Model Performance Analysis")

    st.write(
        "Comparison of the models using the test results "
        "recorded during project evaluation."
    )

    results = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
            "GCN",
            "Hybrid GCN + XGBoost"
        ],

        "Precision": [
            0.1243,
            0.9883,
            0.6967,
            0.2021,
            0.6696
        ],

        "Recall": [
            0.9205,
            0.5900,
            0.6471,
            0.6109,
            0.6360
        ],

        "F1 Score": [
            0.2190,
            0.7389,
            0.6710,
            0.3037,
            0.6524
        ],

        "Average Precision": [
            0.1739,
            0.6977,
            0.6983,
            0.3030,
            0.6831
        ],

        "Accuracy": [
            0.6193,
            0.9758,
            0.9632,
            0.8376,
            0.9607
        ]
    })

    st.subheader("Model Comparison Table")

    st.dataframe(
        results.style.format({
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "Average Precision": "{:.4f}",
            "Accuracy": "{:.4f}"
        }),
        use_container_width=True
    )

    st.subheader("F1 Score Comparison")

    fig = px.bar(
        results,
        x="Model",
        y="F1 Score",
        color="Model",
        title="F1 Score of Different Models"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Average Precision Comparison")

    fig = px.bar(
        results,
        x="Model",
        y="Average Precision",
        color="Model",
        title="Average Precision of Different Models"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "In these recorded test results, XGBoost achieved "
        "the highest F1 score and Average Precision. "
        "Random Forest achieved the highest precision."
    )

    st.caption(
        "These are previously recorded evaluation metrics, "
        "not metrics recalculated by this dashboard."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Graph-Based Fraud Detection | "
    "Elliptic Bitcoin Transaction Dataset | "
    "Machine Learning and Graph Analytics"
)

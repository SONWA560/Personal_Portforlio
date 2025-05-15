import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(page_title="7-Class Model Evaluation", layout="wide")

# Get paths - works in both local and Streamlit Cloud
BASE_DIR = os.getcwd()
DATA_RELATIVE_PATH = os.path.join("model_evaluation", "7-class")

# Construct full paths
DATA_DIR = os.path.join(BASE_DIR, DATA_RELATIVE_PATH)
PRED_FILE = os.path.join(DATA_DIR, "detailed_predictions.csv")
CM_FILE = os.path.join(DATA_DIR, "confusion_matrix.csv")
CR_FILE = os.path.join(DATA_DIR, "classification_report.csv")

# Initialize session state for data persistence
if 'pred_df' not in st.session_state:
    try:
        st.session_state.pred_df = pd.read_csv(PRED_FILE)
        st.session_state.pred_df.columns = st.session_state.pred_df.columns.str.strip().str.lower()
        # Ensure we have a predicted column
        possible_pred_cols = ['predicted', 'prediction', 'predicted_class', 'class', 'predicted_label']
        pred_col = next((col for col in possible_pred_cols if col in st.session_state.pred_df.columns), None)
        if pred_col and pred_col != 'predicted':
            st.session_state.pred_df = st.session_state.pred_df.rename(columns={pred_col: 'predicted'})
    except FileNotFoundError:
        st.session_state.pred_df = None
        st.error(f"Prediction data file not found at: {PRED_FILE}")

if 'cm_df' not in st.session_state:
    try:
        st.session_state.cm_df = pd.read_csv(CM_FILE, index_col=0)
    except FileNotFoundError:
        st.session_state.cm_df = None
        st.error(f"Confusion matrix file not found at: {CM_FILE}")

if 'cr_df' not in st.session_state:
    try:
        st.session_state.cr_df = pd.read_csv(CR_FILE, index_col=0)
    except FileNotFoundError:
        st.session_state.cr_df = None
        st.error(f"Classification report file not found at: {CR_FILE}")

# Sidebar filters
def add_filters():
    st.sidebar.header("Filters")
    if st.session_state.pred_df is not None:
        all_classes = st.session_state.pred_df['predicted'].unique()
        selected_classes = st.sidebar.multiselect("Select classes to display", options=all_classes, default=all_classes)
        min_confidence = st.sidebar.slider("Minimum confidence level", 0.0, 1.0, 0.0, 0.05)
        correctness_filter = st.sidebar.radio("Prediction correctness", ["All", "Correct only", "Incorrect only"])
        return selected_classes, min_confidence, correctness_filter
    return None, None, None

# Sidebar navigation
def navigation():
    st.sidebar.title("Navigation")
    return st.sidebar.radio("Go to", ["Home", "Model Performance", "Confusion Matrix", "Classification Report"])

# Home Page
def home_page():
    st.title("🏠 7-Class Waste Classification Model Evaluation")
    st.markdown("""
    Welcome to the **Model Evaluation Dashboard** for the 7-Class Waste Classification project.

    Use the sidebar to navigate between:
    - **Model Performance**: Class distributions, confidence, and class-wise metrics
    - **Confusion Matrix**: Analyze misclassifications
    - **Classification Report**: Detailed precision, recall, F1-score

    **Start by exploring how your model is performing across different metrics.**
    """)

    if st.session_state.pred_df is not None:
        class_distribution()

# Model Performance Page
def performance_page(selected_classes, min_confidence, correctness_filter):
    st.title("📊 Model Performance Analysis")
    if st.session_state.pred_df is None:
        st.error("Prediction data not loaded. Please ensure the detailed_predictions.csv file is available.")
        return
    class_distribution()
    confidence_analysis(selected_classes, min_confidence, correctness_filter)
    performance_metrics(selected_classes)

# Class Distribution Plot
def class_distribution():
    st.subheader("Class Distribution in Predictions")
    class_counts = st.session_state.pred_df['predicted'].value_counts().reset_index()
    class_counts.columns = ['Class', 'Count']
    fig = px.bar(class_counts, x='Class', y='Count', color='Class',
                 title="Distribution of Predictions Across Classes")
    fig.update_layout(xaxis_title="Class", yaxis_title="Number of Predictions")
    st.plotly_chart(fig, use_container_width=True)

# Confidence Analysis
def confidence_analysis(selected_classes, min_confidence, correctness_filter):
    st.subheader("Confidence Analysis")
    df = st.session_state.pred_df.copy()
    df = df[df['predicted'].isin(selected_classes)]
    df = df[df['confidence'] >= min_confidence]

    if correctness_filter == "Correct only":
        df = df[df['correct']]
    elif correctness_filter == "Incorrect only":
        df = df[~df['correct']]

    fig1 = px.histogram(df, x='confidence', color='correct', nbins=30, barmode='overlay',
                        title="Prediction Confidence Distribution")
    fig2 = px.box(df, x='predicted', y='confidence', color='correct',
                  title="Confidence Distribution by Class")
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

# Performance Metrics
def performance_metrics(selected_classes):
    st.subheader("Performance Metrics")
    df = st.session_state.pred_df
    metrics = df.groupby('predicted').agg(
        accuracy=('correct', 'mean'),
        avg_confidence=('confidence', 'mean'),
        count=('predicted', 'count')
    ).reset_index()
    metrics = metrics[metrics['predicted'].isin(selected_classes)]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy by Class", "Average Confidence by Class"))
    fig.add_trace(go.Bar(x=metrics['predicted'], y=metrics['accuracy'], name="Accuracy", marker_color='blue'), row=1, col=1)
    fig.add_trace(go.Bar(x=metrics['predicted'], y=metrics['avg_confidence'], name="Confidence", marker_color='green'), row=1, col=2)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Confusion Matrix Page
def confusion_matrix_page():
    st.title("📈 Confusion Matrix Analysis")
    if st.session_state.cm_df is None:
        st.error("Confusion matrix not loaded.")
        return

    st.sidebar.header("Confusion Matrix Filters")
    norm_type = st.sidebar.radio("Normalization", ["Counts", "By True Class", "By Predicted Class"], index=1)
    cm = st.session_state.cm_df.copy()

    if norm_type == "By True Class":
        norm_cm = cm.div(cm.sum(axis=1), axis=0)
        fmt = ".1%"
        zmin, zmax = 0, 1
    elif norm_type == "By Predicted Class":
        norm_cm = cm.div(cm.sum(axis=0), axis=1)
        fmt = ".1%"
        zmin, zmax = 0, 1
    else:
        norm_cm = cm
        fmt = "d"
        zmin, zmax = None, None

    fig = px.imshow(norm_cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=norm_cm.columns, y=norm_cm.index, text_auto=fmt,
                    color_continuous_scale='Blues', zmin=zmin, zmax=zmax)
    fig.update_xaxes(side="top")
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)

    # Accuracy
    total = cm.values.sum()
    correct = np.trace(cm)
    accuracy = correct / total
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Accuracy", f"{accuracy:.1%}")
    col2.metric("Correct Predictions", f"{correct:,}")
    col3.metric("Total Samples", f"{total:,}")

# Classification Report Page
def classification_report_page():
    st.title("📝 Classification Report Analysis")
    if st.session_state.cr_df is None:
        st.error("Classification report not available.")
        return

    st.sidebar.header("Report Filters")
    show_avg = st.sidebar.checkbox("Show Averages", value=True)
    metric = st.sidebar.selectbox("Primary Metric", ["precision", "recall", "f1-score"], index=2)

    cr = st.session_state.cr_df.copy()
    if not show_avg:
        cr = cr.drop(index=["accuracy", "macro avg", "weighted avg"], errors='ignore')

    fig = px.bar(cr, y=cr.index, x=[metric, "precision", "recall"], barmode='group',
                 labels={'value': 'Score', 'variable': 'Metric'})
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    fig.add_vline(x=0.9, line_dash="dot", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Precision vs Recall")
    cr_filtered = cr.drop(index=["accuracy", "macro avg", "weighted avg"], errors='ignore')
    fig2 = px.scatter(cr_filtered, x='recall', y='precision', text=cr_filtered.index,
                      size='f1-score', color='f1-score', hover_name=cr_filtered.index)
    fig2.add_shape(type="line", x0=0, y0=1, x1=1, y1=0, line=dict(dash="dash"))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Detailed Report")
    st.dataframe(cr.style.background_gradient(subset=['precision', 'recall', 'f1-score'], cmap='YlGnBu')
                 .format({'precision': '{:.1%}', 'recall': '{:.1%}', 'f1-score': '{:.1%}'}))

# Main app
def main():
    page = navigation()
    selected_classes, min_confidence, correctness_filter = add_filters()
    if page == "Home":
        home_page()
    elif page == "Model Performance":
        performance_page(selected_classes, min_confidence, correctness_filter)
    elif page == "Confusion Matrix":
        confusion_matrix_page()
    elif page == "Classification Report":
        classification_report_page()

if __name__ == "__main__":
    main()

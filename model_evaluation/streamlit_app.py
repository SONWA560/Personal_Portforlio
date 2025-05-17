import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import numpy as np
from pathlib import Path  # More robust path handling

# --- Helper Functions ---
def is_valid_data_path(path):
    required_files = [
        "detailed_predictions.csv",
        "confusion_matrix.csv",
        "classification_report.csv"
    ]
    return all((path / f).exists() for f in required_files)

def get_data_path():
    """Resolves correct data path for both local and GitHub environments"""
    try:
        base_path = Path(__file__).parent
        candidate_paths = [
            base_path / "model_evaluation" / "7-class",
            Path("model_evaluation") / "7-class",
            Path("data")
        ]
        for path in candidate_paths:
            if is_valid_data_path(path):
                return path
        raise FileNotFoundError("Data directory not found. Please ensure the expected structure is present.")
    except Exception as e:
        st.error(f"Data path resolution failed: {e}")
        st.stop()

DATA_DIR = get_data_path()

# --- Data Loading with Error Handling ---
@st.cache_data
def load_data():
    try:
        predictions_df = pd.read_csv(DATA_DIR / "detailed_predictions.csv")
        confusion_df = pd.read_csv(DATA_DIR / "confusion_matrix.csv", index_col=0)
        report_df = pd.read_csv(DATA_DIR / "classification_report.csv")

        # Validate required columns
        required_cols = {'true_label', 'predicted_label', 'confidence', 'correct'}
        missing_cols = required_cols - set(predictions_df.columns)
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.stop()

        return predictions_df, confusion_df, report_df

    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.error(f"Checked path: {DATA_DIR}")
        st.error("Ensure your data files exist in the correct directory structure:")
        st.code("""
        model_evaluation/
        └── 7-class/
            ├── detailed_predictions.csv
            ├── confusion_matrix.csv
            └── classification_report.csv
        """)
        st.stop()

# --- Metric Color Helper ---
def colour_gradient(val, max_val):
    return f"background-color: rgba(0, 123, 255, {val/max_val})"

# --- Main App ---
def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Model Evaluation Dashboard")

    try:
        predictions_df, confusion_df, report_df = load_data()
        classes = sorted(predictions_df['true_label'].unique())

        page = st.sidebar.radio("Go to", [
            "Overview",
            "Class Distribution",
            "Confidence Analysis",
            "Confusion Matrix",
            "Classification Metrics"
        ])

        # 1. Overview
        if page == "Overview":
            st.title("Model Performance Overview")

            correct = predictions_df['correct'].sum()
            total = len(predictions_df)
            accuracy = correct / total

            col1, col2, col3 = st.columns(3)
            col1.metric("Overall Accuracy", f"{accuracy:.2%}")
            col2.metric("Correct Predictions", f"{correct}")
            col3.metric("Total Predictions", f"{total}")

            # Pie chart of correct vs incorrect
            fig_pie = px.pie(
                names=["Correct", "Incorrect"],
                values=[correct, total - correct],
                title="Correct vs Incorrect Predictions",
                color_discrete_sequence=["#2ca02c", "#d62728"]
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # 2. Class Distribution
        elif page == "Class Distribution":
            st.title("Class Distribution in Predictions")
            class_counts = predictions_df['predicted_label'].value_counts().reindex(classes, fill_value=0)
            fig_bar = px.bar(x=class_counts.index, y=class_counts.values,
                             labels={'x': 'Class', 'y': 'Number of Predictions'},
                             title="Number of Predictions Per Class")
            st.plotly_chart(fig_bar, use_container_width=True)

        # 3. Confidence Analysis
        elif page == "Confidence Analysis":
            st.title("Confidence Distribution")

            # Histogram
            st.subheader("Prediction Confidence (All Classes)")
            fig_hist = px.histogram(predictions_df, x='confidence', color='correct', nbins=20,
                                    title="Prediction Confidence Distribution",
                                    color_discrete_map={True: "green", False: "red"})
            st.plotly_chart(fig_hist, use_container_width=True)

            # Box plot
            st.subheader("Confidence by Class")
            fig_box = px.box(predictions_df, x='predicted_label', y='confidence', color='correct',
                            title="Prediction Confidence by Class",
                            labels={'predicted_label': 'Predicted Class'})
            st.plotly_chart(fig_box, use_container_width=True)

            # Accuracy & confidence by class
            st.subheader("Class-wise Accuracy and Confidence")
            class_metrics = predictions_df.groupby('true_label').agg(
                accuracy=('correct', 'mean'),
                avg_confidence=('confidence', 'mean')
            ).reindex(classes)

            fig_acc_conf = go.Figure()
            fig_acc_conf.add_trace(go.Bar(x=classes, y=class_metrics['accuracy'], name='Accuracy'))
            fig_acc_conf.add_trace(go.Bar(x=classes, y=class_metrics['avg_confidence'], name='Avg Confidence'))
            fig_acc_conf.update_layout(barmode='group', title="Accuracy vs Average Confidence")
            st.plotly_chart(fig_acc_conf, use_container_width=True)

            # False positives vs false negatives
            st.subheader("False Positives vs False Negatives")
            false_positives = predictions_df[predictions_df['correct'] == False].groupby('predicted_label').size().reindex(classes, fill_value=0)
            false_negatives = predictions_df[predictions_df['correct'] == False].groupby('true_label').size().reindex(classes, fill_value=0)

            fig_fp_fn = go.Figure()
            fig_fp_fn.add_trace(go.Bar(x=classes, y=false_positives, name="False Positives"))
            fig_fp_fn.add_trace(go.Bar(x=classes, y=false_negatives, name="False Negatives"))
            fig_fp_fn.update_layout(barmode='group', title="False Positives vs False Negatives by Class")
            st.plotly_chart(fig_fp_fn, use_container_width=True)

        # 4. Confusion Matrix
        elif page == "Confusion Matrix":
            st.title("Confusion Matrix")
            normalize_option = st.selectbox("Normalisation", ["None", "True", "Pred"])
            cm = confusion_matrix(predictions_df['true_label'], predictions_df['predicted_label'], labels=classes)

            if normalize_option == "True":
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            elif normalize_option == "Pred":
                cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]

            fig_cm = px.imshow(cm, x=classes, y=classes, color_continuous_scale='Blues',
                               labels=dict(x="Predicted Label", y="True Label", color="Count"),
                               title=f"Confusion Matrix ({normalize_option} Normalised)")
            st.plotly_chart(fig_cm, use_container_width=True)

            # Top misclassifications
            st.subheader("Top Misclassified Pairs")
            misclassified = predictions_df[predictions_df['correct'] == False]
            mis_pairs = misclassified.groupby(['true_label', 'predicted_label']).size().reset_index(name='count')
            mis_pairs = mis_pairs.sort_values(by='count', ascending=False).head(10)
            fig_top_mis = px.bar(mis_pairs, x='count', y='true_label', color='predicted_label', orientation='h',
                                 title="Top 10 Misclassified Class Pairs")
            st.plotly_chart(fig_top_mis, use_container_width=True)

        # 5. Classification Metrics
        elif page == "Classification Metrics":
            st.title("Classification Report")

            # Grouped bar chart
            melted_report = report_df.melt(id_vars='class', value_vars=['precision', 'recall', 'f1-score'],
                                           var_name='metric', value_name='value')
            fig_grouped = px.bar(melted_report, x='class', y='value', color='metric', barmode='group',
                                 title="Precision, Recall, F1-score by Class")
            st.plotly_chart(fig_grouped, use_container_width=True)

            # Precision vs Recall scatter
            st.subheader("Precision vs Recall")
            fig_pr = px.scatter(report_df, x='recall', y='precision', size='f1-score', color='f1-score',
                                hover_name='class', title="Precision vs Recall with F1-score")
            st.plotly_chart(fig_pr, use_container_width=True)

            # Table
            st.subheader("Detailed Classification Report")
            styled_report = report_df.style.apply(lambda row: [colour_gradient(val, 1.0) for val in row[1:]], axis=1)
            st.dataframe(styled_report, use_container_width=True)

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Check the repository structure and data files")

if __name__ == "__main__":
    main()

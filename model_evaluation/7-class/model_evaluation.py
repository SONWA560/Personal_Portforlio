import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

st.set_page_config(page_title="7-Class Model Evaluation", layout="wide")
st.title("🧠 Multi-Class Waste Classification Model Evaluation")
st.markdown("This dashboard visualises the performance of your trained model across 7 waste categories.")

# File upload
detailed_predictions_path = st.file_uploader("📂 Upload detailed_predictions.csv", type=["csv"])
conf_matrix_path = st.file_uploader("📂 Upload confusion_matrix.csv", type=["csv"])
class_report_path = st.file_uploader("📂 Upload classification_report.csv", type=["csv"])

if detailed_predictions_path:
    pred_df = pd.read_csv(detailed_predictions_path)

    # Normalise column names
    pred_df.columns = pred_df.columns.str.strip().str.lower()
    
    # Check if required columns exist
    required_columns = {'confidence', 'correct'}
    if not required_columns.issubset(pred_df.columns):
        st.error(f"Error: The uploaded file is missing required columns. Needs {required_columns}, found {set(pred_df.columns)}")
    else:
        st.subheader("Prediction Confidence Histogram")
        fig, ax = plt.subplots()
        sns.histplot(data=pred_df, x="confidence", hue="correct", bins=30, kde=True, 
                     palette="Set2", multiple="stack")
        plt.title("Prediction Confidence by Correctness")
        st.pyplot(fig)
        st.markdown(
            "This histogram shows the model's prediction confidence for correct versus incorrect classifications. "
            "Higher confidence for correct predictions and lower confidence for incorrect ones is desirable. "
            "It helps us assess whether the model is overconfident in its mistakes or well-calibrated."
        )

        st.subheader("Correct vs Incorrect Predictions (Pie Chart)")
        correct_counts = pred_df["correct"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(correct_counts, labels=correct_counts.index, autopct="%1.1f%%", 
               startangle=90, colors=["#66c2a5", "#fc8d62"])
        plt.title("Overall Prediction Correctness")
        st.pyplot(fig)
        st.markdown(
            "This pie chart illustrates the proportion of correct versus incorrect predictions. "
            "It provides a high-level view of the model's overall accuracy. A balanced model should aim for a high share of correct predictions."
        )

        st.subheader("Per-Class Confidence Distribution")
        # Try different common column names for predicted class
        possible_pred_cols = ['predicted', 'prediction', 'predicted_class', 'class', 'predicted_label']
        pred_col = next((col for col in possible_pred_cols if col in pred_df.columns), None)
        
        if pred_col:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=pred_df, x=pred_col, y="confidence", palette="Set3")
            plt.xticks(rotation=45)
            plt.title(f"Confidence Distribution by {pred_col.title()} Class")
            st.pyplot(fig)
            st.markdown(
                "This box plot shows how confident the model is for each predicted class. "
                "A consistent and high confidence per class suggests strong predictive certainty, whereas wide variability may suggest inconsistency in classification reliability."
            )
        else:
            st.error(f"Could not find predicted class column. Tried: {possible_pred_cols}")
            st.write("Available columns:", pred_df.columns.tolist())

        st.subheader("Sample Predictions")
        st.dataframe(pred_df.head(20))

if conf_matrix_path:
    try:
        cm_df = pd.read_csv(conf_matrix_path, index_col=0)
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix")
        st.pyplot(fig)
        st.markdown(
            "The confusion matrix shows the actual versus predicted labels. Diagonal values represent correct predictions, "
            "while off-diagonal values indicate misclassifications. This helps pinpoint which classes are most commonly confused by the model."
        )
    except Exception as e:
        st.error(f"Error loading confusion matrix: {str(e)}")

if class_report_path:
    try:
        cr_df = pd.read_csv(class_report_path, index_col=0)
        st.subheader("Classification Report Table")
        st.dataframe(cr_df.style.background_gradient(cmap='YlGnBu', axis=1))
        st.markdown(
            "The classification report presents precision, recall, and F1-score for each class. "
            "Precision indicates how many of the predicted positives are truly positive. "
            "Recall shows how many actual positives were correctly predicted. "
            "F1-score balances both. This table is useful for identifying class-specific performance."
        )

        st.subheader("Precision, Recall and F1-score by Class")
        cr_plot_df = cr_df.drop(index=["accuracy", "macro avg", "weighted avg"], errors='ignore')
        cr_plot_df = cr_plot_df[["precision", "recall", "f1-score"]].reset_index().melt(
            id_vars="index", var_name="Metric", value_name="Score")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=cr_plot_df, x="index", y="Score", hue="Metric", palette="Set2")
        plt.title("Precision, Recall and F1-score per Class")
        plt.ylabel("Score")
        plt.xlabel("Class")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.markdown(
            "This grouped bar chart gives a visual comparison of precision, recall, and F1-score across all classes. "
            "It highlights classes with poor or strong performance and is useful for identifying where to focus improvements."
        )
    except Exception as e:
        st.error(f"Error loading classification report: {str(e)}")
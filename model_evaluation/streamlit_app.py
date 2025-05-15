import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Set page config
st.set_page_config(page_title="7-Class Model Evaluation", layout="wide")

# File paths 
DATA_DIR = "7-class"
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

# Navigation function
def navigation():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Model Performance", "Confusion Matrix", "Classification Report"])
    return page

# Visualization Functions
def class_distribution():
    st.subheader("Class Distribution in Predictions")
    class_counts = st.session_state.pred_df['predicted'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
    plt.xticks(rotation=45)
    plt.ylabel("Count")
    st.pyplot(fig)

def confidence_accuracy():
    st.subheader("Confidence vs Accuracy by Class")
    avg_data = st.session_state.pred_df.groupby('predicted').agg(
        avg_confidence=('confidence', 'mean'),
        accuracy=('correct', 'mean')
    ).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=avg_data, x='avg_confidence', y='accuracy', 
                   hue='predicted', s=200, palette="Set2")
    plt.xlabel("Average Confidence")
    plt.ylabel("Accuracy")
    plt.axhline(y=0.9, color='r', linestyle='--', label='Target Accuracy')
    plt.legend()
    st.pyplot(fig)

def misclassification_heatmap():
    st.subheader("Normalized Misclassification Heatmap")
    norm_cm = st.session_state.cm_df.div(st.session_state.cm_df.sum(axis=1), axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(norm_cm, annot=True, fmt=".1%", cmap="Reds", cbar=False)
    plt.title("Normalized by True Class")
    st.pyplot(fig)

def pr_curve():
    st.subheader("Precision-Recall Tradeoff")
    cr_plot_df = st.session_state.cr_df.drop(
        index=["accuracy", "macro avg", "weighted avg"], errors='ignore')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=cr_plot_df, x='recall', y='precision', 
                   hue=cr_plot_df.index, s=200, palette="Set3")
    plt.plot([0, 1], [1, 0], linestyle='--', color='gray')
    plt.title("Precision vs Recall by Class")
    st.pyplot(fig)

def top_misclassifications():
    st.subheader("Top Misclassification Pairs")
    cm = st.session_state.cm_df.copy()
    np.fill_diagonal(cm.values, 0)  # Remove correct predictions
    melted = cm.reset_index().melt(id_vars='index', var_name='predicted', value_name='count')
    top_errors = melted.nlargest(10, 'count')
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_errors, x='count', y='index', hue='predicted', 
               palette="YlOrRd", dodge=False)
    plt.xlabel("Misclassification Count")
    plt.ylabel("Actual Class")
    st.pyplot(fig)

# Home page
def home_page():
    st.title("Multi-Class Waste Classification Model Evaluation")
    st.markdown("""
    ## Waste Classification Model Evaluation Dashboard
    
    This dashboard displays comprehensive performance metrics of our 7-class waste classification model.
    """)
    
    # Show data status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediction Data", "Loaded" if st.session_state.pred_df is not None else "Missing")
    with col2:
        st.metric("Confusion Matrix", "Loaded" if st.session_state.cm_df is not None else "Missing")
    with col3:
        st.metric("Classification Report", "Loaded" if st.session_state.cr_df is not None else "Missing")
    
    if st.session_state.pred_df is not None:
        st.markdown("### Quick Statistics")
        cols = st.columns(4)
        with cols[0]:
            accuracy = st.session_state.pred_df['correct'].mean()
            st.metric("Overall Accuracy", f"{accuracy:.1%}")
        with cols[1]:
            avg_conf = st.session_state.pred_df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
        with cols[2]:
            class_count = st.session_state.pred_df['predicted'].nunique()
            st.metric("Classes", class_count)
        with cols[3]:
            total_pred = len(st.session_state.pred_df)
            st.metric("Total Predictions", total_pred)

# Model Performance page
def performance_page():
    st.title("📊 Model Performance")
    
    if st.session_state.pred_df is None:
        st.error("Prediction data not available. Please check the data files.")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs(["Confidence Analysis", "Accuracy Metrics", "Class Distribution", "Misclassification Insights"])
    
    with tab1:
        st.subheader("Prediction Confidence Histogram")
        fig, ax = plt.subplots()
        sns.histplot(data=st.session_state.pred_df, x="confidence", hue="correct", bins=30, 
                    kde=True, palette="Set2", multiple="stack")
        plt.title("Prediction Confidence by Correctness")
        st.pyplot(fig)
        
        st.subheader("Per-Class Confidence Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=st.session_state.pred_df, x='predicted', y="confidence", palette="Set3")
        plt.xticks(rotation=45)
        plt.title("Confidence Distribution by Class")
        st.pyplot(fig)
        
        confidence_accuracy()
    
    with tab2:
        st.subheader("Correct vs Incorrect Predictions")
        correct_counts = st.session_state.pred_df["correct"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(correct_counts, labels=correct_counts.index, autopct="%1.1f%%", 
              startangle=90, colors=["#66c2a5", "#fc8d62"])
        plt.title("Overall Prediction Correctness")
        st.pyplot(fig)
        
        st.subheader("Confidence Statistics by Class")
        conf_stats = st.session_state.pred_df.groupby('predicted')['confidence'].agg(
            ['mean', 'std', 'count'])
        st.dataframe(conf_stats.style.background_gradient(cmap='Blues'))
    
    with tab3:
        class_distribution()
        
        st.subheader("Class-wise Accuracy")
        class_acc = st.session_state.pred_df.groupby('predicted')['correct'].mean().sort_values()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=class_acc.index, y=class_acc.values, palette="viridis")
        plt.xticks(rotation=45)
        plt.ylabel("Accuracy")
        plt.axhline(y=0.9, color='r', linestyle='--')
        st.pyplot(fig)
    
    with tab4:
        top_misclassifications()
        
        st.subheader("False Positive vs False Negative Rates")
        cm = st.session_state.cm_df.copy()
        total = cm.sum(axis=1)
        fp = (cm.sum(axis=0) - np.diag(cm)) / total
        fn = (cm.sum(axis=1) - np.diag(cm)) / total
        
        error_rates = pd.DataFrame({
            'Class': cm.index,
            'False Positive': fp,
            'False Negative': fn
        }).melt(id_vars='Class', var_name='Error Type', value_name='Rate')
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=error_rates, x='Class', y='Rate', hue='Error Type', palette="RdBu")
        plt.xticks(rotation=45)
        plt.title("Error Rates by Class")
        st.pyplot(fig)

# Confusion Matrix page
def confusion_matrix_page():
    st.title("📈 Confusion Matrix")
    
    if st.session_state.cm_df is None:
        st.error("Confusion matrix data not available. Please check the data files.")
        return
    
    tab1, tab2 = st.tabs(["Standard View", "Detailed Analysis"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(st.session_state.cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix")
        st.pyplot(fig)
        
        total = st.session_state.cm_df.values.sum()
        correct = np.trace(st.session_state.cm_df)
        st.metric("Classification Accuracy", f"{correct/total:.1%}")
    
    with tab2:
        misclassification_heatmap()
        
        st.subheader("Most Confused Classes")
        cm = st.session_state.cm_df.copy()
        np.fill_diagonal(cm.values, 0)
        confused_pairs = cm.stack().sort_values(ascending=False).head(10)
        st.dataframe(confused_pairs.reset_index().rename(
            columns={'level_0': 'Actual', 'level_1': 'Predicted', 0: 'Count'}))

# Classification Report page
def classification_report_page():
    st.title("📝 Classification Report")
    
    if st.session_state.cr_df is None:
        st.error("Classification report data not available. Please check the data files.")
        return
    
    tab1, tab2, tab3 = st.tabs(["Metrics Table", "Visualization", "Precision-Recall"])
    
    with tab1:
        st.subheader("Classification Metrics")
        st.dataframe(st.session_state.cr_df.style.background_gradient(cmap='YlGnBu', axis=1))
    
    with tab2:
        st.subheader("Precision, Recall and F1-score by Class")
        cr_plot_df = st.session_state.cr_df.drop(
            index=["accuracy", "macro avg", "weighted avg"], errors='ignore')
        cr_plot_df = cr_plot_df[["precision", "recall", "f1-score"]].reset_index().melt(
            id_vars="index", var_name="Metric", value_name="Score")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=cr_plot_df, x="index", y="Score", hue="Metric", palette="Set2")
        plt.title("Precision, Recall and F1-score per Class")
        plt.ylabel("Score")
        plt.xlabel("Class")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with tab3:
        pr_curve()
        
        st.subheader("F1-Score Distribution")
        f1_scores = st.session_state.cr_df.drop(
            index=["accuracy", "macro avg", "weighted avg"], errors='ignore')['f1-score']
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(f1_scores, bins=10, kde=True)
        plt.xlabel("F1-Score")
        plt.title("Distribution of Class F1-Scores")
        st.pyplot(fig)

# Main app logic
def main():
    page = navigation()
    
    if page == "Home":
        home_page()
    elif page == "Model Performance":
        performance_page()
    elif page == "Confusion Matrix":
        confusion_matrix_page()
    elif page == "Classification Report":
        classification_report_page()

if __name__ == "__main__":
    main()

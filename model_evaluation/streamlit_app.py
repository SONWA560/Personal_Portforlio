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

# Add interactive filters in sidebar
def add_filters():
    st.sidebar.header("Filters")
    if st.session_state.pred_df is not None:
        # Class filter
        all_classes = st.session_state.pred_df['predicted'].unique()
        selected_classes = st.sidebar.multiselect(
            "Select classes to display",
            options=all_classes,
            default=all_classes
        )
        
        # Confidence threshold
        min_confidence = st.sidebar.slider(
            "Minimum confidence level",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05
        )
        
        # Correctness filter
        correctness_filter = st.sidebar.radio(
            "Prediction correctness",
            options=["All", "Correct only", "Incorrect only"],
            index=0
        )
        
        return selected_classes, min_confidence, correctness_filter
    return None, None, None

# Navigation function
def navigation():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Model Performance", "Confusion Matrix", "Classification Report"])
    return page

# Visualization Functions with Plotly
def class_distribution():
    st.subheader("Class Distribution in Predictions")
    class_counts = st.session_state.pred_df['predicted'].value_counts().reset_index()
    class_counts.columns = ['Class', 'Count']
    
    fig = px.bar(class_counts, x='Class', y='Count', color='Class',
                 title="Distribution of Predictions Across Classes")
    fig.update_layout(xaxis_title="Class", yaxis_title="Number of Predictions")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Interpretation:**
    - Shows how predictions are distributed across different waste classes
    - Balanced distribution suggests good representation in the dataset
    - Skewed distribution may indicate class imbalance issues
    - Ideally, all classes should have similar counts for balanced performance
    """)

def confidence_analysis(selected_classes, min_confidence, correctness_filter):
    st.subheader("Confidence Analysis")
    
    # Apply filters
    filtered_df = st.session_state.pred_df.copy()
    filtered_df = filtered_df[filtered_df['predicted'].isin(selected_classes)]
    filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]
    
    if correctness_filter == "Correct only":
        filtered_df = filtered_df[filtered_df['correct']]
    elif correctness_filter == "Incorrect only":
        filtered_df = filtered_df[~filtered_df['correct']]
    
    # Confidence distribution
    fig1 = px.histogram(filtered_df, x='confidence', color='correct',
                       nbins=30, barmode='overlay',
                       title="Prediction Confidence Distribution")
    fig1.update_layout(xaxis_title="Confidence Score", yaxis_title="Count")
    
    # Confidence by class
    fig2 = px.box(filtered_df, x='predicted', y='confidence', color='correct',
                 title="Confidence Distribution by Class")
    fig2.update_layout(xaxis_title="Class", yaxis_title="Confidence Score")
    
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("""
    **Interpretation:**
    - Confidence scores should be higher for correct predictions (good model calibration)
    - Wide confidence ranges suggest uncertainty in some predictions
    - Classes with consistently low confidence may need more training data
    - The 0.9 threshold (red line) indicates our target confidence level
    - Gaps between correct/incorrect distributions show model's discriminative power
    """)

def performance_metrics(selected_classes):
    st.subheader("Performance Metrics")
    
    # Calculate metrics
    metrics_df = st.session_state.pred_df.groupby('predicted').agg(
        accuracy=('correct', 'mean'),
        avg_confidence=('confidence', 'mean'),
        count=('predicted', 'count')
    ).reset_index()
    metrics_df = metrics_df[metrics_df['predicted'].isin(selected_classes)]
    
    # Create subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy by Class", "Average Confidence by Class"))
    
    # Accuracy plot
    fig.add_trace(
        go.Bar(x=metrics_df['predicted'], y=metrics_df['accuracy'], 
              name="Accuracy", marker_color='#636EFA'),
        row=1, col=1
    )
    fig.add_hline(y=0.9, line_dash="dot", line_color="red", row=1, col=1)
    
    # Confidence plot
    fig.add_trace(
        go.Bar(x=metrics_df['predicted'], y=metrics_df['avg_confidence'], 
              name="Avg Confidence", marker_color='#00CC96'),
        row=1, col=2
    )
    fig.add_hline(y=0.9, line_dash="dot", line_color="red", row=1, col=2)
    
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Interpretation:**
    - Accuracy should be above the 0.9 threshold (red line) for all classes
    - Classes below threshold need investigation (data quality or model limitations)
    - Confidence should correlate with accuracy (higher confidence → higher accuracy)
    - Large gaps between accuracy and confidence suggest calibration issues
    - The relative performance across classes shows model strengths/weaknesses
    """)

# [Rest of your functions (home_page, performance_page, etc.) would be similarly updated...]

# Main app logic
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

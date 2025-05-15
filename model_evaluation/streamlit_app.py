import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import plotly.express as px

# Initialize session state
if 'pred_df' not in st.session_state:
    st.session_state.pred_df = None
if 'cm_df' not in st.session_state:
    st.session_state.cm_df = None
if 'cr_df' not in st.session_state:
    st.session_state.cr_df = None

# Path handling
try:
    SCRIPT_DIR = Path(__file__).parent.resolve()
except:
    SCRIPT_DIR = Path(os.getcwd())

DATA_DIR = SCRIPT_DIR / "7-class"
PRED_FILE = DATA_DIR / "detailed_predictions.csv"
CM_FILE = DATA_DIR / "confusion_matrix.csv" 
CR_FILE = DATA_DIR / "classification_report.csv"

# Load data
if st.session_state.pred_df is None and PRED_FILE.exists():
    st.session_state.pred_df = pd.read_csv(PRED_FILE)
    st.session_state.pred_df.columns = st.session_state.pred_df.columns.str.strip().str.lower()

# Filters
def add_filters():
    st.sidebar.header("Filters")
    if st.session_state.pred_df is not None:
        selected_classes = st.sidebar.multiselect(
            "Classes",
            options=st.session_state.pred_df['predicted'].unique(),
            default=st.session_state.pred_df['predicted'].unique()
        )
        min_confidence = st.sidebar.slider(
            "Min Confidence",
            min_value=0.0, max_value=1.0, value=0.7, step=0.05
        )
        correctness = st.sidebar.radio(
            "Show",
            options=["All", "Correct", "Incorrect"]
        )
        return selected_classes, min_confidence, correctness
    return None, None, None

# Pages
def home_page():
    st.title("Home")
    if st.session_state.pred_df is not None:
        st.write(f"Total predictions: {len(st.session_state.pred_df)}")

def performance_page(selected_classes, min_confidence, correctness_filter):
    st.title("Performance")
    if st.session_state.pred_df is None:
        return
    
    # Apply filters
    df = st.session_state.pred_df.copy()
    df = df[df['predicted'].isin(selected_classes)]
    df = df[df['confidence'] >= min_confidence]
    
    if correctness_filter == "Correct":
        df = df[df['correct']]
    elif correctness_filter == "Incorrect":
        df = df[~df['correct']]
    
    # Visualization
    fig = px.box(df, x='predicted', y='confidence', color='correct',
                title="Confidence by Class")
    st.plotly_chart(fig)

# Main app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Performance"])
    
    selected_classes, min_confidence, correctness_filter = add_filters()
    
    if page == "Home":
        home_page()
    elif page == "Performance":
        performance_page(selected_classes, min_confidence, correctness_filter)

if __name__ == "__main__":
    main()

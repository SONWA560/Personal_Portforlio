import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(page_title="7-Class Model Evaluation", layout="wide")

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

# Load data with column verification
def load_data():
    # Load prediction data
    if PRED_FILE.exists():
        try:
            df = pd.read_csv(PRED_FILE)
            df.columns = df.columns.str.strip().str.lower()
            
            # Check for prediction column with multiple possible names
            pred_col = None
            possible_cols = ['predicted', 'prediction', 'predicted_label', 'predicted_class']
            for col in possible_cols:
                if col in df.columns:
                    pred_col = col
                    break
            
            if pred_col is None:
                st.error(f"Could not find prediction column. Available columns: {df.columns.tolist()}")
                return None, None, None
            
            # Standardize column names
            df = df.rename(columns={pred_col: 'predicted'})
            
            # Verify required columns exist
            required_cols = {'predicted', 'confidence', 'correct'}
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                return None, None, None
                
            # Load other files
            cm_df = pd.read_csv(CM_FILE, index_col=0) if CM_FILE.exists() else None
            cr_df = pd.read_csv(CR_FILE, index_col=0) if CR_FILE.exists() else None
            
            return df, cm_df, cr_df
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None, None, None
    return None, None, None

# Load data if not already loaded
if st.session_state.pred_df is None:
    st.session_state.pred_df, st.session_state.cm_df, st.session_state.cr_df = load_data()

# Filters
def add_filters():
    st.sidebar.header("Filters")
    
    if st.session_state.pred_df is None:
        return [], 0.0, "All"  # Return safe defaults when no data
    
    try:
        # Class filter
        classes = st.session_state.pred_df['predicted'].unique().tolist()
        selected_classes = st.sidebar.multiselect(
            "Select classes",
            options=classes,
            default=classes
        )
        
        # Confidence threshold
        min_confidence = st.sidebar.slider(
            "Minimum confidence",
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
        
    except Exception as e:
        st.error(f"Filter error: {str(e)}")
        return [], 0.0, "All"

# Apply filters to data
def apply_filters(df, selected_classes, min_confidence, correctness_filter):
    if df is None:
        return None
        
    filtered_df = df.copy()
    
    # Apply class filter
    if selected_classes:
        filtered_df = filtered_df[filtered_df['predicted'].isin(selected_classes)]
    
    # Apply confidence filter
    filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]
    
    # Apply correctness filter
    if correctness_filter == "Correct only":
        filtered_df = filtered_df[filtered_df['correct']]
    elif correctness_filter == "Incorrect only":
        filtered_df = filtered_df[~filtered_df['correct']]
    
    return filtered_df

# Visualization Functions (restored from your original)
def class_distribution(df):
    st.subheader("Class Distribution in Predictions")
    class_counts = df['predicted'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
    plt.xticks(rotation=45)
    plt.ylabel("Count")
    st.pyplot(fig)
    
    st.markdown("""
    **Interpretation:**
    - Shows distribution of predictions across classes
    - Balanced distribution indicates good representation
    - Significant imbalances may require class weighting
    """)

def confidence_analysis(df):
    st.subheader("Confidence Analysis")
    
    # Histogram
    fig1 = px.histogram(df, x="confidence", color="correct", 
                       nbins=30, barmode="overlay",
                       title="Confidence Distribution by Correctness")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Boxplot
    fig2 = px.box(df, x="predicted", y="confidence", color="correct",
                 title="Confidence by Class and Correctness")
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("""
    **Interpretation:**
    - Higher confidence for correct predictions indicates good calibration
    - Overlapping distributions suggest areas for model improvement
    - Classes with wider confidence ranges need investigation
    """)

def confusion_matrix_analysis():
    if st.session_state.cm_df is None:
        st.warning("No confusion matrix data available")
        return
    
    st.subheader("Confusion Matrix Analysis")
    
    # Normalized heatmap
    norm_cm = st.session_state.cm_df.div(st.session_state.cm_df.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(norm_cm, annot=True, fmt=".1%", cmap="Blues")
    plt.title("Normalized Confusion Matrix")
    st.pyplot(fig)
    
    st.markdown("""
    **Interpretation:**
    - Diagonal shows correct classification rates
    - Off-diagonal reveals common misclassifications
    - Normalization accounts for class imbalance
    """)

# Pages
def home_page():
    st.title("🧠 Waste Classification Model Evaluation")
    
    if st.session_state.pred_df is not None:
        st.markdown("### Quick Statistics")
        cols = st.columns(3)
        with cols[0]:
            accuracy = st.session_state.pred_df['correct'].mean()
            st.metric("Accuracy", f"{accuracy:.1%}")
        with cols[1]:
            classes = st.session_state.pred_df['predicted'].nunique()
            st.metric("Classes", classes)
        with cols[2]:
            conf = st.session_state.pred_df['confidence'].mean()
            st.metric("Avg Confidence", f"{conf:.1%}")

def performance_page(selected_classes, min_confidence, correctness_filter):
    st.title("📊 Model Performance")
    
    if st.session_state.pred_df is None:
        st.error("No prediction data available")
        return
    
    # Apply filters
    filtered_df = apply_filters(st.session_state.pred_df, selected_classes, min_confidence, correctness_filter)
    
    if filtered_df is None or filtered_df.empty:
        st.warning("No data matches the selected filters")
        return
    
    # Visualizations
    tab1, tab2 = st.tabs(["Distribution", "Confidence"])
    
    with tab1:
        class_distribution(filtered_df)
        
    with tab2:
        confidence_analysis(filtered_df)
        
    # Confusion matrix (not filtered)
    confusion_matrix_analysis()

# Main app
def main():
    selected_classes, min_confidence, correctness_filter = add_filters()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Performance"])
    
    if page == "Home":
        home_page()
    elif page == "Performance":
        performance_page(selected_classes, min_confidence, correctness_filter)

if __name__ == "__main__":
    main()

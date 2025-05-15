import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Set page config
st.set_page_config(page_title="7-Class Model Evaluation", layout="wide")

# Initialize session state
if 'pred_df' not in st.session_state:
    try:
        # Load data (update path as needed)
        st.session_state.pred_df = pd.read_csv("detailed_predictions.csv")
        
        # Verify and rename columns if needed
        column_map = {
            'True_Label': 'true_label',
            'Predicted_Label': 'predicted',
            'Correct': 'correct',
            'Confidence': 'confidence'
        }
        st.session_state.pred_df = st.session_state.pred_df.rename(columns=column_map)
        
        # Convert boolean column if needed
        if st.session_state.pred_df['correct'].dtype == object:
            st.session_state.pred_df['correct'] = st.session_state.pred_df['correct'].map({'True': True, 'False': False})
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.session_state.pred_df = None

# Home Page with Data Validation
def home_page():
    st.title("🧠 Waste Classification Model Evaluation")
    
    if st.session_state.pred_df is None:
        st.error("Data not loaded. Please check the data file.")
        return
    
    # Data validation
    required_cols = {'true_label', 'predicted', 'correct', 'confidence'}
    missing_cols = required_cols - set(st.session_state.pred_df.columns)
    
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        st.write("Available columns:", st.session_state.pred_df.columns.tolist())
        return
    
    # Calculate metrics
    accuracy = st.session_state.pred_df['correct'].mean()
    class_count = st.session_state.pred_df['predicted'].nunique()
    avg_confidence = st.session_state.pred_df['confidence'].mean()
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.1%}")
    with col2:
        st.metric("Classes", class_count)
    with col3:
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    # Class distribution
    st.subheader("Class Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(data=st.session_state.pred_df, x='predicted', 
                 order=st.session_state.pred_df['predicted'].value_counts().index)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Performance Page
def performance_page():
    st.title("📊 Model Performance")
    
    if st.session_state.pred_df is None:
        return
    
    tab1, tab2 = st.tabs(["Confidence Analysis", "Class Probabilities"])
    
    with tab1:
        st.subheader("Prediction Confidence")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=st.session_state.pred_df, x='predicted', y='confidence',
                   hue='correct', palette={True: 'green', False: 'red'})
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        st.pyplot(fig)
        
    with tab2:
        st.subheader("Class Probability Distributions")
        prob_cols = [c for c in st.session_state.pred_df.columns if c.startswith('Prob_')]
        selected_class = st.selectbox("Select class", prob_cols)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(data=st.session_state.pred_df, x=selected_class, bins=20, kde=True)
        plt.xlim(0, 1)
        st.pyplot(fig)

# Main App Logic
def main():
    page = st.sidebar.selectbox("Menu", ["Home", "Performance"])
    
    if page == "Home":
        home_page()
    elif page == "Performance":
        performance_page()

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# === File paths ===
BASE_DIR = "/Users/sonwabise/Documents/Anaconda/Python/venv/Multi Class classification"
CM_FILE = os.path.join(BASE_DIR, "confusion_matrix.csv")
CR_FILE = os.path.join(BASE_DIR, "classification_report.csv")

# === Load Data Once ===
@st.cache_data
def load_data():
    try:
        cm_raw = pd.read_csv(CM_FILE, index_col=0)
        cm_df = cm_raw.apply(pd.to_numeric, errors='coerce').fillna(0)
    except Exception as e:
        st.error(f"Error loading confusion matrix: {e}")
        cm_df = None

    try:
        cr_df = pd.read_csv(CR_FILE, index_col=0)
    except Exception as e:
        st.error(f"Error loading classification report: {e}")
        cr_df = None

    return cm_df, cr_df

# === Initialize session state ===
if 'cm_df' not in st.session_state or 'cr_df' not in st.session_state:
    st.session_state.cm_df, st.session_state.cr_df = load_data()

# === Confusion Matrix Page ===
def confusion_matrix_page():
    st.title("📈 Confusion Matrix Analysis")
    
    if st.session_state.cm_df is None:
        st.error("Confusion matrix data not available. Please check the data files.")
        return
    
    st.sidebar.header("Confusion Matrix Filters")
    norm_type = st.sidebar.radio("Normalization", 
                                 ["Counts", "By True Class", "By Predicted Class"],
                                 index=1)
    
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
    
    fig = px.imshow(norm_cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=norm_cm.columns,
                    y=norm_cm.index,
                    text_auto=fmt,
                    color_continuous_scale='Blues',
                    zmin=zmin, zmax=zmax)
    
    fig.update_xaxes(side="top")
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)
    
    total = cm.values.sum()
    correct = np.trace(cm.values)
    accuracy = correct / total

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Accuracy", f"{accuracy:.1%}", help="Percentage of all correct predictions")
    with col2:
        st.metric("Correct Predictions", f"{correct:,}", help="Total number of correct classifications")
    with col3:
        st.metric("Total Samples", f"{total:,}", help="Total number of predictions")

    st.markdown("""
    **Interpretation:**
    - Diagonal elements show correct classifications (higher is better)
    - Off-diagonal elements show misclassifications (lower is better)
    - Normalized view helps compare performance across classes
    - Common misclassification patterns reveal model weaknesses
    - Ideal matrix would have all values on the diagonal
    """)

    st.subheader("Top Misclassification Pairs")
    misclass = cm.copy()
    np.fill_diagonal(misclass.values, 0)
    top_misclass = misclass.stack().sort_values(ascending=False).head(10).reset_index()
    top_misclass.columns = ['Actual', 'Predicted', 'Count']
    
    fig2 = px.bar(top_misclass, 
                  x='Count', 
                  y='Actual', 
                  color='Predicted',
                  orientation='h',
                  title="Most Common Misclassifications")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    **Interpretation:**
    - Shows which classes are most frequently confused
    - Pairs with high counts may need better feature differentiation
    """)

# === Classification Report Page ===
def classification_report_page():
    st.title("📝 Classification Report Analysis")
    
    if st.session_state.cr_df is None:
        st.error("Classification report data not available. Please check the data files.")
        return
    
    st.sidebar.header("Report Filters")
    show_avg = st.sidebar.checkbox("Show Averages", value=True)
    metric = st.sidebar.selectbox("Primary Metric", ["precision", "recall", "f1-score"], index=2)
    
    cr = st.session_state.cr_df.copy()
    if not show_avg:
        cr = cr.drop(index=["accuracy", "macro avg", "weighted avg"], errors='ignore')

    fig = px.bar(cr, 
                 y=cr.index, 
                 x=[metric, "precision", "recall"],
                 barmode='group',
                 title="Classification Metrics by Class",
                 labels={'value': 'Score', 'variable': 'Metric'})
    
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    fig.add_vline(x=0.9, line_dash="dot", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Interpretation:**
    - Precision: When the model predicts this class, how often is it correct
    - Recall: What percentage of actual class instances were identified correctly
    - F1-score: Harmonic mean of precision and recall (ideal balance)
    """)

    st.subheader("Precision-Recall Tradeoff")
    cr_filtered = cr.drop(index=["accuracy", "macro avg", "weighted avg"], errors='ignore')
    
    fig2 = px.scatter(cr_filtered,
                      x='recall',
                      y='precision',
                      text=cr_filtered.index,
                      size='f1-score',
                      color='f1-score',
                      hover_name=cr_filtered.index,
                      title="Precision vs Recall by Class")
    
    fig2.update_traces(textposition='top center')
    fig2.add_shape(type="line", x0=0, y0=1, x1=1, y1=0, line=dict(dash="dash"))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Detailed Metrics")
    st.dataframe(cr.style
                 .background_gradient(subset=['precision', 'recall', 'f1-score'], cmap='YlGnBu')
                 .format({'precision': '{:.1%}', 'recall': '{:.1%}', 'f1-score': '{:.1%}'}),
                 height=600)
    
    st.markdown("""
    **Key Metrics:**
    - **Support**: Number of actual occurrences in dataset
    - **Macro Avg**: Unweighted mean of all classes
    - **Weighted Avg**: Support-weighted mean of all classes
    - **Accuracy**: Overall correct prediction rate
    """)

# === Main App ===
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Confusion Matrix", "Classification Report"])

    if page == "Confusion Matrix":
        confusion_matrix_page()
    elif page == "Classification Report":
        classification_report_page()

if __name__ == "__main__":
    main()

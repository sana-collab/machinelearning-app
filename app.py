import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image
import random
import os

# Import our utility modules
from utils.data_processing import load_yahoo_finance_data, process_data, engineer_features, split_data
from utils.ml_models import (
    train_linear_regression, train_logistic_regression, train_kmeans,
    evaluate_linear_regression, evaluate_logistic_regression, evaluate_kmeans,
    save_model
)
from utils.visualization import (
    plot_data_overview, plot_missing_values, plot_feature_importance,
    plot_train_test_split, plot_regression_results, plot_classification_results,
    plot_clusters, plot_pca_clusters, plot_time_series
)

# Set page config
st.set_page_config(
    page_title="Upside Down Finance | ML Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open("assets/custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'engineered_data' not in st.session_state:
    st.session_state.engineered_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_info' not in st.session_state:
    st.session_state.model_info = None
if 'evaluation' not in st.session_state:
    st.session_state.evaluation = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'clusters' not in st.session_state:
    st.session_state.clusters = None

# Define Stranger Things themed palette
ST_RED = "#e6194b"
ST_BLACK = "#000000"
ST_DARK_GRAY = "#1e1e1e"

# Helper function to create a flicker effect on text
def flicker_text(text, tag="h1"):
    html = f"""
    <{tag} class='flicker-text' style='
        color: {ST_RED};
        text-shadow: 0 0 10px {ST_RED}, 0 0 20px {ST_RED};
        animation: flicker 1.5s infinite alternate;
        font-family: "Courier New", monospace;
        text-transform: uppercase;
        letter-spacing: 2px;
    '>
        {text}
    </{tag}>
    <style>
        @keyframes flicker {{
            0%, 18%, 22%, 25%, 53%, 57%, 100% {{
                text-shadow: 0 0 10px {ST_RED}, 0 0 20px {ST_RED};
            }}
            20%, 24%, 55% {{
                text-shadow: none;
            }}
        }}
    </style>
    """
    return html

# Display an animated stranger things inspired title
st.markdown(flicker_text("Upside Down Finance"), unsafe_allow_html=True)
st.markdown(flicker_text("Machine Learning Analytics Portal", "h3"), unsafe_allow_html=True)

# Sidebar
st.sidebar.image("assets/images/sidebar_image.svg", use_container_width=True)
st.sidebar.title("Navigation")

# Define navigation sections for easier reference
nav_sections = ["Welcome", "Data Loading", "Preprocessing", "Feature Engineering", 
               "Train/Test Split", "Model Training", "Evaluation", "Results Visualization"]

# Set the default selection based on current_step
default_nav_index = min(st.session_state.current_step, len(nav_sections) - 1)

# Main navigation
main_nav = st.sidebar.radio(
    "Choose a Section",
    nav_sections,
    index=default_nav_index
)

# Update current_step if user manually changes the navigation
if nav_sections.index(main_nav) != st.session_state.current_step:
    st.session_state.current_step = nav_sections.index(main_nav)

# Welcome section
if main_nav == "Welcome":
    st.markdown("## Welcome to the Upside Down World of Financial Machine Learning")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        In this strange dimension, we'll explore financial data using machine learning algorithms.
        Navigate through the portal to discover insights lurking in your data.
        
        This application allows you to:
        - Upload your own financial datasets or fetch real-time stock data
        - Clean and preprocess your data to prepare for analysis
        - Engineer new features to enhance your models
        - Train machine learning models on your data
        - Evaluate model performance and visualize results
        
        **Ready to begin your journey into the Upside Down?**
        
        Use the sidebar to navigate through each step of the machine learning pipeline.
        """)
    
    with col2:
        st.image("assets/images/welcome_image.svg", use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("## About This Portal")
    st.markdown("""
    This application was created for the AF3005 – Programming for Finance course at FAST-NUCES, Islamabad.
    
    The portal features:
    - Interactive machine learning workflow
    - Real-time financial data analysis
    - Stranger Things inspired UI design
    - Comprehensive visualization tools
    """)
    
    if st.button("Begin Your Journey", key="welcome_button"):
        st.session_state.current_step = 1
        st.rerun()

# Data Loading section
elif main_nav == "Data Loading":
    st.markdown(flicker_text("Enter the Portal: Data Loading", "h2"), unsafe_allow_html=True)
    
    st.markdown("""
    Choose your data source to begin the analysis. You can either:
    - Upload your own financial dataset (CSV format)
    - Fetch real-time stock market data from Yahoo Finance
    """)
    
    data_source = st.radio(
        "Select Data Source",
        ["Upload CSV Dataset", "Fetch Yahoo Finance Data"]
    )
    
    if data_source == "Upload CSV Dataset":
        uploaded_file = st.file_uploader("Upload your financial dataset (CSV file)", type=["csv"])
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                
                st.success("✅ Dataset successfully loaded!")
                
                st.subheader("Dataset Preview")
                st.dataframe(data.head())
                
                st.subheader("Dataset Information")
                buffer = BytesIO()
                data.info(buf=buffer)
                info_str = buffer.getvalue().decode("utf-8")
                st.text(info_str)
                
                st.subheader("Dataset Statistics")
                st.dataframe(data.describe())
                
                # Show data overview visualization
                st.subheader("Data Visualization")
                fig = plot_data_overview(data)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading the dataset: {str(e)}")
    
    else:  # Fetch Yahoo Finance Data
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ticker = st.text_input("Stock Ticker Symbol", "AAPL")
        
        with col2:
            start_date = st.date_input(
                "Start Date",
                datetime.now() - timedelta(days=365)
            )
        
        with col3:
            end_date = st.date_input(
                "End Date",
                datetime.now()
            )
        
        if st.button("Fetch Stock Data", key="fetch_data"):
            with st.spinner("Fetching data from Yahoo Finance..."):
                data = load_yahoo_finance_data(ticker, start_date, end_date)
                
                if data is not None:
                    st.session_state.data = data
                    
                    st.success(f"✅ Stock data for {ticker} successfully loaded!")
                    
                    st.subheader("Dataset Preview")
                    st.dataframe(data.head())
                    
                    st.subheader("Dataset Statistics")
                    st.dataframe(data.describe())
                    
                    # Plot time series data
                    st.subheader("Stock Price History")
                    fig = plot_time_series(
                        data, 
                        "Date", 
                        ["Close", "Open", "High", "Low"],
                        f"{ticker} Stock Price History"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data overview visualization
                    st.subheader("Data Overview")
                    fig = plot_data_overview(data)
                    st.plotly_chart(fig, use_container_width=True)
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    
    with col2:
        if st.session_state.data is not None:
            if st.button("Proceed to Preprocessing", key="to_preprocessing"):
                st.session_state.current_step = 2
                st.rerun()

# Preprocessing section
elif main_nav == "Preprocessing":
    st.markdown(flicker_text("Clean the Transmission: Data Preprocessing", "h2"), unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ No data loaded. Please go back to the Data Loading step.")
        if st.button("Go to Data Loading", key="to_data_loading_from_preprocessing"):
            st.session_state.current_step = 1
            st.rerun()
    else:
        st.markdown("""
        Prepare your data for the Upside Down by cleaning and transforming it. 
        This step handles missing values, outliers, and other data issues.
        """)
        
        st.subheader("Original Data Preview")
        st.dataframe(st.session_state.data.head())
        
        # Missing values visualization
        st.subheader("Missing Values Analysis")
        fig = plot_missing_values(st.session_state.data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Preprocessing options
        st.subheader("Preprocessing Options")
        
        # Choose columns to drop
        all_columns = st.session_state.data.columns.tolist()
        drop_columns = st.multiselect(
            "Select columns to drop",
            all_columns,
            default=[]
        )
        
        # Button to process data
        if st.button("Process Data", key="process_data"):
            with st.spinner("Preprocessing data..."):
                # Add slight delay for visual effect
                time.sleep(1)
                
                processed_data, process_info = process_data(st.session_state.data, drop_columns=drop_columns)
                
                if process_info["success"]:
                    st.session_state.processed_data = processed_data
                    
                    st.success("✅ Data successfully preprocessed!")
                    
                    # Show preprocessing info
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Rows Before", 
                            process_info["original_shape"][0]
                        )
                    
                    with col2:
                        st.metric(
                            "Rows After", 
                            processed_data.shape[0], 
                            delta=-process_info["rows_removed"]
                        )
                    
                    with col3:
                        st.metric(
                            "Missing Values Removed", 
                            process_info["missing_values_before"] - process_info["missing_values_after"]
                        )
                    
                    st.subheader("Processed Data Preview")
                    st.dataframe(processed_data.head())
                    
                    # Show data overview visualization after preprocessing
                    st.subheader("Processed Data Overview")
                    fig = plot_data_overview(processed_data)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"❌ Error during preprocessing: {process_info['message']}")
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Back to Data Loading", key="back_to_data_loading"):
                st.session_state.current_step = 1
                st.rerun()
        
        with col2:
            if st.session_state.processed_data is not None:
                if st.button("Proceed to Feature Engineering", key="to_feature_engineering"):
                    st.session_state.current_step = 3
                    st.rerun()

# Feature Engineering section
elif main_nav == "Feature Engineering":
    st.markdown(flicker_text("Enhance the Signal: Feature Engineering", "h2"), unsafe_allow_html=True)
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ No processed data available. Please complete the Preprocessing step first.")
        if st.button("Go to Preprocessing", key="to_preprocessing_from_features"):
            st.session_state.current_step = 2
            st.rerun()
    else:
        st.markdown("""
        Create new features to strengthen your connection to the Upside Down. 
        Feature engineering can improve model performance by providing more 
        meaningful information for your algorithms to learn from.
        """)
        
        st.subheader("Current Data Preview")
        st.dataframe(st.session_state.processed_data.head())
        
        # Select target column
        all_columns = st.session_state.processed_data.columns.tolist()
        
        target_column = st.selectbox(
            "Select target column for prediction (dependent variable)",
            all_columns
        )
        
        # Feature selection
        feature_selection = st.multiselect(
            "Select features to include (leave empty to use all available features)",
            [col for col in all_columns if col != target_column],
            default=[]
        )
        
        # Button to engineer features
        if st.button("Engineer Features", key="engineer_features"):
            with st.spinner("Engineering features..."):
                # Add slight delay for visual effect
                time.sleep(1)
                
                engineered_data, engineer_info = engineer_features(
                    st.session_state.processed_data, 
                    target_column, 
                    feature_selection
                )
                
                if engineer_info["success"]:
                    st.session_state.engineered_data = engineered_data
                    st.session_state.target_column = target_column
                    
                    st.success("✅ Features successfully engineered!")
                    
                    # Show feature engineering info
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Original Features", 
                            len(engineer_info["original_features"])
                        )
                    
                    with col2:
                        st.metric(
                            "New Features Created", 
                            len(engineer_info["new_features"])
                        )
                    
                    if len(engineer_info["new_features"]) > 0:
                        st.subheader("New Features Created")
                        # Convert any non-string features (like tuples) to strings
                        feature_names = [str(feature) for feature in engineer_info["new_features"]]
                        st.write(", ".join(feature_names))
                    
                    st.subheader("Engineered Data Preview")
                    st.dataframe(engineered_data.head())
                    
                    # Correlation plot
                    st.subheader("Feature Correlation Matrix")
                    corr = engineered_data.corr()
                    fig = px.imshow(
                        corr, 
                        text_auto=True, 
                        aspect="auto",
                        color_continuous_scale="RdBu_r",
                        template="plotly_dark"
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"❌ Error during feature engineering: {engineer_info['message']}")
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Back to Preprocessing", key="back_to_preprocessing"):
                st.session_state.current_step = 2
                st.rerun()
        
        with col2:
            if st.session_state.engineered_data is not None:
                if st.button("Proceed to Train/Test Split", key="to_train_test_split"):
                    st.session_state.current_step = 4
                    st.rerun()

# Train/Test Split section
elif main_nav == "Train/Test Split":
    st.markdown(flicker_text("Open the Gate: Train/Test Split", "h2"), unsafe_allow_html=True)
    
    if st.session_state.engineered_data is None:
        st.warning("⚠️ No engineered data available. Please complete the Feature Engineering step first.")
        if st.button("Go to Feature Engineering", key="to_feature_engineering_from_split"):
            st.session_state.current_step = 3
            st.rerun()
    else:
        st.markdown("""
        Divide your data into training and testing sets. The training set teaches 
        your model about patterns in the data, while the testing set evaluates 
        how well your model generalizes to new, unseen data.
        """)
        
        st.subheader("Data Ready for Splitting")
        st.dataframe(st.session_state.engineered_data.head())
        
        # Split settings
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider(
                "Test Set Size",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Proportion of the dataset to include in the test split"
            )
        
        with col2:
            random_state = st.number_input(
                "Random State",
                min_value=0,
                max_value=100,
                value=42,
                help="Seed for reproducible results"
            )
        
        # Button to split data
        if st.button("Split Data", key="split_data"):
            with st.spinner("Splitting data into training and testing sets..."):
                # Add slight delay for visual effect
                time.sleep(1)
                
                X_train, X_test, y_train, y_test, split_info = split_data(
                    st.session_state.engineered_data,
                    st.session_state.target_column,
                    test_size=test_size,
                    random_state=int(random_state)
                )
                
                if split_info["success"]:
                    # Store split data in session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.split_info = split_info
                    
                    st.success("✅ Data successfully split into training and testing sets!")
                    
                    # Show split info
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Training Set Size", 
                            split_info["train_shape"][0]
                        )
                    
                    with col2:
                        st.metric(
                            "Testing Set Size", 
                            split_info["test_shape"][0]
                        )
                    
                    # Visualize the split
                    st.subheader("Train/Test Split Visualization")
                    fig = plot_train_test_split(X_train, X_test)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Preview training and testing sets
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Training Features Preview")
                        st.dataframe(X_train.head())
                    
                    with col2:
                        st.subheader("Testing Features Preview")
                        st.dataframe(X_test.head())
                else:
                    st.error(f"❌ Error during data splitting: {split_info['message']}")
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Back to Feature Engineering", key="back_to_feature_engineering"):
                st.session_state.current_step = 3
                st.rerun()
        
        with col2:
            if st.session_state.X_train is not None:
                if st.button("Proceed to Model Training", key="to_model_training"):
                    st.session_state.current_step = 5
                    st.rerun()

# Model Training section
elif main_nav == "Model Training":
    st.markdown(flicker_text("Activate the Machine: Model Training", "h2"), unsafe_allow_html=True)
    
    if st.session_state.X_train is None:
        st.warning("⚠️ No training data available. Please complete the Train/Test Split step first.")
        if st.button("Go to Train/Test Split", key="to_split_from_training"):
            st.session_state.current_step = 4
            st.rerun()
    else:
        st.markdown("""
        Train your machine learning model on the prepared data. This step is like teaching 
        the Upside Down to recognize patterns in your financial data.
        """)
        
        # Model selection
        model_type = st.radio(
            "Select Machine Learning Model",
            ["Linear Regression", "Logistic Regression", "K-Means Clustering"]
        )
        
        # Model-specific parameters
        if model_type == "K-Means Clustering":
            n_clusters = st.slider(
                "Number of Clusters",
                min_value=2,
                max_value=10,
                value=3
            )
        
        # Button to train model
        if st.button("Train Model", key="train_model"):
            with st.spinner(f"Training {model_type} model..."):
                # Add slight delay for visual effect
                time.sleep(2)
                
                try:
                    if model_type == "Linear Regression":
                        model, model_info = train_linear_regression(
                            st.session_state.X_train, 
                            st.session_state.y_train
                        )
                    elif model_type == "Logistic Regression":
                        model, model_info = train_logistic_regression(
                            st.session_state.X_train, 
                            st.session_state.y_train
                        )
                    else:  # K-Means Clustering
                        model, model_info, clusters = train_kmeans(
                            st.session_state.X_train, 
                            n_clusters=n_clusters
                        )
                        st.session_state.clusters = clusters
                    
                    # Store model and info in session state
                    st.session_state.model = model
                    st.session_state.model_info = model_info
                    st.session_state.model_type = model_type
                    
                    st.success(f"✅ {model_type} model successfully trained!")
                    
                    # Show training info
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Model Type", 
                            model_info["model_type"]
                        )
                    
                    with col2:
                        st.metric(
                            "Training Time", 
                            f"{model_info['training_time']:.4f} seconds"
                        )
                    
                    # Model-specific information
                    if model_type in ["Linear Regression", "Logistic Regression"]:
                        # Show feature importance
                        st.subheader("Feature Importance")
                        fig = plot_feature_importance(model_info["feature_importance"])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif model_type == "K-Means Clustering":
                        # Show cluster information
                        st.subheader("Cluster Information")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                "Number of Clusters", 
                                model_info["n_clusters"]
                            )
                        
                        with col2:
                            st.metric(
                                "Silhouette Score", 
                                f"{model_info['silhouette_score']:.4f}"
                            )
                        
                        # Show cluster sizes
                        cluster_sizes = pd.DataFrame({
                            'Cluster': range(model_info["n_clusters"]),
                            'Size': model_info["cluster_sizes"]
                        })
                        
                        st.subheader("Cluster Sizes")
                        fig = px.bar(
                            cluster_sizes,
                            x='Cluster',
                            y='Size',
                            title='Number of Samples in Each Cluster',
                            color='Size',
                            color_continuous_scale='Plasma',
                            template="plotly_dark"
                        )
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show cluster centers
                        st.subheader("Cluster Centers")
                        st.dataframe(model_info["cluster_centers"])
                        
                        # PCA visualization of clusters
                        st.subheader("PCA Visualization of Clusters")
                        fig = plot_pca_clusters(st.session_state.X_train, clusters, n_components=3)
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Error during model training: {str(e)}")
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Back to Train/Test Split", key="back_to_split"):
                st.session_state.current_step = 4
                st.rerun()
        
        with col2:
            if st.session_state.model is not None:
                if st.button("Proceed to Evaluation", key="to_evaluation"):
                    st.session_state.current_step = 6
                    st.rerun()

# Evaluation section
elif main_nav == "Evaluation":
    st.markdown(flicker_text("Test the Connection: Model Evaluation", "h2"), unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("⚠️ No trained model available. Please complete the Model Training step first.")
        if st.button("Go to Model Training", key="to_training_from_evaluation"):
            st.session_state.current_step = 5
            st.rerun()
    else:
        st.markdown("""
        Evaluate how well your model performs on unseen data. This step checks
        if your connection to the Upside Down is strong enough to make accurate predictions.
        """)
        
        # Button to evaluate model
        if st.button("Evaluate Model", key="evaluate_model"):
            with st.spinner(f"Evaluating {st.session_state.model_type} model..."):
                # Add slight delay for visual effect
                time.sleep(2)
                
                try:
                    if st.session_state.model_type == "Linear Regression":
                        evaluation = evaluate_linear_regression(
                            st.session_state.model,
                            st.session_state.X_test,
                            st.session_state.y_test
                        )
                    
                    elif st.session_state.model_type == "Logistic Regression":
                        evaluation = evaluate_logistic_regression(
                            st.session_state.model,
                            st.session_state.X_test,
                            st.session_state.y_test
                        )
                    
                    else:  # K-Means Clustering
                        clustered_data, evaluation = evaluate_kmeans(
                            st.session_state.model,
                            st.session_state.X_test,
                            pd.concat([st.session_state.X_test, st.session_state.y_test], axis=1)
                        )
                        st.session_state.clustered_data = clustered_data
                    
                    # Store evaluation in session state
                    st.session_state.evaluation = evaluation
                    
                    st.success(f"✅ {st.session_state.model_type} model successfully evaluated!")
                    
                    # Show evaluation metrics based on model type
                    if st.session_state.model_type == "Linear Regression":
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Mean Squared Error (MSE)", 
                                f"{evaluation['mse']:.4f}"
                            )
                        
                        with col2:
                            st.metric(
                                "Root Mean Squared Error (RMSE)", 
                                f"{evaluation['rmse']:.4f}"
                            )
                        
                        with col3:
                            st.metric(
                                "R² Score", 
                                f"{evaluation['r2']:.4f}"
                            )
                        
                        # Visualize regression results
                        st.subheader("Regression Results Visualization")
                        fig = plot_regression_results(evaluation["actual"], evaluation["predictions"])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif st.session_state.model_type == "Logistic Regression":
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Accuracy", 
                                f"{evaluation['accuracy']:.4f}"
                            )
                        
                        with col2:
                            st.metric(
                                "Precision", 
                                f"{evaluation['precision']:.4f}"
                            )
                        
                        with col3:
                            st.metric(
                                "Recall", 
                                f"{evaluation['recall']:.4f}"
                            )
                        
                        with col4:
                            st.metric(
                                "F1 Score", 
                                f"{evaluation['f1_score']:.4f}"
                            )
                        
                        # Visualize classification results
                        st.subheader("Classification Results Visualization")
                        fig = plot_classification_results(
                            evaluation["actual"], 
                            evaluation["predictions"],
                            evaluation["probabilities"],
                            evaluation["classes"]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    else:  # K-Means Clustering
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                "Silhouette Score", 
                                f"{evaluation['silhouette_score']:.4f}"
                            )
                        
                        # Show cluster distribution
                        st.subheader("Cluster Distribution")
                        cluster_counts = pd.DataFrame({
                            'Cluster': range(len(evaluation['cluster_counts'])),
                            'Count': evaluation['cluster_counts']
                        })
                        
                        fig = px.pie(
                            cluster_counts,
                            values='Count',
                            names='Cluster',
                            title='Sample Distribution Across Clusters',
                            color_discrete_sequence=px.colors.qualitative.Plotly,
                            template="plotly_dark"
                        )
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Visualize clusters in 2D
                        if len(st.session_state.X_test.columns) >= 2:
                            feature1, feature2 = st.session_state.X_test.columns[:2]
                            
                            st.subheader("Cluster Visualization")
                            fig = plot_clusters(
                                st.session_state.X_test,
                                evaluation["predictions"],
                                feature1,
                                feature2,
                                st.session_state.model_info["cluster_centers"]
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Error during model evaluation: {str(e)}")
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Back to Model Training", key="back_to_training"):
                st.session_state.current_step = 5
                st.rerun()
        
        with col2:
            if st.session_state.evaluation is not None:
                if st.button("Proceed to Results Visualization", key="to_visualization"):
                    st.session_state.current_step = 7
                    st.rerun()

# Results Visualization section
elif main_nav == "Results Visualization":
    st.markdown(flicker_text("Visualize the Upside Down: Results", "h2"), unsafe_allow_html=True)
    
    if st.session_state.evaluation is None:
        st.warning("⚠️ No evaluation results available. Please complete the Evaluation step first.")
        if st.button("Go to Evaluation", key="to_evaluation_from_visualization"):
            st.session_state.current_step = 6
            st.rerun()
    else:
        st.markdown("""
        See the patterns revealed from the Upside Down. This final step shows you 
        detailed visualizations of your model's predictions and insights.
        """)
        
        # Different visualizations based on model type
        if st.session_state.model_type == "Linear Regression":
            # Prepare data for visualization
            results_df = pd.DataFrame({
                'Actual': st.session_state.evaluation["actual"],
                'Predicted': st.session_state.evaluation["predictions"]
            })
            
            # Calculate error
            results_df['Error'] = results_df['Actual'] - results_df['Predicted']
            results_df['Absolute_Error'] = abs(results_df['Error'])
            results_df['Percent_Error'] = (results_df['Error'] / results_df['Actual']) * 100
            
            # Show summary statistics
            st.subheader("Prediction Results Summary")
            st.dataframe(results_df.describe())
            
            # Error distribution
            st.subheader("Error Distribution")
            fig = px.histogram(
                results_df,
                x="Error",
                nbins=30,
                title="Distribution of Prediction Errors",
                color_discrete_sequence=['#e6194b'],
                template="plotly_dark"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction scatter plot with error color
            st.subheader("Predictions with Error Magnitude")
            fig = px.scatter(
                results_df,
                x="Actual",
                y="Predicted",
                color="Absolute_Error",
                color_continuous_scale="Plasma",
                title="Prediction Accuracy with Error Magnitude",
                template="plotly_dark"
            )
            
            # Add perfect prediction line
            min_val = min(results_df["Actual"].min(), results_df["Predicted"].min())
            max_val = max(results_df["Actual"].max(), results_df["Predicted"].max())
            
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(
                        color='white',
                        dash='dash'
                    )
                )
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance recap
            st.subheader("Feature Importance Recap")
            fig = plot_feature_importance(st.session_state.model_info["feature_importance"])
            st.plotly_chart(fig, use_container_width=True)
            
            # Provide downloadable results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Prediction Results",
                data=csv,
                file_name="linear_regression_results.csv",
                mime="text/csv"
            )
        
        elif st.session_state.model_type == "Logistic Regression":
            # Prepare data for visualization
            results_df = pd.DataFrame({
                'Actual': st.session_state.evaluation["actual"],
                'Predicted': st.session_state.evaluation["predictions"]
            })
            
            # Add probabilities for each class
            probs = np.array(st.session_state.evaluation["probabilities"])
            for i, class_name in enumerate(st.session_state.evaluation["classes"]):
                results_df[f'Prob_Class_{class_name}'] = probs[:, i]
            
            # Confusion matrix visualization
            st.subheader("Confusion Matrix")
            fig = plot_classification_results(
                st.session_state.evaluation["actual"],
                st.session_state.evaluation["predictions"],
                st.session_state.evaluation["probabilities"],
                st.session_state.evaluation["classes"]
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction distribution
            st.subheader("Class Distribution")
            fig = px.histogram(
                results_df,
                x="Predicted",
                color="Actual",
                barmode="group",
                title="Distribution of Predicted vs Actual Classes",
                template="plotly_dark"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Probability distribution for binary classification
            if len(st.session_state.evaluation["classes"]) == 2:
                st.subheader("Prediction Probability Distribution")
                prob_col = f'Prob_Class_{st.session_state.evaluation["classes"][1]}'
                
                fig = px.histogram(
                    results_df,
                    x=prob_col,
                    color="Actual",
                    nbins=50,
                    opacity=0.7,
                    barmode="overlay",
                    title="Distribution of Prediction Probabilities by Actual Class",
                    template="plotly_dark"
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance recap
            st.subheader("Feature Importance Recap")
            fig = plot_feature_importance(st.session_state.model_info["feature_importance"])
            st.plotly_chart(fig, use_container_width=True)
            
            # Provide downloadable results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Prediction Results",
                data=csv,
                file_name="logistic_regression_results.csv",
                mime="text/csv"
            )
        
        else:  # K-Means Clustering
            st.subheader("Cluster Analysis")
            
            # PCA visualization
            st.subheader("PCA Visualization of Clusters")
            fig = plot_pca_clusters(
                st.session_state.X_test, 
                st.session_state.evaluation["predictions"], 
                n_components=3
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature distribution by cluster
            st.subheader("Feature Distribution by Cluster")
            
            # Get the data with cluster assignments
            if hasattr(st.session_state, "clustered_data"):
                clustered_df = st.session_state.clustered_data
            else:
                # Create one if not available
                clustered_df = pd.concat([st.session_state.X_test, st.session_state.y_test], axis=1)
                clustered_df['Cluster'] = st.session_state.evaluation["predictions"]
            
            # Select features to analyze
            feature_to_analyze = st.selectbox(
                "Select a feature to analyze across clusters",
                st.session_state.X_test.columns.tolist()
            )
            
            # Box plot of feature distribution by cluster
            fig = px.box(
                clustered_df,
                x="Cluster",
                y=feature_to_analyze,
                color="Cluster",
                title=f"Distribution of {feature_to_analyze} Across Clusters",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster statistics
            st.subheader("Cluster Statistics")
            
            # Calculate statistics for each cluster
            cluster_stats = clustered_df.groupby('Cluster').describe().reset_index()
            st.dataframe(cluster_stats)
            
            # Provide downloadable results
            clustered_df['Cluster'] = clustered_df['Cluster'].astype(int)
            csv = clustered_df.to_csv(index=False)
            st.download_button(
                label="Download Clustering Results",
                data=csv,
                file_name="kmeans_clustering_results.csv",
                mime="text/csv"
            )
        
        # Final summary and conclusion
        st.markdown("---")
        st.markdown(flicker_text("Mission Complete", "h2"), unsafe_allow_html=True)
        
        st.markdown("""
        You've successfully navigated the Upside Down world of financial machine learning!
        
        Here's what you've accomplished:
        1. Loaded and processed your financial data
        2. Engineered meaningful features
        3. Trained a machine learning model
        4. Evaluated its performance
        5. Visualized the results
        
        The insights you've gained can help inform your financial decisions and strategies.
        """)
        
        # Animated GIF for completion
        st.image("assets/images/results_image.svg", use_container_width=True)
        
        # Option to start over
        if st.button("Start Over", key="start_over"):
            # Reset session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            # Go back to welcome page
            st.session_state.current_step = 0
            st.rerun()
        
        # Navigation button
        if st.button("Back to Evaluation", key="back_to_evaluation"):
            st.session_state.current_step = 6
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
    Upside Down Finance | AF3005 – Programming for Finance | FAST-NUCES, Islamabad<br>
    Created for the Stranger Things-themed ML Finance Application Project
</div>
""", unsafe_allow_html=True)

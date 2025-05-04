import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA

def plot_data_overview(df):
    """
    Create an overview plot of the dataset
    
    Args:
        df (pandas.DataFrame): Input dataframe
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create subplots with 2 rows
    fig = make_subplots(
        rows=2, 
        cols=1,
        subplot_titles=("Data Distribution", "Correlation Heatmap"),
        vertical_spacing=0.3
    )
    
    try:
        # Only get numerical columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) > 0:
            # Plot histograms for the first subplot - use only first 5 columns to avoid overcrowding
            for i, col in enumerate(numeric_cols[:5]):
                # Convert column name to string if it's a tuple or other complex type
                col_name = str(col) if not isinstance(col, str) else col
                
                try:
                    fig.add_trace(
                        go.Histogram(
                            x=df[col],
                            name=col_name,
                            opacity=0.7,
                            marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                        ),
                        row=1, col=1
                    )
                except Exception as e:
                    st.warning(f"Could not create histogram for column '{col_name}': {str(e)}")
            
            # Add correlation heatmap for the second subplot
            try:
                corr_matrix = df[numeric_cols].corr()
                
                # Convert column names to strings if needed
                x_labels = [str(col) for col in corr_matrix.columns]
                y_labels = [str(col) for col in corr_matrix.columns]
                
                fig.add_trace(
                    go.Heatmap(
                        z=corr_matrix.values,
                        x=x_labels,
                        y=y_labels,
                        colorscale="RdBu_r",
                        zmid=0,
                        colorbar=dict(title="Correlation")
                    ),
                    row=2, col=1
                )
            except Exception as e:
                st.warning(f"Could not create correlation heatmap: {str(e)}")
    except Exception as e:
        st.error(f"Error creating data overview: {str(e)}")
        # Create a basic fallback figure
        fig.add_annotation(
            text="Could not create data visualization. Please check your data.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="white")
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Dataset Overview",
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def plot_missing_values(df):
    """
    Create a visualization of missing values in the dataset
    
    Args:
        df (pandas.DataFrame): Input dataframe
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create a DataFrame with missing values
    missing_data = []
    for column in df.columns:
        col_name = str(column) if not isinstance(column, str) else column
        missing_count = df[column].isna().sum()
        missing_percent = (missing_count / len(df)) * 100
        missing_data.append({
            'Column': col_name,
            'Missing Values': missing_count,
            'Percentage': missing_percent
        })
    
    missing = pd.DataFrame(missing_data)
    missing = missing.sort_values('Missing Values', ascending=False)
    
    # Create a bar chart
    fig = px.bar(
        missing,
        x='Column',
        y='Percentage',
        title='Missing Values by Column',
        color='Percentage',
        color_continuous_scale='Plasma',
        template="plotly_dark"
    )
    
    fig.update_layout(
        xaxis_title='Column',
        yaxis_title='Missing Values (%)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def plot_feature_importance(feature_importance):
    """
    Create a bar plot of feature importances
    
    Args:
        feature_importance (pandas.DataFrame): DataFrame with Feature and Importance columns
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=True)
    
    # Create a horizontal bar chart
    fig = px.bar(
        feature_importance.tail(15),  # Show top 15 features
        y='Feature',
        x='Importance',
        orientation='h',
        title='Feature Importance',
        color='Importance',
        color_continuous_scale='Plasma',
        template="plotly_dark"
    )
    
    fig.update_layout(
        xaxis_title='Importance',
        yaxis_title='Feature',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def plot_train_test_split(X_train, X_test):
    """
    Create a pie chart showing the train/test split
    
    Args:
        X_train: Training features
        X_test: Testing features
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Calculate proportions
    train_size = len(X_train)
    test_size = len(X_test)
    total = train_size + test_size
    
    # Create a pie chart
    fig = px.pie(
        values=[train_size, test_size],
        names=['Training Set', 'Testing Set'],
        title='Train/Test Split',
        color_discrete_sequence=['#e6194b', '#3cb44b'],
        template="plotly_dark"
    )
    
    fig.update_traces(
        textinfo='percent+label',
        textfont_size=14
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def plot_regression_results(actual, predicted):
    """
    Create scatter and residual plots for regression results
    
    Args:
        actual: Actual values
        predicted: Predicted values
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create subplots with 2 rows
    fig = make_subplots(
        rows=2, 
        cols=1,
        subplot_titles=("Actual vs Predicted", "Residuals"),
        vertical_spacing=0.2
    )
    
    # Actual vs Predicted
    fig.add_trace(
        go.Scatter(
            x=actual,
            y=predicted,
            mode='markers',
            name='Predictions',
            marker=dict(
                color='#e6194b',
                size=8,
                opacity=0.7
            )
        ),
        row=1, col=1
    )
    
    # Add perfect prediction line
    min_val = min(min(actual), min(predicted))
    max_val = max(max(actual), max(predicted))
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
        ),
        row=1, col=1
    )
    
    # Residuals
    residuals = np.array(actual) - np.array(predicted)
    fig.add_trace(
        go.Scatter(
            x=predicted,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(
                color='#3cb44b',
                size=8,
                opacity=0.7
            )
        ),
        row=2, col=1
    )
    
    # Add zero line
    fig.add_trace(
        go.Scatter(
            x=[min(predicted), max(predicted)],
            y=[0, 0],
            mode='lines',
            name='Zero Line',
            line=dict(
                color='white',
                dash='dash'
            )
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        showlegend=True,
        title_text="Regression Model Results",
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    fig.update_xaxes(title_text="Actual Values", row=1, col=1)
    fig.update_yaxes(title_text="Predicted Values", row=1, col=1)
    fig.update_xaxes(title_text="Predicted Values", row=2, col=1)
    fig.update_yaxes(title_text="Residuals", row=2, col=1)
    
    return fig

def plot_classification_results(actual, predicted, probabilities=None, classes=None):
    """
    Create confusion matrix and ROC curve for classification results
    
    Args:
        actual: Actual values
        predicted: Predicted values
        probabilities: Prediction probabilities
        classes: Class labels
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create confusion matrix
    cm = confusion_matrix(actual, predicted)
    
    # Create subplots with 1 or 2 rows (depending on if we have probabilities)
    if probabilities is not None and len(np.unique(actual)) == 2:
        fig = make_subplots(
            rows=1, 
            cols=2,
            subplot_titles=("Confusion Matrix", "ROC Curve"),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}]]
        )
    else:
        fig = make_subplots(
            rows=1, 
            cols=1,
            subplot_titles=("Confusion Matrix",),
            specs=[[{"type": "heatmap"}]]
        )
    
    # Confusion Matrix
    if classes is None:
        classes = [str(i) for i in range(len(cm))]
    
    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=classes,
            y=classes,
            colorscale="Plasma",
            showscale=True,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 14}
        ),
        row=1, col=1
    )
    
    # ROC Curve for binary classification
    if probabilities is not None and len(np.unique(actual)) == 2:
        # Get probabilities for the positive class
        if len(probabilities[0]) >= 2:  # If we have probabilities for multiple classes
            probs = np.array(probabilities)[:, 1]
        else:  # If we only have a single probability
            probs = np.array(probabilities)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(actual, probs)
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {roc_auc:.3f})',
                line=dict(
                    color='#e6194b',
                    width=2
                )
            ),
            row=1, col=2
        )
        
        # Add diagonal line (random classifier)
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(
                    color='white',
                    dash='dash'
                )
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=500,
        title_text="Classification Results",
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    fig.update_xaxes(title_text="Predicted", row=1, col=1)
    fig.update_yaxes(title_text="Actual", row=1, col=1)
    
    if probabilities is not None and len(np.unique(actual)) == 2:
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)
    
    return fig

def plot_clusters(data, clusters, feature1, feature2, cluster_centers=None):
    """
    Create a scatter plot of clusters
    
    Args:
        data: DataFrame with features
        clusters: Cluster assignments
        feature1: First feature for x-axis
        feature2: Second feature for y-axis
        cluster_centers: Cluster centers (optional)
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create a scatter plot
    fig = px.scatter(
        data,
        x=feature1,
        y=feature2,
        color=clusters,
        color_discrete_sequence=px.colors.qualitative.Plotly,
        title=f'Clusters by {feature1} and {feature2}',
        template="plotly_dark"
    )
    
    # Add cluster centers if provided
    if cluster_centers is not None:
        for i, center in enumerate(cluster_centers):
            fig.add_trace(
                go.Scatter(
                    x=[center[feature1]],
                    y=[center[feature2]],
                    mode='markers',
                    marker=dict(
                        color='white',
                        size=15,
                        symbol='x'
                    ),
                    name=f'Cluster {i} Center'
                )
            )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def plot_pca_clusters(X, clusters, n_components=2):
    """
    Create a PCA visualization of clusters
    
    Args:
        X: Feature matrix
        clusters: Cluster assignments
        n_components: Number of PCA components
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Apply PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X)
    
    # Create a DataFrame with PCA components
    df_pca = pd.DataFrame(data=components, columns=[f'PC{i+1}' for i in range(n_components)])
    df_pca['Cluster'] = clusters
    
    # Create a scatter plot
    if n_components == 2:
        fig = px.scatter(
            df_pca,
            x='PC1',
            y='PC2',
            color='Cluster',
            color_discrete_sequence=px.colors.qualitative.Plotly,
            title='PCA Cluster Visualization',
            template="plotly_dark"
        )
    else:  # 3D plot for 3 components
        fig = px.scatter_3d(
            df_pca,
            x='PC1',
            y='PC2',
            z='PC3',
            color='Cluster',
            color_discrete_sequence=px.colors.qualitative.Plotly,
            title='3D PCA Cluster Visualization',
            template="plotly_dark"
        )
    
    # Show variance explained
    variance_explained = pca.explained_variance_ratio_
    total_variance = sum(variance_explained)
    variance_info = [f"PC{i+1}: {var:.1%}" for i, var in enumerate(variance_explained)]
    
    fig.update_layout(
        annotations=[
            dict(
                text=f"Variance Explained:<br>{'<br>'.join(variance_info)}<br>Total: {total_variance:.1%}",
                align="left",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=1.02,
                y=0.8,
                bordercolor="white",
                borderwidth=1,
                bgcolor="rgba(0,0,0,0.5)",
                font=dict(color="white")
            )
        ],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def plot_time_series(df, date_column, value_columns, title="Time Series Data"):
    """
    Create a time series plot
    
    Args:
        df: DataFrame with time series data
        date_column: Column with dates
        value_columns: List of columns to plot
        title: Plot title
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    fig = go.Figure()
    
    for column in value_columns:
        fig.add_trace(
            go.Scatter(
                x=df[date_column],
                y=df[column],
                mode='lines',
                name=column
            )
        )
    
    fig.update_layout(
        title=title,
        xaxis_title=date_column,
        yaxis_title="Value",
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        hovermode="x unified"
    )
    
    return fig

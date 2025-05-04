import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, silhouette_score
import joblib
import time

def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model
    
    Args:
        X_train: Training features
        y_train: Training target
    
    Returns:
        tuple: Trained model and information dictionary
    """
    start_time = time.time()
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    end_time = time.time()
    
    # Feature importance for linear regression
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': np.abs(model.coef_)
    }).sort_values('Importance', ascending=False)
    
    info = {
        "model_type": "Linear Regression",
        "training_time": end_time - start_time,
        "coefficients": model.coef_.tolist(),
        "intercept": model.intercept_,
        "feature_importance": feature_importance
    }
    
    return model, info

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model
    
    Args:
        X_train: Training features
        y_train: Training target
    
    Returns:
        tuple: Trained model and information dictionary
    """
    start_time = time.time()
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    end_time = time.time()
    
    # Feature importance for logistic regression
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': np.abs(model.coef_[0])
    }).sort_values('Importance', ascending=False)
    
    info = {
        "model_type": "Logistic Regression",
        "training_time": end_time - start_time,
        "classes": model.classes_.tolist(),
        "feature_importance": feature_importance
    }
    
    return model, info

def train_kmeans(X_train, n_clusters=3):
    """
    Train a K-Means Clustering model
    
    Args:
        X_train: Training features
        n_clusters: Number of clusters
    
    Returns:
        tuple: Trained model and information dictionary
    """
    start_time = time.time()
    
    model = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = model.fit_predict(X_train)
    
    end_time = time.time()
    
    # Calculate silhouette score
    silhouette = silhouette_score(X_train, clusters) if len(np.unique(clusters)) > 1 else 0
    
    # Create a dataframe with cluster information
    cluster_info = pd.DataFrame({
        'Cluster': range(n_clusters),
        'Size': np.bincount(clusters)
    })
    
    # Calculate cluster centers in original feature space
    centers = pd.DataFrame(
        model.cluster_centers_,
        columns=X_train.columns
    )
    
    info = {
        "model_type": "K-Means Clustering",
        "training_time": end_time - start_time,
        "n_clusters": n_clusters,
        "silhouette_score": silhouette,
        "cluster_sizes": np.bincount(clusters).tolist(),
        "cluster_centers": centers
    }
    
    return model, info, clusters

def evaluate_linear_regression(model, X_test, y_test):
    """
    Evaluate a Linear Regression model
    
    Args:
        model: Trained model
        X_test: Testing features
        y_test: Testing target
    
    Returns:
        dict: Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    evaluation = {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "predictions": y_pred.tolist(),
        "actual": y_test.tolist()
    }
    
    return evaluation

def evaluate_logistic_regression(model, X_test, y_test):
    """
    Evaluate a Logistic Regression model
    
    Args:
        model: Trained model
        X_test: Testing features
        y_test: Testing target
    
    Returns:
        dict: Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    # Calculate precision, recall, and F1 score with appropriate averaging
    # based on the number of classes
    if len(np.unique(y_test)) > 2:
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
    else:
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
    
    evaluation = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "predictions": y_pred.tolist(),
        "probabilities": y_prob.tolist(),
        "actual": y_test.tolist(),
        "classes": model.classes_.tolist()
    }
    
    return evaluation

def evaluate_kmeans(model, X, original_data):
    """
    Evaluate a K-Means Clustering model
    
    Args:
        model: Trained model
        X: Features used for clustering
        original_data: Original dataframe
    
    Returns:
        tuple: Predictions and evaluation metrics
    """
    predictions = model.predict(X)
    
    # Calculate silhouette score
    silhouette = silhouette_score(X, predictions) if len(np.unique(predictions)) > 1 else 0
    
    # Add cluster labels to original data
    clustered_data = original_data.copy()
    clustered_data['Cluster'] = predictions
    
    # Calculate cluster statistics
    cluster_stats = clustered_data.groupby('Cluster').agg(['mean', 'std']).reset_index()
    
    evaluation = {
        "silhouette_score": silhouette,
        "cluster_counts": np.bincount(predictions).tolist(),
        "predictions": predictions.tolist(),
    }
    
    return clustered_data, evaluation

def save_model(model, filename="model.joblib"):
    """
    Save a trained model to a file
    
    Args:
        model: Trained model
        filename: Output filename
    
    Returns:
        bool: Success status
    """
    try:
        joblib.dump(model, filename)
        return True
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")
        return False

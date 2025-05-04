import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import streamlit as st
import datetime

def load_yahoo_finance_data(ticker, start_date, end_date):
    """
    Fetch data from Yahoo Finance using the yfinance library
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (datetime): Start date for data
        end_date (datetime): End date for data
    
    Returns:
        pandas.DataFrame: Dataframe with stock data
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            st.error(f"No data found for ticker {ticker} within the specified date range.")
            return None
        df.reset_index(inplace=True)  # Reset index to make Date a column
        
        # Convert Date column to string to avoid Arrow serialization issues
        df['Date'] = df['Date'].astype(str)
        
        # Add ticker column for better identification
        df['Ticker'] = ticker
        
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def process_data(df, target_column=None, drop_columns=None):
    """
    Clean and preprocess the dataframe
    
    Args:
        df (pandas.DataFrame): Input dataframe
        target_column (str, optional): Target column for prediction
        drop_columns (list, optional): List of columns to drop
    
    Returns:
        tuple: Processed dataframe and information dictionary
    """
    if df is None:
        return None, {"success": False, "message": "No data to process"}
    
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Store data info
    info = {
        "success": True,
        "original_shape": df.shape,
        "missing_values_before": df.isna().sum().sum(),
        "columns_before": list(df.columns),
    }
    
    # Handle missing values
    df_processed = df_processed.dropna()
    
    # Drop specified columns
    if drop_columns:
        df_processed = df_processed.drop(columns=[col for col in drop_columns if col in df_processed.columns])
    
    # Encode categorical columns if needed
    categorical_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
    if target_column in categorical_columns:
        categorical_columns.remove(target_column)
    
    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        encoders[col] = le
    
    # Update info
    info["missing_values_after"] = df_processed.isna().sum().sum()
    info["rows_removed"] = info["original_shape"][0] - df_processed.shape[0]
    info["columns_after"] = list(df_processed.columns)
    info["categorical_encoded"] = categorical_columns
    
    return df_processed, info

def engineer_features(df, target_column=None, feature_selection=None):
    """
    Engineer features from the dataframe
    
    Args:
        df (pandas.DataFrame): Input dataframe
        target_column (str, optional): Target column for prediction
        feature_selection (list, optional): List of features to select
    
    Returns:
        tuple: DataFrame with engineered features and information dictionary
    """
    if df is None:
        return None, {"success": False, "message": "No data for feature engineering"}
    
    # Make a copy to avoid modifying the original
    df_engineered = df.copy()
    
    info = {
        "success": True,
        "original_features": list(df.columns),
    }
    
    # If it's financial time-series data, create some common financial indicators
    if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        # Calculate daily return
        df_engineered['Daily_Return'] = df_engineered['Close'].pct_change()
        
        # Calculate moving averages
        df_engineered['MA_5'] = df_engineered['Close'].rolling(window=5).mean()
        df_engineered['MA_20'] = df_engineered['Close'].rolling(window=20).mean()
        
        # Calculate volatility (standard deviation over a window)
        df_engineered['Volatility_5'] = df_engineered['Close'].rolling(window=5).std()
        
        # Calculate price momentum
        df_engineered['Momentum_5'] = df_engineered['Close'] - df_engineered['Close'].shift(5)
        
        # Calculate RSI (Relative Strength Index) - simplified version
        delta = df_engineered['Close'].diff()
        gain = delta.mask(delta < 0, 0)
        loss = -delta.mask(delta > 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df_engineered['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Drop rows with NaN values created by lag features
    df_engineered = df_engineered.dropna()
    
    # If specific features are selected, keep only those
    if feature_selection and len(feature_selection) > 0:
        valid_features = [col for col in feature_selection if col in df_engineered.columns]
        if target_column and target_column not in valid_features and target_column in df_engineered.columns:
            valid_features.append(target_column)
        df_engineered = df_engineered[valid_features]
    
    # Update info
    info["new_features"] = [col for col in df_engineered.columns if col not in info["original_features"]]
    info["final_features"] = list(df_engineered.columns)
    info["rows_after_engineering"] = df_engineered.shape[0]
    
    return df_engineered, info

def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Split the dataframe into training and testing sets
    
    Args:
        df (pandas.DataFrame): Input dataframe
        target_column (str): Target column for prediction
        test_size (float, optional): Proportion of the dataset to include in the test split
        random_state (int, optional): Random seed for reproducibility
    
    Returns:
        tuple: X_train, X_test, y_train, y_test, and information dictionary
    """
    if df is None or target_column not in df.columns:
        return None, None, None, None, {"success": False, "message": "Invalid data or target column"}
    
    info = {
        "success": True,
        "test_size": test_size,
        "random_state": random_state,
    }
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Standard scaling for numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to dataframes to maintain column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Update info
    info["train_shape"] = X_train.shape
    info["test_shape"] = X_test.shape
    info["features"] = list(X.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, info

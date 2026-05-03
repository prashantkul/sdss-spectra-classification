#!/usr/bin/env python3
"""
Data loading utilities for SDSS dataset
"""

import pandas as pd
import numpy as np

def load_sdss_data(file_path):
    """
    Load SDSS dataset from CSV file
    
    Parameters:
    file_path (str): Path to the CSV file containing SDSS data
    
    Returns:
    pandas.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded dataset with shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def validate_sdss_columns(df):
    """
    Validate that the dataset contains required SDSS columns
    
    Parameters:
    df (pandas.DataFrame): SDSS dataset
    
    Returns:
    bool: True if all required columns are present
    """
    required_columns = [
        'obj_ID', 'alpha', 'delta', 'u', 'g', 'r', 'i', 'z',
        'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'spec_obj_ID',
        'class', 'redshift', 'plate', 'MJD', 'fiber_ID'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    else:
        print("All required columns are present")
        return True

def preprocess_sdss_data(df):
    """
    Preprocess SDSS data for machine learning
    
    Parameters:
    df (pandas.DataFrame): Raw SDSS dataset
    
    Returns:
    tuple: (features, target) ready for modeling
    """
    # Remove ID columns that don't contribute to classification
    feature_columns = [
        'alpha', 'delta', 'u', 'g', 'r', 'i', 'z',
        'run_ID', 'rerun_ID', 'cam_col', 'field_ID',
        'redshift', 'plate', 'MJD', 'fiber_ID'
    ]
    
    # Check if all feature columns exist
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    
    X = df[feature_columns]
    y = df['class']
    
    # Handle missing values if any
    X = X.fillna(X.mean())
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y

if __name__ == "__main__":
    print("SDSS Data Loading Utilities")
    print("=" * 30)
    print("This module provides utilities for loading and preprocessing SDSS data")
    print("Use load_sdss_data() to load your dataset")
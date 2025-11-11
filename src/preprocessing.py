#!/usr/bin/env python3
"""
Data preprocessing utilities for heart disease prediction.
"""

import pandas as pd


def load_data(path):
    """Load and validate dataset.
    
    Args:
        path (str): Path to CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(path)
    return df


def remove_duplicates(df):
    """Remove duplicate records.
    
    Args:
        df (pd.DataFrame): Input dataset.
        
    Returns:
        pd.DataFrame: Dataset with duplicates removed.
    """
    return df.drop_duplicates()


def check_missing(df):
    """Check for missing values.
    
    Args:
        df (pd.DataFrame): Input dataset.
        
    Returns:
        pd.Series: Missing value counts per column.
    """
    return df.isnull().sum()


def split_features_target(df, target_column="target"):
    """Split features and target variable.
    
    Args:
        df (pd.DataFrame): Input dataset.
        target_column (str): Name of target column.
        
    Returns:
        tuple: (X, y) feature matrix and target vector.
    """
    X = df.drop(columns=[target_column], axis=1)
    y = df[target_column]
    return X, y

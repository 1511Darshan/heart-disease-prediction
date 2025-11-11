#!/usr/bin/env python3
"""
Inference script for heart disease prediction model.

Example:
    python -m src.predict --model-path models/heart_model.joblib --input data.csv
"""

import argparse
import joblib
import pandas as pd


def load_model(path):
    """Load trained model from disk.
    
    Args:
        path (str): Path to model file.
        
    Returns:
        model: Loaded model.
    """
    return joblib.load(path)


def predict(model, X):
    """Make predictions on new data.
    
    Args:
        model: Trained model.
        X: Feature matrix.
        
    Returns:
        np.ndarray: Predicted classes.
    """
    return model.predict(X)


def predict_proba(model, X):
    """Get prediction probabilities.
    
    Args:
        model: Trained model.
        X: Feature matrix.
        
    Returns:
        np.ndarray: Prediction probabilities.
    """
    return model.predict_proba(X)


def main():
    parser = argparse.ArgumentParser(
        description="Make predictions using trained model"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to trained model file"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file"
    )
    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.model_path)
    
    print("Loading data...")
    X = pd.read_csv(args.input)
    
    print("Making predictions...")
    predictions = predict(model, X)
    
    print(f"Predictions: {predictions}")
    print("✓ Inference complete!")


if __name__ == "__main__":
    main()

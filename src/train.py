#!/usr/bin/env python3
"""
Training script for heart disease prediction model.

Example:
    python -m src.train --data-path data/raw/heart.csv --output-model models/heart_model.joblib
"""

import argparse
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import load_data, split_features_target, remove_duplicates

RANDOM_STATE = 2


def build_pipeline(model_type="random_forest"):
    """Build preprocessing and model pipeline.

    Args:
        model_type (str): Type of model ('random_forest', 'logistic_regression').

    Returns:
        Pipeline: Scikit-learn pipeline.
    """
    if model_type == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    else:  # random_forest
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=7, random_state=RANDOM_STATE, n_jobs=-1
        )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )
    return pipeline


def train(X, y, model_type="random_forest", cv=5):
    """Train model with cross-validation.

    Args:
        X: Feature matrix.
        y: Target vector.
        model_type (str): Type of model to train.
        cv (int): Number of cross-validation folds.

    Returns:
        Pipeline: Trained pipeline.
    """
    pipeline = build_pipeline(model_type)

    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    print(f"CV Accuracy: mean={scores.mean():.4f}, std={scores.std():.4f}")

    pipeline.fit(X, y)
    return pipeline


def save_model(model, path):
    """Save trained model to disk.

    Args:
        model: Trained model.
        path (str): Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✓ Model saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train heart disease prediction model")
    parser.add_argument("--data-path", required=True, help="Path to dataset CSV file")
    parser.add_argument(
        "--output-model",
        default="models/heart_model.joblib",
        help="Output path for trained model",
    )
    parser.add_argument(
        "--model-type",
        default="random_forest",
        choices=["random_forest", "logistic_regression"],
        help="Type of model to train",
    )
    parser.add_argument(
        "--target-column", default="target", help="Name of target column"
    )
    args = parser.parse_args()

    print("Loading data...")
    df = load_data(args.data_path)
    df = remove_duplicates(df)
    print(f"Dataset shape: {df.shape}")

    print("Preprocessing...")
    X, y = split_features_target(df, target_column=args.target_column)

    print(f"Training {args.model_type} model...")
    model = train(X, y, model_type=args.model_type)

    save_model(model, args.output_model)
    print("✓ Training complete!")


if __name__ == "__main__":
    main()

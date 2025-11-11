"""Unit tests for preprocessing module."""

import pytest
import pandas as pd
from src.preprocessing import (
    remove_duplicates,
    check_missing,
    split_features_target,
)


@pytest.fixture
def sample_data():
    """Create sample dataframe for testing."""
    return pd.DataFrame({
        "age": [25, 35, 45, 55, 25],
        "cholesterol": [200, 220, 210, 230, 200],
        "target": [0, 1, 0, 1, 0],
    })


def test_remove_duplicates(sample_data):
    """Test duplicate removal."""
    result = remove_duplicates(sample_data)
    assert result.shape[0] == 4  # One duplicate removed


def test_check_missing(sample_data):
    """Test missing value check."""
    missing = check_missing(sample_data)
    assert missing.sum() == 0  # No missing values


def test_split_features_target(sample_data):
    """Test features and target split."""
    X, y = split_features_target(sample_data, target_column="target")
    assert X.shape == (5, 2)
    assert y.shape == (5,)
    assert "target" not in X.columns

# Heart Disease Prediction

A machine learning project for predicting heart disease from clinical features using ensemble methods and logistic regression.

## Overview

This repository contains:
- Exploratory data analysis (EDA) and baseline models in Jupyter Notebook
- Training and inference scripts for reproducible ML workflows
- CI/CD pipeline and containerized deployment
- Comprehensive model evaluation with cross-validation

## Quick Start

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/1511Darshan/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place dataset:**
   - Copy `heart.csv` to the `data/raw/` directory or update the path in `src/train.py`

5. **Run training:**
   ```bash
   python -m src.train --data-path data/raw/heart.csv --output-model models/heart_model.joblib
   ```

## Dataset

- **Source:** UCI Machine Learning Repository (Heart Disease)
- **Target:** Binary classification (presence/absence of heart disease)
- **Features:** 13 clinical measurements (age, sex, cholesterol, blood pressure, etc.)
- **Samples:** 303 records

## Models

Three classifiers are evaluated:

| Model | Train Accuracy | Test Accuracy | Notes |
|-------|---|---|---|
| Logistic Regression | 85.9% | 80.5% | Baseline, good generalization |
| Decision Tree (regularized) | 91.7% | 81.9% | max_depth=5, min_samples_split=10 |
| Random Forest | 99.6% | 85.2% | Best test accuracy, 100 estimators |

## Project Structure

```
heart-disease-prediction/
├── data/
│   ├── raw/              # Original dataset
│   └── processed/        # Processed datasets
├── models/               # Trained model artifacts
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── train.py          # Training script
│   ├── predict.py        # Inference script
│   └── preprocessing.py  # Data processing utilities
├── tests/
│   └── test_preprocessing.py
├── .github/workflows/
│   └── ci.yml            # CI/CD pipeline
├── requirements.txt
├── .gitignore
├── Dockerfile
└── README.md
```

## Running Tests

```bash
pytest -v
```

## Docker

Build and run with Docker:

```bash
docker build -t heart-disease-prediction .
docker run heart-disease-prediction
```

## Next Steps

- [ ] Add k-fold cross-validation for robust metrics
- [ ] Implement hyperparameter tuning (GridSearchCV)
- [ ] Add SHAP explainability analysis
- [ ] Build inference API (FastAPI)
- [ ] Add unit and integration tests
- [ ] Document feature importance

## License

MIT License – see LICENSE file for details.

## Contact

**Author:** [@1511Darshan](https://github.com/1511Darshan)  
**Issues:** [GitHub Issues](https://github.com/1511Darshan/heart-disease-prediction/issues)
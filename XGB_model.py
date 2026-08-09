import sklearn
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

import pandas as pd
import numpy as np

""" XGB model for expected finishing position"""

# Best params: {'model__subsample': np.float64(0.625), 
# 'model__reg_lambda': np.float64(3.5), 
# 'model__n_estimators': np.int64(450), 
# 'model__max_depth': np.int64(6), 
# 'model__learning_rate': np.float64(0.01), 
# 'model__colsample_bytree': np.float64(1.0)}

def xgboost_model(random_state = 42):
    """finetuned model hyperparameters """
    model = XGBRegressor(n_estimators = 450, 
                         learning_rate = 0.01,
                         reg_lambda = 3.5,
                         max_depth = 6,
                         colsample_bytree = 1,
                         subsample = 0.625,
                         random_state = random_state
                    )
    return model

def build_mu_pipeline(feature_df: pd.DataFrame) -> Pipeline:
    """Create preprocessing + XGBoost model pipeline."""
    model = xgboost_model()

    # separate categorical and numeric data based on dtypes
    categorical_cols = feature_df.select_dtypes(include = ["object", "category", "bool"]).columns

    # hardcode the compound columns to fill nan values with 0 instead
    compound_cols = [c for c in feature_df.columns if c.startswith('compound_frac')]
    feature_df.loc[:, compound_cols] = feature_df[compound_cols].fillna(0)

    numeric_cols = [ col for col in feature_df.columns if col not in categorical_cols]
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()), # can be excluded
        ]
    )
    
    # use one-hot encoder for categorical data
    categorical_transformer = OneHotEncoder(handle_unknown = 'ignore')

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_cols), 
        ("cat", categorical_transformer, categorical_cols)
    ])
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

def train_mu_model(feature_df: pd.DataFrame, target_col: str, n_splits=5):
    """Feature_df rows are already in chronological order."""

    if feature_df[target_col].isna().any():
        feature_df = feature_df.dropna(subset=[target_col])

    X = feature_df.drop(columns=[target_col])
    y = feature_df[target_col]

    # first .any() collapses each column down to a single boolean
    # second .any() collapses the series down to a single boolean

    if np.isinf(X.select_dtypes(include="number")).any().any():
        raise ValueError("Non-finite values (inf/-inf) found in features")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # do time-series cross validation here 
    mu_scores = []
    residuals = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        cv_pipeline = build_mu_pipeline(X_train)
        cv_pipeline.fit(X_train, y_train)
        y_prediction = cv_pipeline.predict(X_test)
        mu_scores.append(mean_absolute_error(y_test, y_prediction))
        residuals.extend(y_test.to_numpy() - y_prediction)

    mu_pipeline = build_mu_pipeline(X)
    mu_pipeline.fit(X, y)
    return mu_pipeline, mu_scores, residuals


def predict_mu(feature_df: pd.DataFrame, model: Pipeline):
    return model.predict(feature_df)


import os
import joblib

def save_mu_model(model: Pipeline, filepath: str = 'models/mu_model.joblib'):
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_mu_model(filepath: str = 'models/mu_model.joblib') -> Pipeline:
    return joblib.load(filepath)



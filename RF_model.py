import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np


"""RF model for volatility (sigma)"""


def random_forest_model(random_state = 42):
    model = RandomForestRegressor(n_estimators=300,
                                  max_depth=None, 
                                  min_samples_split=5,
                                  min_samples_leaf=3,
                                  max_features='sqrt',
                                  random_state= random_state,
                                  n_jobs=-1,
)
    return model

def build_sigma_pipeline(feature_df: pd.DataFrame) -> Pipeline:
    """Create preprocessing + RF model pipeline."""
    model = random_forest_model()

    # separate categorical and numeric data based on dtypes
    categorical_cols = feature_df.select_dtypes(include = ["object", "category", "bool"]).columns

    # hardcode the compound columns to fill nan values with 0 instead
    compound_cols = [col for col in feature_df.columns if col.startswith('compound_frac')]
    feature_df.loc[:,compound_cols] = feature_df[compound_cols].fillna(0)

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




def train_sigma_model(feature_df: pd.DataFrame, target_col: str, n_splits=5, min_rows : int = 200):
    """Assumes feature_df rows are already in chronological order."""

    if feature_df[target_col].isna().any():
        feature_df = feature_df.dropna(subset=[target_col])

    X = feature_df.drop(columns=[target_col])
    y = feature_df[target_col]

    if len(feature_df) < min_rows:
        return None, None
    
    # first .any() collapses each column down to a single boolean
    # second .any() collapses the series down to a single boolean
    if np.isinf(X.select_dtypes(include="number")).any().any():
        raise ValueError("Non-finite values (inf/-inf) found in features")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # do time-series cross validation here 

    sigma_scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        cv_pipeline = build_sigma_pipeline(X_train)
        cv_pipeline.fit(X_train, y_train)
        y_prediction = cv_pipeline.predict(X_test)
        sigma_scores.append(mean_absolute_error(y_test, y_prediction))

    sigma_pipeline = build_sigma_pipeline(X)
    sigma_pipeline.fit(X, y)
    return sigma_pipeline, sigma_scores

def predict_sigma(feature_df, sigma_model, residual_fallback):
    if sigma_model is not None:
        preds = sigma_model.predict(feature_df)
        return np.clip(preds, 0.1, None)
    else:
        return np.full(len(feature_df), residual_fallback)

import os
import joblib

def save_sigma_model(model: Pipeline, filepath: str = 'models/sigma_model.joblib'):
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_sigma_model(filepath: str = 'models/sigma_model.joblib') -> Pipeline:
    return joblib.load(filepath)

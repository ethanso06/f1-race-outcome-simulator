from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from feature_engineering import load_features
from XGB_model import build_mu_pipeline
import numpy as np 

param_distributions = {
    'model__n_estimators': list(np.linspace(100, 800, 15, dtype=int)),
    'model__max_depth': list(np.linspace(2, 8, 6, dtype=int)),
    'model__learning_rate': list(np.linspace(0.01, 0.2, 10)),
    'model__subsample': list(np.linspace(0.5, 1.0, 5)),
    'model__colsample_bytree': list(np.linspace(0.5, 1.0, 5)),
    'model__reg_lambda': list(np.linspace(0.5, 3.5, 7)),
}
def finetune_model(feature_df, target_col, n_iter=200, n_splits=5):
    if feature_df[target_col].isna().any():
        feature_df = feature_df.dropna(subset = [target_col])

    # features X = all columns except target_col
    # output y = target_col
    X = feature_df.drop(columns=[target_col]) 
    y = feature_df[target_col]

    pipeline = build_mu_pipeline(X)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1, # run all CPU cores at the same time (not sequentially)
    )
    search.fit(X, y)

    print("Best params:", search.best_params_)
    print("Best MAE:", -search.best_score_)
    return search.best_estimator_


if __name__ == "__main__":
    df = load_features([2022,2023,2024,2025,2026])
    best_model = finetune_model(df.drop(columns=['race_points']), target_col='finish_position')



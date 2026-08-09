from feature_engineering import load_features
from XGB_model import train_mu_model, save_mu_model
from RF_model import train_sigma_model, save_sigma_model
import numpy as np

df = load_features([2022, 2023, 2024, 2025, 2026])

mu_pipeline, mu_scores, mu_residuals = train_mu_model(
    df.drop(columns=['race_points']), target_col='finish_position'
)
save_mu_model(mu_pipeline)
np.save('models/mu_residuals.npy', np.array(mu_residuals))
print(f"mu CV MAE: {np.mean(mu_scores):.3f}")

sigma_pipeline, sigma_scores = train_sigma_model(
    df.drop(columns=['race_points', 'finish_position']), target_col='volatility'
)
save_sigma_model(sigma_pipeline)
print(f"sigma CV MAE: {np.mean(sigma_scores):.3f}")
import pandas as pd
from feature_engineering import load_features
from XGB_model import train_mu_model, predict_mu, save_mu_model, load_mu_model
from RF_model import train_sigma_model, predict_sigma, save_sigma_model, load_sigma_model
from monte_carlo import simulate_race_outcome, result_df
import numpy as np

df = load_features([2022, 2023, 2024, 2025, 2026])


def train_for_backtest(feature_df, year):
    """train model for backtest"""

    backtest_df = feature_df[feature_df["year"] < year].copy()

    mu_pipeline, mu_scores, mu_residuals = train_mu_model(backtest_df.drop(columns = "race_points"), target_col= "finish_position", )
    sigma_pipeline, _ = train_sigma_model(backtest_df.drop(columns = ["race_points", "finish_position"]), target_col= "volatility")
    print(f"CV MAE (mu, training folds): {mu_scores}")
    print(f"CV MAE (mu, mean): {np.mean(mu_scores):.3f}")
    return mu_pipeline, sigma_pipeline, mu_residuals, mu_scores


def backtest_race(feature_df, mu_pipeline, sigma_pipeline, residuals, race_year, race_round):
    "Predict + simulate ONE race"

    # fetch data of one race
    one_race = (feature_df["year"] == race_year) & (feature_df["round_num"] == race_round)
    one_race_df = feature_df[one_race].copy()
    driver_names = one_race_df['driver'].tolist()
    actual_positions = one_race_df['finish_position'].to_numpy()

    # prevent leakage
    bt_race_df = one_race_df.drop(columns=['finish_position', 'race_points'], errors='ignore')
    mu = predict_mu(bt_race_df, mu_pipeline)
    sigma = predict_sigma(bt_race_df, sigma_pipeline, residual_fallback = np.std(residuals))
  
    # run monte carlo sims 
    pred_positions = simulate_race_outcome(mu, sigma, residuals)

    # compare prediction with actual result
    pred_results = result_df(finishing_positions = pred_positions, driver_names = driver_names)
    pred_results['actual_position'] = actual_positions

    return pred_results


def backtest_szn(feature_df, race_year, rounds=None):
    mu_pipeline, sigma_pipeline, residuals, mu_scores = train_for_backtest(feature_df, race_year)

    # bt whole season
    if rounds is None:
        rounds = feature_df[feature_df['year'] == race_year]['round_num'].unique()

    all_results = []
    for round_num in rounds:
        results = backtest_race(
            feature_df=feature_df,
            mu_pipeline=mu_pipeline,
            sigma_pipeline=sigma_pipeline,
            residuals=residuals,
            race_year=race_year,
            race_round=round_num,
        )
        all_results.append(results)

    return pd.concat(all_results, ignore_index=True), mu_scores

if __name__ == "__main__":
    # use 22-25 to predict 26
    season_backtest, mu_scores = backtest_szn(df, race_year=2026)

    season_backtest['abs_error'] = (season_backtest['expected position'] - season_backtest['actual_position']).abs()
    backtest_mae = season_backtest['abs_error'].mean()

    print(f"Training CV MAE (mean across 5 folds): {np.mean(mu_scores):.3f}")
    print(f"Backtest MAE (on 2026, truly unseen): {backtest_mae:.3f}")

    # use 22-24 to predict 25
    season_backtest, mu_scores = backtest_szn(df, race_year=2025)

    season_backtest['abs_error'] = (season_backtest['expected position'] - season_backtest['actual_position']).abs()
    backtest_mae = season_backtest['abs_error'].mean()

    print(f"Training CV MAE (mean across 5 folds): {np.mean(mu_scores):.3f}")
    print(f"Backtest MAE (on 2025, truly unseen): {backtest_mae:.3f}")


    # use 22-23 to predict 24
    season_backtest, mu_scores = backtest_szn(df, race_year=2024)

    season_backtest['abs_error'] = (season_backtest['expected position'] - season_backtest['actual_position']).abs()
    backtest_mae = season_backtest['abs_error'].mean()

    print(f"Training CV MAE (mean across 5 folds): {np.mean(mu_scores):.3f}")
    print(f"Backtest MAE (on 2024, truly unseen): {backtest_mae:.3f}")
import numpy as np
import pandas as pd

def simulate_race_outcome(mu, sigma, residuals, n_sims: int = 10000, seed = 42):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n_drivers = len(mu)

    # each driver has its own mu and sigma 
    # therefore mu and sigma should always be equal to n_drivers

    if len(sigma) != n_drivers or len(mu) != n_drivers:
        raise ValueError("mu, sigma, and driver_names must all be the same length")
    if np.any(sigma <= 0):
        raise ValueError("sigma must be strictly positive for all drivers")

    rng = np.random.default_rng(seed)

    # account for skewness in residual distribution 
    residuals = np.asarray(residuals, dtype = float)
    standardized_z = (residuals - residuals.mean()) / residuals.std()
    z = rng.choice(standardized_z, size = (n_sims, n_drivers), replace = True)

    # monte carlo 
    samples = mu + sigma * z
    pred_positions = np.argsort(np.argsort(samples, axis=1), axis=1) + 1
 
    return pred_positions #there will be n_sims number of different finishing positions




def result_df(finishing_positions, driver_names : list):
    """
    - expected finishing position
    - win probability
    - podium probability
    - top 10 probability (finishing within the points)"""

    win_prob = (finishing_positions == 1).mean(axis = 0)
    podium_prob = (finishing_positions <= 3).mean(axis = 0)
    points_prob = (finishing_positions <= 10).mean(axis = 0)
    expected_position = finishing_positions.mean(axis=0)

    monte_carlo_results = pd.DataFrame({
        "Drivers" : driver_names,
        "win probability" : win_prob,
        "podium probability" : podium_prob,
        "points probability" : points_prob,
        "expected position" : expected_position
    })
    return monte_carlo_results


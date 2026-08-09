import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from feature_engineering import load_features
from XGB_model import train_mu_model

df = load_features([2022, 2023, 2024, 2025, 2026])
mu_pipeline, mu_scores, residuals = train_mu_model(
    df.drop(columns=['race_points']), target_col='finish_position'
)
residuals = np.array(residuals)

print(f"Skewness: {stats.skew(residuals):.3f}")
print(f"Excess kurtosis: {stats.kurtosis(residuals):.3f}")
print(f"Shapiro p-value: {stats.shapiro(residuals)[1]:.4f}")

fig, ax = plt.subplots(figsize=(6, 5))
ax.hist(residuals, bins=40, density=True, alpha=0.6)
mu, std = residuals.mean(), residuals.std()
x = np.linspace(residuals.min(), residuals.max(), 100)
ax.plot(x, stats.norm.pdf(x, mu, std), 'r-')
ax.set_title('Residual distribution vs Normal')

plt.tight_layout()
plt.savefig('residual_check.png')
plt.show()
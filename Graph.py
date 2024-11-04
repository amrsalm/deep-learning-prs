import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, gaussian_kde

# Define the x-axis range based on min and max of both distributions
x_range = np.linspace(min(case_prs.min(), control_prs.min()), max(case_prs.max(), control_prs.max()), 1000)

# Generate KDEs for case and control PRS
case_kde = gaussian_kde(case_prs, bw_method=0.2)
control_kde = gaussian_kde(control_prs, bw_method=0.2)

# Create plot
plt.figure(figsize=(12, 7))

# Plot PRS density for Case group
plt.fill_between(x_range, case_kde(x_range), color='blue', alpha=0.2)
plt.plot(x_range, case_kde(x_range), label='Case PRS Density', color='blue', linewidth=2)

# Plot PRS density for Control group
plt.fill_between(x_range, control_kde(x_range), color='green', alpha=0.2)
plt.plot(x_range, control_kde(x_range), label='Control PRS Density', color='green', linewidth=2)

# Fit Gaussian Mixture Model (GMM) for the Case PRS to show bimodality
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(case_prs.reshape(-1, 1))
means = gmm.means_.flatten()
weights = gmm.weights_.flatten()
stds = np.sqrt(gmm.covariances_.flatten())

# Plot GMM fit and components for Case PRS
pdf = weights[0] * norm.pdf(x_range, means[0], stds[0]) + weights[1] * norm.pdf(x_range, means[1], stds[1])
plt.plot(x_range, pdf, color='red', linestyle='--', linewidth=2, label='Case Bimodal Fit (GMM)')

# Plot GMM components with corrected color and linestyle syntax
plt.plot(x_range, weights[0] * norm.pdf(x_range, means[0], stds[0]), linestyle='--', color='green', linewidth=1, label=f'Case Component 1 (μ={means[0]:.2f})')
plt.plot(x_range, weights[1] * norm.pdf(x_range, means[1], stds[1]), linestyle='--', color='purple', linewidth=1, label=f'Case Component 2 (μ={means[1]:.2f})')

# Add labels, title, and legend
plt.xlabel("Polygenic Risk Score (PRS)")
plt.ylabel("Density")
plt.title("PRS Distribution Comparison between Case and Control Populations with Bimodal Fit for Case")
plt.legend(loc="upper right", fontsize='small', frameon=False)

# Save and display the plot
plt.savefig('prs_case_control_bimodal.png')
plt.show()

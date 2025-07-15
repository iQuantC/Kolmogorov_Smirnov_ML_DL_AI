# Kolmogorov-Smirnov (KS) Statistical Test in ML, DL & AI
In this project, we illustrate how the Kolmogorov-Smirnov (KS) statistical test works, and why it is commonly used in Machine Learning (ML), Deep Learning (DL) and Artificial Intelligence (AI).

KS Test is a statistical test that compares two sets of data to see if they come from the same distribution. It helps to detect feature-level drift (per feature check).


## How KS Test Works
1. Take two (2) samples of data 
2. Create Cummulative Distribution Functions (CDFs) for both samples
3. Find the largest distance between the two CDF curves (i.e., the KS Statistic or D-Value)
4. Interprete the results.  
    1. If the KS Statistic is SMALL, (p-value > alpha). Then, the two distributions are "SIMILAR" which implies "NO SIGNIFICANT DRIFT"
    2. If the KS Statistic is LARGE, (p-value < alpha). Then, the two distributions are "DIFFERENT" which implies "POSSIBLE DATA DRIFT" or there is "STATISTICALLY SIGNIFICANT DRIFT"


## Summary
The KS test compares how similar the old and new data distributions are. A big difference means data drift, which can hurt model performance (model drift). 


## KS Test Illustration
Let's simulate KS Test for two datasets

```sh
touch ksmirnov.py requirements.txt
```

```sh
# requirements.txt

numpy
matplotlib
scipy
```

```sh
# ksmirnov.py


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.stats import norm

# Step 1: Simulate Data but make sure that the random numbers generated are always the same every time we run the code.
np.random.seed(42)


# Sample A: Training data (1000 normally distributed dataset centered around mean=50, std=5)
sample_A = np.random.normal(loc=50, scale=5, size=1000)

# Sample B: Live data (can simulate drift by changing mean or std - centered around mean=60, std=5)
sample_B = np.random.normal(loc=60, scale=5, size=1000)  # Drifted data

# Step 2: Compute means and standard deviations
mean_A = np.mean(sample_A)
std_A = np.std(sample_A)

mean_B = np.mean(sample_B)
std_B = np.std(sample_B)


# Step 3: Create x-axis range that covers both distributions
x = np.linspace(min(min(sample_A), min(sample_B)) - 5, 
                max(max(sample_A), max(sample_B)) + 5, 1000)

# Step 4: Calculate theoretical PDFs
pdf_A = norm.pdf(x, loc=mean_A, scale=std_A)
pdf_B = norm.pdf(x, loc=mean_B, scale=std_B)


# Step 5: Plot histograms
plt.figure(figsize=(12, 6))
# plt.hist(sample_A, bins=30, density=True, alpha=0.5, color='blue', edgecolor='black', label="Sample A (Training)")
# plt.hist(sample_B, bins=30, density=True, alpha=0.5, color='orange', edgecolor='black', label="Sample B (New)")


# Step 6: Overlay bell curves
plt.plot(x, pdf_A, 'b-', lw=2, label=f"Theoretical A (mean={mean_A:.2f})")
plt.plot(x, pdf_B, 'orange', lw=2, label=f"Theoretical B (mean={mean_B:.2f})")


# Step 7: Add vertical lines for means
plt.axvline(mean_A, color='blue', linestyle='--', linewidth=2, label="Mean A")
plt.axvline(mean_B, color='orange', linestyle='--', linewidth=2, label="Mean B")


# Step 8: Annotate data drift
plt.annotate("Data Drift",
             xy=((mean_A + mean_B) / 2, max(pdf_A.max(), pdf_B.max()) * 0.8),
             xytext=(mean_A, max(pdf_A.max(), pdf_B.max()) * 0.95),
             arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
             fontsize=12, color='red')

# Step 9: Labels
plt.title("Comparison of Bell Curves: Sample A vs Sample B")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)


# Step 10: Save the plot
plt.savefig("bell_curve_comparison.png")
print("Plot saved as bell_curve_comparison.png")


# Step 11: Run KS Test
ks_stat, p_value = ks_2samp(sample_A, sample_B)

print("KS Statistic (D-value):", ks_stat)
print("P-value:", p_value)


# Step 12: Plot CDFs
def plot_cdf(data, label):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf, label=label)

plt.figure(figsize=(10, 6))
plot_cdf(sample_A, "Training Data (Sample A)")
plot_cdf(sample_B, "New Data (Sample B)")

plt.title("CDF Comparison - Kolmogorov-Smirnov Test")
plt.xlabel("Value")
plt.ylabel("Cumulative Probability")
plt.grid(True)
plt.legend()
plt.savefig("ks_cdf_plot.png")
print("Plot saved as ks_cdf_plot.png")
```

### Create Python Virtual Environment
```sh
python3 -m venv ksmirnov-venv
source ksmirnov-venv/bin/activate
```
 
### Install Dependencies
```sh
pip install -r requirements.txt
```

### Run the Script
```sh
python ksmirnov.py
```

### Clean Up
```sh
deactivate
rm -rf ksmirnov-venv
```

# NEXT UP
Data/Model Drift Real-time Project with Automated ML Model Retraining using Scikit-learn, Alibi Detect (using KSDrift), and Apache Airflow. 

# Like, Comment, and Subscribe to iQuant YouTube Channel

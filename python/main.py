import numpy as np
import pandas as pd
import time
import tracemalloc

# Paths
csv_path = "/home/hofst/dataScienceComparison/datasets/dataset.csv"
results_path = "/home/hofst/dataScienceComparison/results/pythonresults.txt"

df = pd.read_csv(csv_path, header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_intercept = np.hstack([np.ones((X.shape[0], 1)), X])

# Lists to store results
r2_list = []
runtime_list = []
memory_list = []

# Repeat test 10 times
for i in range(10):
    start_time = time.time()
    tracemalloc.start()

    beta_hat = np.linalg.inv(X_intercept.T @ X_intercept) @ X_intercept.T @ y
    y_pred = X_intercept @ beta_hat

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Store results
    r2_list.append(r_squared)
    runtime_list.append(end_time - start_time)
    memory_list.append(peak / 1024)  # in KB

# Compute averages
avg_r2 = np.mean(r2_list)
avg_runtime = np.mean(runtime_list)
avg_memory = np.mean(memory_list)

# Print to console
print(f"Average R²: {avg_r2:.4f}")
print(f"Average runtime: {avg_runtime:.6f} seconds")
print(f"Average memory peak: {avg_memory:.2f} KB")

# Write to results.txt
with open(results_path, "w") as f:
    f.write("Linear Regression Benchmark (Python, 10 runs)\n")
    f.write(f"Average R²: {avg_r2:.4f}\n")
    f.write(f"Average runtime: {avg_runtime:.6f} seconds\n")
    f.write(f"Average memory peak: {avg_memory:.2f} KB\n")


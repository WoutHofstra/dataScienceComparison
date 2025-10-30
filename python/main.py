import numpy as np
import pandas as pd
import time
import tracemalloc

csv_path = input("Enter the path to your CSV file: ")
df = pd.read_csv(csv_path)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_intercept = np.hstack([np.ones((X.shape[0], 1)), X])

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

print("Estimated coefficients:", beta_hat)
print(f"R² score: {r_squared:.4f}")
print(f"Runtime: {end_time - start_time:.6f} seconds")
print(f"Memory peak: {peak / 1024:.2f} KB")

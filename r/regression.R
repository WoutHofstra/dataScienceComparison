
# Load required libraries
if(!require(data.table)) install.packages("data.table", repos='http://cran.rstudio.com/')
if(!require(pryr)) install.packages("pryr", repos='http://cran.rstudio.com/')
library(data.table)
library(pryr)

# Paths
csv_path <- "/home/hofst/dataScienceComparison/datasets/dataset.csv"
results_path <- "/home/hofst/dataScienceComparison/results/Rresults.txt"

# Read CSV (no header)
data <- fread(csv_path, header = FALSE)

# Separate features and target
X <- as.matrix(data[, 1:(ncol(data)-1), with=FALSE])
y <- as.matrix(data[[ncol(data)]])

# Add intercept column
X_intercept <- cbind(1, X)

# Initialize lists to store results
r2_list <- numeric(10)
runtime_list <- numeric(10)
memory_list <- numeric(10)

# Repeat regression 10 times
for(i in 1:10){
  start_time <- Sys.time()
  mem_before <- mem_used()

  # Raw matrix linear regression (like Python)
  beta_hat <- solve(t(X_intercept) %*% X_intercept) %*% t(X_intercept) %*% y
  y_pred <- X_intercept %*% beta_hat
  
  mem_after <- mem_used()
  end_time <- Sys.time()

  # Calculate R²
  ss_res <- sum((y - y_pred)^2)
  ss_tot <- sum((y - mean(y))^2)
  r2_list[i] <- 1 - (ss_res / ss_tot)

  # Store runtime and memory
  runtime_list[i] <- as.numeric(end_time - start_time, units="secs")
  memory_list[i] <- as.numeric(mem_after - mem_before)/1024  # KB
}

# Compute averages
avg_r2 <- mean(r2_list)
avg_runtime <- mean(runtime_list)
avg_memory <- mean(memory_list)

# Print results
cat(sprintf("Average R²: %.4f\n", avg_r2))
cat(sprintf("Average runtime: %.6f seconds\n", avg_runtime))
cat(sprintf("Average memory peak: %.2f KB\n", avg_memory))

# Write results to file
output_lines <- c(
  "Linear Regression Benchmark (R, 10 runs, raw matrix)",
  sprintf("Average R²: %.4f", avg_r2),
  sprintf("Average runtime: %.6f seconds", avg_runtime),
  sprintf("Average memory peak: %.2f KB", avg_memory)
)
writeLines(output_lines, results_path)

using CSV
using DataFrames
using LinearAlgebra
using Statistics
using BenchmarkTools

csv_path = "/home/hofst/dataScienceComparison/datasets/dataset.csv"
results_path = "/home/hofst/dataScienceComparison/results/julia_results.txt"

df = CSV.read(csv_path, DataFrame, header=false)

X = Matrix(df[:, 1:end-1])
y = Matrix(df[:, end:end])

X_intercept = hcat(ones(size(X,1)), X)

r2_list = Float64[]
runtime_list = Float64[]
memory_list = Float64[]

for i in 1:10
    mem_before = Base.summarysize(X_intercept) + Base.summarysize(y)
    start_time = time()

    beta_hat = inv(X_intercept' * X_intercept) * X_intercept' * y
    y_pred = X_intercept * beta_hat

    end_time = time()
    mem_after = Base.summarysize(beta_hat) + Base.summarysize(y_pred)

    ss_res = sum((y .- y_pred).^2)
    ss_tot = sum((y .- mean(y)).^2)
    push!(r2_list, 1 - ss_res/ss_tot)

    push!(runtime_list, end_time - start_time)
    push!(memory_list, mem_after + mem_before)
end

avg_r2 = mean(r2_list)
avg_runtime = mean(runtime_list)
avg_memory = mean(memory_list)/1024  # KB

println("Average R²: ", round(avg_r2, digits=4))
println("Average runtime: ", round(avg_runtime, digits=6), " seconds")
println("Average memory peak: ", round(avg_memory, digits=2), " KB")

output_lines = [
    "Linear Regression Benchmark (Julia, 10 runs)",
    "Average R²: $(round(avg_r2, digits=4))",
    "Average runtime: $(round(avg_runtime, digits=6)) seconds",
    "Average memory peak: $(round(avg_memory, digits=2)) KB"
]

open(results_path, "w") do f
    for line in output_lines
        println(f, line)
    end
end

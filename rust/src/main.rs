use std::time::Instant;
use std::fs::File;
use std::io::Write;
use csv::ReaderBuilder;
use ndarray::{Array1, Array2};
use ndarray_linalg::LeastSquaresSvd;
use procinfo::pid; // new line for memory tracking

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let csv_path = "/home/hofst/dataScienceComparison/datasets/dataset.csv";
    let results_path = "/home/hofst/dataScienceComparison/results/rustresults.txt";

    // --- Read CSV ---
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(csv_path)?;
    let mut x_data: Vec<f64> = Vec::new();
    let mut y_data: Vec<f64> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        for i in 0..record.len()-1 {
            x_data.push(record[i].parse::<f64>()?);
        }
        y_data.push(record[record.len()-1].parse::<f64>()?);
    }

    let n_rows = y_data.len();
    let n_cols = x_data.len() / n_rows;

    // --- Add intercept column ---
    let mut x_matrix = Array2::<f64>::zeros((n_rows, n_cols + 1));
    for i in 0..n_rows {
        x_matrix[[i, 0]] = 1.0; // intercept
        for j in 0..n_cols {
            x_matrix[[i, j+1]] = x_data[i * n_cols + j];
        }
    }
    let y_vector = Array1::from_vec(y_data);

    // --- Run regression 10 times ---
    let mut total_r2 = 0.0;
    let mut total_time = 0.0;
    let mut peak_memory_kb = 0.0;
    let runs = 10;

    for _ in 0..runs {
        let start = Instant::now();
        let beta_hat = x_matrix.clone().least_squares(&y_vector)?.solution;
        let duration = start.elapsed();

        // Track memory usage (RSS in KB)
        if let Ok(stat) = pid::statm_self() {
            let rss_kb = stat.resident * 4; // page size ~4 KB
            if rss_kb as f64 > peak_memory_kb {
                peak_memory_kb = rss_kb as f64;
            }
        }

        let y_hat = x_matrix.dot(&beta_hat);
        let residuals = &y_vector - &y_hat;
        let ss_res = residuals.dot(&residuals);
        let y_mean = y_vector.mean().unwrap();
        let ss_tot = (&y_vector - y_mean).dot(&(&y_vector - y_mean));
        let r2 = 1.0 - ss_res / ss_tot;

        total_r2 += r2;
        total_time += duration.as_secs_f64();
    }

    let avg_r2 = total_r2 / runs as f64;
    let avg_time = total_time / runs as f64;

    // --- Print & save results ---
    let output = format!(
        "Linear Regression Benchmark (Rust, {} runs)\n\
         Average R²: {:.4}\n\
         Average runtime: {:.6} seconds\n\
         Average memory peak: {:.2} KB\n",
        runs, avg_r2, avg_time, peak_memory_kb
    );

    println!("{}", output);

    let mut file = File::create(results_path)?;
    write!(file, "{}", output)?;

    Ok(())
}

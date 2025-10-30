package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"runtime"
	"strconv"
	"time"

	"gonum.org/v1/gonum/mat"
)

func main() {
	csvPath := "/home/hofst/dataScienceComparison/datasets/dataset.csv"
	resultsPath := "/home/hofst/dataScienceComparison/results/golangresults.txt"

	// --- Read CSV ---
	file, err := os.Open(csvPath)
	if err != nil {
		log.Fatalf("cannot open CSV: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		log.Fatalf("cannot read CSV: %v", err)
	}

	nRows := len(records)
	nCols := len(records[0]) - 1

	xData := mat.NewDense(nRows, nCols+1, nil) // +1 for intercept
	yData := mat.NewVecDense(nRows, nil)

	for i, record := range records {
		xData.Set(i, 0, 1.0) // intercept
		for j := 0; j < nCols; j++ {
			val, err := strconv.ParseFloat(record[j], 64)
			if err != nil {
				log.Fatalf("cannot parse X[%d,%d]: %v", i, j, err)
			}
			xData.Set(i, j+1, val)
		}
		yVal, err := strconv.ParseFloat(record[nCols], 64)
		if err != nil {
			log.Fatalf("cannot parse Y[%d]: %v", i, err)
		}
		yData.SetVec(i, yVal)
	}

	runs := 10
	var totalR2 float64
	var totalTime float64
	var totalMem float64

	for run := 0; run < runs; run++ {
		// --- Memory tracking start ---
		var mStart runtime.MemStats
		runtime.ReadMemStats(&mStart)
		start := time.Now()

		// --- QR regression ---
		var qr mat.QR
		qr.Factorize(xData)

		var beta mat.Dense
		err := qr.SolveTo(&beta, false, yData)
		if err != nil {
			log.Fatalf("QR solve failed: %v", err)
		}

		duration := time.Since(start).Seconds()
		totalTime += duration

		// --- y_hat ---
		yHat := mat.NewVecDense(nRows, nil)
		yHatMat := mat.NewDense(nRows, 1, nil)
		yHatMat.Mul(xData, &beta)
		for i := 0; i < nRows; i++ {
			yHat.SetVec(i, yHatMat.At(i, 0))
		}

		// --- R² calculation ---
		residuals := mat.NewVecDense(nRows, nil)
		residuals.SubVec(yData, yHat)

		ssRes := mat.Dot(residuals, residuals)

		var yMean float64
		for i := 0; i < nRows; i++ {
			yMean += yData.AtVec(i)
		}
		yMean /= float64(nRows)

		ssTot := 0.0
		for i := 0; i < nRows; i++ {
			diff := yData.AtVec(i) - yMean
			ssTot += diff * diff
		}

		r2 := 1.0 - ssRes/ssTot
		totalR2 += r2

		// --- Memory tracking end ---
		var mEnd runtime.MemStats
		runtime.ReadMemStats(&mEnd)
		allocated := float64(mEnd.Alloc - mStart.Alloc)
		totalMem += allocated
	}

	avgR2 := totalR2 / float64(runs)
	avgTime := totalTime / float64(runs)
	avgMemKB := totalMem / float64(runs) / 1024.0

	output := fmt.Sprintf(
		"Linear Regression Benchmark (Go, %d runs)\n"+
			"Average R²: %.4f\n"+
			"Average runtime: %.6f seconds\n"+
			"Average memory peak: %.2f KB\n",
		runs, avgR2, avgTime, avgMemKB)

	fmt.Print(output)

	// --- Save results ---
	outFile, err := os.Create(resultsPath)
	if err != nil {
		log.Fatalf("cannot create results file: %v", err)
	}
	defer outFile.Close()

	_, err = outFile.WriteString(output)
	if err != nil {
		log.Fatalf("cannot write results: %v", err)
	}
}

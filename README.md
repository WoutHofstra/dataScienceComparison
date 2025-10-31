# dataScienceComparison
A personal project where I compare various languages' performance in data science, based on speed, readability, practicality

In this project, I have implemented a simple linear regression model in five different programming languages: Python, R, Julia, Rust and Go.  
The goal of this was to compare their runtimes, memory usage and I wanted to figure out which one would actually be the most suitable to perform data analytics/ data science with.  

## My hypothesis:  
I assume Python will be the slowest language out of these, as it is known for being really slow. I do think its slowness will be countered by the libraries it can use, as these can make Python seem faster most of the time. I think either Rust of Golang will be the fastest language.

## Steps:  
First I obtained a simple dataset which I store inside the datasets directory.  
Then I implemented linear regression in every language using various libraries: NumPy/ SciPy for python, lm() for R, LinearAlgebra for Julia, ndarray for Rust and gonum/ mat for Go.  
During this coding process, I made sure that every language got similar results in terms of accuracy, I did this based on their respective R^2 values.  
I stored all results in the languages' own 'results' files, to easily compare later.  

## Results
My overall results were as follows: 

### R^2 score:  
All languages got a R^2 score of 0.9268, proving their accuracy, and making my comparisons way more reliable.  

### Speed:  
The fastest language was Rust, with a score of only 0.000047 seconds on average, and this makes total sense: Rust is compiled to native machine code with zero-cost abstractions, meaning high-level features don’t slow it down. Its ownership system eliminates garbage collection, LLVM optimizations make numeric code very efficient, and libraries like ndarray-linalg use optimized native BLAS/LAPACK routines for fast matrix computations. Not far behind it though, is Golang with a score of 0.000063 seconds. Go is a compiled language with a simple runtime and efficient garbage collector. Its code is compiled to native machine instructions, and its concurrency model with goroutines allows lightweight parallelism. While not as low-level as Rust, Go achieves very good performance for numeric and I/O-heavy tasks. Python also was pretty fast, with a score of 0.000174. Python scores well in speed because libraries like NumPy and pandas delegate heavy computations to optimized C and Fortran code, so most of the work runs in compiled, efficient routines rather than in Python itself.

### Memory Usage:  
The language that scored the best in memory usage (using the least memory) is R. R used only a maximum of 6.1 Kb, which is extremely low. R scored well in memory usage because it stores data efficiently in column-major order and uses internal optimizations for handling vectors and matrices, minimizing memory overhead for typical statistical operations. Python also did well, peaking at 32.13 KB, because libraries like NumPy and pandas store large arrays in efficient C/Fortran memory layouts rather than Python objects. Golang came next at 84.80 KB, using a simple garbage-collected runtime with relatively low overhead. Surprisingly, Rust showed the highest memory peak at 5,632 KB, which is likely due to how its native arrays and BLAS/LAPACK routines allocate memory for matrix computations—even though Rust is extremely fast, it can use more memory for temporary allocations during numeric operations.

### Overall:  
In general, the best language to use depends heavily on the nature of your project and the priorities you have—speed, memory usage, or ease of development. According to my benchmark tests, Python and Go stand out as strong choices for linear regression tasks on medium-sized datasets. Python is particularly appealing if memory efficiency is your priority: thanks to libraries like NumPy and pandas, it handles large numeric arrays in optimized, low-overhead memory layouts. Go, on the other hand, is the choice when raw execution speed matters, with its compiled code and efficient runtime providing excellent performance for numeric computations. Rust, while incredibly fast, may consume more memory than expected for certain operations, so it’s ideal if speed outweighs memory concerns. Overall, Python and Go strike the best balance between performance and usability, with Python favoring memory efficiency and Go favoring speed.
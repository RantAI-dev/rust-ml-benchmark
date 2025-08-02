# Rust vs Python ML Benchmark Results

## Performance Comparison by Algorithm Category

This table organizes results by algorithm type, standardizing names across different libraries for fair comparison.

## Support Vector Machines

| Library | Model | Training Time (s) | Inference Time (s) | Accuracy | Memory Usage (MB) |
|---------|-------|------------------|-------------------|----------|-------------------|
| linfa | SVM (Binary) | 0.055504 | 0.002579 | 1.00000 | 0.25 |
| linfa | SVM (RBF Kernel) | 610.935271 | 8.600239 | 0.69164 | 1.97 |
| scikit-learn | SVM | 0.001437 | 0.000193 | 1.00000 | 0.25 |
| scikit-learn | SVM (Linear) | 16.487870 | 0.000670 | 0.90714 | 0.30 |
| smartcore | SVM (Multiclass) | 0.174034 | 0.001000 | 0.83333 | 0.25 |
| smartcore | SVM (RBF Kernel) | 14888.138760 | 25.037999 | 0.69167 | -1.27 |


## Tree-Based Models

| Library | Model | Training Time (s) | Inference Time (s) | Accuracy | Memory Usage (MB) |
|---------|-------|------------------|-------------------|----------|-------------------|
| linfa | Decision Tree | 0.502817 | 0.000139 | 0.72041 | 6.50 |
| linfa | Decision Tree | 0.002332 | 0.000026 | 0.68966 | 0.00 |
| scikit-learn | Decision Tree | 0.162300 | 0.001210 | 0.90653 | 0.00 |
| scikit-learn | Decision Tree | 0.001109 | 0.000264 | 1.00000 | 0.77 |
| scikit-learn | Random Forest | 0.564070 | 0.034870 | 0.91211 | 78.96 |
| scikit-learn | Random Forest | 0.118056 | 0.004227 | 1.00000 | 0.38 |
| smartcore | Decision Tree | 0.000513 | 0.000022 | 0.73333 | 0.00 |
| smartcore | Random Forest | 0.042207 | 0.001733 | 0.73333 | 0.25 |
| smartcore | Random Forest | 17.194597 | 0.132472 | 0.70029 | 51.52 |


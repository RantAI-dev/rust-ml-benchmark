# Comprehensive ML Benchmark Report

## System Information

- **Hostname**: shironeko
- **CPU**: AMD Ryzen 5 4600G with Radeon Graphics
- **CPU Cores**: 12
- **Total Memory**: 19.4 GB
- **Timestamp**: 2025-07-27T09:21:58.636442+00:00

## Dataset Information

- **Dataset**: iris
- **Total Samples**: 150
- **Features**: 4
- **Classes**: 3

## Performance Summary by Algorithm Category

### Support Vector Machines

| Library | Model | Training Time (s) | Inference Time (s) | Accuracy | Memory (MB) | Runs |
|---------|-------|------------------|-------------------|----------|-------------|------|
| scikit-learn | SVM | 0.0008±0.0000 | 0.0002±0.0000 | 1.0000±0.0000 | 151.3 | 2 |
| scikit-learn | SVM | 0.0012±0.0004 | 0.0002±0.0001 | 1.0000±0.0000 | 151.8 | 15 |
| scikit-learn | SVM | 0.0014±0.0007 | 0.0004±0.0002 | 1.0000±0.0000 | 151.7 | 5 |
| scikit-learn | SVM | 0.0010±0.0000 | 0.0003±0.0000 | 1.0000±0.0000 | 151.7 | 2 |

### Tree-Based Models

| Library | Model | Training Time (s) | Inference Time (s) | Accuracy | Memory (MB) | Runs |
|---------|-------|------------------|-------------------|----------|-------------|------|
| scikit-learn | Decision Tree | 0.0016±0.0006 | 0.0002±0.0001 | 1.0000±0.0000 | 150.6 | 2 |
| scikit-learn | Decision Tree | 0.0007±0.0001 | 0.0002±0.0001 | 1.0000±0.0000 | 150.7 | 15 |
| scikit-learn | Decision Tree | 0.0010±0.0002 | 0.0002±0.0001 | 1.0000±0.0000 | 150.8 | 5 |
| scikit-learn | Decision Tree | 0.0007±0.0001 | 0.0001±0.0000 | 1.0000±0.0000 | 150.9 | 2 |
| scikit-learn | Random Forest | 0.0975±0.0044 | 0.0037±0.0006 | 1.0000±0.0000 | 151.0 | 2 |
| scikit-learn | Random Forest | 0.0941±0.0039 | 0.0034±0.0004 | 1.0000±0.0000 | 151.5 | 15 |
| scikit-learn | Random Forest | 0.1009±0.0022 | 0.0035±0.0005 | 1.0000±0.0000 | 151.4 | 5 |
| scikit-learn | Random Forest | 0.1032±0.0062 | 0.0033±0.0002 | 1.0000±0.0000 | 151.3 | 2 |

## Overall Performance Champions

- **🏃 Fastest Training**: Decision Tree (scikit-learn) - 0.0007s
- **⚡ Fastest Inference**: Decision Tree (scikit-learn) - 0.0001s
- **🎯 Best Accuracy**: SVM (scikit-learn) - 1.0000
- **💾 Most Memory Efficient**: Decision Tree (scikit-learn) - 150.6 MB

## Library Comparison Summary

### scikit-learn

- **Models tested**: 12
- **Average training time**: 0.0337s
- **Average inference time**: 0.0013s
- **Average accuracy**: 1.0000
- **Average memory usage**: 151.2 MB

## Methodology

### Enhanced Benchmarking Features

1. **Multiple Runs with Statistics**: Each model is trained and tested multiple times to collect statistical measures (mean, standard deviation, min, max)
2. **Warmup Runs**: Initial warmup runs are performed to account for JIT compilation and system caching effects
3. **Resource Monitoring**: CPU and memory usage is tracked throughout training and inference phases
4. **System Environment Capture**: Complete system information is recorded for reproducibility
5. **Separate Preprocessing Timing**: Data loading and preprocessing time is measured separately from model training
6. **High-Precision Timing**: Using `time.perf_counter()` for microsecond-level timing precision
7. **Algorithm Standardization**: Models are categorized and standardized across different libraries for fair comparison

### Algorithm Categorization

Models are standardized and grouped into categories:
- **Support Vector Machines**: All SVM/SVC variants (binary, multiclass, different kernels)
- **Tree-Based Models**: Decision Trees and Random Forests
- **Neural Networks**: Multi-layer perceptrons and neural network implementations
- **Distance-Based Models**: K-Nearest Neighbors and similar algorithms
- **Linear Models**: Logistic Regression and other linear algorithms

### Statistical Significance

All timing measurements include standard deviation to indicate the variability across multiple runs. This provides insight into the consistency and reliability of each algorithm's performance.

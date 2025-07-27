# Enhanced ML Benchmark Report

## System Information

- **CPU**: AMD Ryzen 5 4600G with Radeon Graphics
- **CPU Cores**: 12
- **Total Memory**: 19.4 GB
- **Library Version**: 1.7.1
- **Timestamp**: 2025-07-27T09:24:43.899524+00:00

## Dataset Information

- **Dataset**: iris
- **Total Samples**: 150
- **Features**: 4
- **Training Samples**: 120
- **Test Samples**: 30

## Performance Summary by Algorithm Category

### Support Vector Machines

| Library | Model | Training Time (s) | Inference Time (s) | Accuracy | Memory (MB) |
|---------|-------|------------------|-------------------|----------|-------------|
| scikit-learn | SVM | 0.0010±0.0000 | 0.0003±0.0000 | 1.0000±0.0000 | 151.7 |

### Tree-Based Models

| Library | Model | Training Time (s) | Inference Time (s) | Accuracy | Memory (MB) |
|---------|-------|------------------|-------------------|----------|-------------|
| scikit-learn | Decision Tree | 0.0007±0.0001 | 0.0001±0.0000 | 1.0000±0.0000 | 150.9 |
| scikit-learn | Random Forest | 0.1032±0.0062 | 0.0033±0.0002 | 1.0000±0.0000 | 151.3 |

## Performance Champions

- **🏃 Fastest Training**: Decision Tree (scikit-learn) - 0.0007s
- **⚡ Fastest Inference**: Decision Tree (scikit-learn) - 0.0001s
- **🎯 Best Accuracy**: Decision Tree (scikit-learn) - 1.0000
- **💾 Most Memory Efficient**: Decision Tree (scikit-learn) - 150.9 MB

## Methodology

### Enhanced Benchmarking Features

1. **Multiple Runs with Statistics**: Each model is trained and tested multiple times to collect statistical measures (mean, standard deviation, min, max)
2. **Warmup Runs**: Initial warmup runs are performed to account for JIT compilation and system caching effects
3. **Resource Monitoring**: CPU and memory usage is tracked throughout training and inference phases
4. **System Environment Capture**: Complete system information is recorded for reproducibility
5. **Separate Preprocessing Timing**: Data loading and preprocessing time is measured separately from model training
6. **High-Precision Timing**: Using `time.perf_counter()` for microsecond-level timing precision
7. **Algorithm Standardization**: Models are categorized and standardized across different libraries for fair comparison

### Statistical Significance

All timing measurements include standard deviation to indicate the variability across multiple runs. This provides insight into the consistency and reliability of each algorithm's performance.

### Resource Usage

Memory usage is monitored at three key points:
- **Baseline**: Memory usage before any model operations
- **Peak Training**: Maximum memory usage during training phase
- **Peak Inference**: Maximum memory usage during inference phase

### Algorithm Categorization

Models are standardized and grouped into categories:
- **Support Vector Machines**: All SVM/SVC variants (binary, multiclass, different kernels)
- **Tree-Based Models**: Decision Trees and Random Forests
- **Neural Networks**: Multi-layer perceptrons and neural network implementations
- **Distance-Based Models**: K-Nearest Neighbors and similar algorithms
- **Linear Models**: Logistic Regression and other linear algorithms

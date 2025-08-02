# Enhanced ML Benchmark Report

## System Information

- **Hostname**: shironeko
- **OS**: Linux 6.14.0-24-generic x86_64
- **CPU**: AMD Ryzen 5 4600G with Radeon Graphics
- **CPU Cores**: 12
- **Total Memory**: 19.4 GB
- **Library Version**: 1.7.1
- **Timestamp**: 2025-07-27T08:46:28.012397+00:00

## Dataset Information

- **Dataset**: iris
- **Total Samples**: 150
- **Features**: 4
- **Classes**: 3
- **Training Samples**: 120
- **Test Samples**: 30

## Performance Summary

| Model | Training Time (s) | Inference Time (s) | Accuracy | Memory Peak (MB) | Runs |
|-------|-------------------|-------------------|----------|------------------|------|
| DecisionTreeClassifier | 0.0007±0.0001 | 0.0002±0.0001 | 1.0000±0.0000 | 150.7 | 15 |
| RandomForestClassifier | 0.0941±0.0039 | 0.0034±0.0004 | 1.0000±0.0000 | 151.5 | 15 |
| SVC | 0.0012±0.0004 | 0.0002±0.0001 | 1.0000±0.0000 | 151.8 | 15 |

## Performance Champions

- **🏃 Fastest Training**: DecisionTreeClassifier (0.0007s)
- **⚡ Fastest Inference**: DecisionTreeClassifier (0.0002s)
- **🎯 Best Accuracy**: DecisionTreeClassifier (1.0000)
- **💾 Most Memory Efficient**: DecisionTreeClassifier (150.7 MB)

## Methodology

### Enhanced Benchmarking Features

1. **Multiple Runs with Statistics**: Each model is trained and tested multiple times to collect statistical measures (mean, standard deviation, min, max)
2. **Warmup Runs**: Initial warmup runs are performed to account for JIT compilation and system caching effects
3. **Resource Monitoring**: CPU and memory usage is tracked throughout training and inference phases
4. **System Environment Capture**: Complete system information is recorded for reproducibility
5. **Separate Preprocessing Timing**: Data loading and preprocessing time is measured separately from model training
6. **High-Precision Timing**: Using `time.perf_counter()` for microsecond-level timing precision

### Statistical Significance

All timing measurements include standard deviation to indicate the variability across multiple runs. This provides insight into the consistency and reliability of each algorithm's performance.

### Resource Usage

Memory usage is monitored at three key points:
- **Baseline**: Memory usage before any model operations
- **Peak Training**: Maximum memory usage during training phase
- **Peak Inference**: Maximum memory usage during inference phase


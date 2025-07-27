# ML Benchmark Report: Iris

## Dataset Information

- **Name**: iris
- **Size**: small (150 samples)
- **Features**: 4
- **Classes**: 3
- **Timestamp**: 2025-07-27 17:43:11

## Neural Networks

| Library | Algorithm | Training (s) | Inference (s) | Accuracy | Memory (MB) |
|---------|-----------|--------------|---------------|----------|-------------|
| scikit-learn | Neural Network | 0.147155±0.002092 | 0.000284±0.000145 | 1.0000±0.0000 | 1.81 |

## Support Vector Machines

| Library | Algorithm | Training (s) | Inference (s) | Accuracy | Memory (MB) |
|---------|-----------|--------------|---------------|----------|-------------|
| scikit-learn | SVM | 0.001365±0.000339 | 0.000233±0.000063 | 1.0000±0.0000 | 0.40 |
| scikit-learn | SVM (Linear) | 0.001000±0.000215 | 0.000207±0.000033 | 1.0000±0.0000 | 0.16 |

## Tree-Based Models

| Library | Algorithm | Training (s) | Inference (s) | Accuracy | Memory (MB) |
|---------|-----------|--------------|---------------|----------|-------------|
| scikit-learn | Decision Tree | 0.000988±0.000233 | 0.000140±0.000045 | 1.0000±0.0000 | 0.27 |
| scikit-learn | Random Forest | 0.093351±0.007551 | 0.003772±0.000291 | 1.0000±0.0000 | 0.72 |

## Performance Champions

- **🏃 Fastest Training**: Decision Tree (scikit-learn) - 0.000988s
- **⚡ Fastest Inference**: Decision Tree (scikit-learn) - 0.000140s
- **🎯 Best Accuracy**: Decision Tree (scikit-learn) - 1.0000
- **💾 Most Memory Efficient**: SVM (Linear) (scikit-learn) - 0.16 MB


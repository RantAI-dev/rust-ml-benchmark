# ML Benchmark Report: Iris

## Dataset Information

- **Name**: iris
- **Size**: small (150 samples)
- **Features**: 4
- **Classes**: 3
- **Timestamp**: 2025-07-27 19:18:13

## Neural Networks

| Library | Algorithm | Training (s) | Inference (s) | Accuracy | Memory (MB) |
|---------|-----------|--------------|---------------|----------|-------------|
| scikit-learn | Neural Network | 0.134045±nan | 0.000191±nan | 1.0000±nan | 1.75 |

## Other

| Library | Algorithm | Training (s) | Inference (s) | Accuracy | Memory (MB) |
|---------|-----------|--------------|---------------|----------|-------------|
| linfa | DecisionTree | 0.000107±nan | 0.000003±nan | 0.6897±nan | 2.12 |
| smartcore | SVC (One-vs-Rest) | 0.019242±nan | 0.001000±nan | 0.8333±nan | 0.62 |

## Support Vector Machines

| Library | Algorithm | Training (s) | Inference (s) | Accuracy | Memory (MB) |
|---------|-----------|--------------|---------------|----------|-------------|
| linfa | SVM (Binary) | 0.001045±nan | 0.000035±nan | 1.0000±nan | 0.71 |
| scikit-learn | SVM | 0.001716±nan | 0.000343±nan | 1.0000±nan | 0.39 |
| scikit-learn | SVM (Linear) | 0.001110±nan | 0.000196±nan | 1.0000±nan | 0.16 |

## Tree-Based Models

| Library | Algorithm | Training (s) | Inference (s) | Accuracy | Memory (MB) |
|---------|-----------|--------------|---------------|----------|-------------|
| scikit-learn | Decision Tree | 0.000992±nan | 0.000135±nan | 1.0000±nan | 0.31 |
| scikit-learn | Random Forest | 0.083311±nan | 0.003076±nan | 1.0000±nan | 0.75 |
| smartcore | Decision Tree | 0.000120±nan | 0.000006±nan | 0.7333±nan | 2.22 |
| smartcore | Random Forest | 0.002361±nan | 0.000228±nan | 0.7333±nan | 1.04 |

## Performance Champions

- **🏃 Fastest Training**: DecisionTree (linfa) - 0.000107s
- **⚡ Fastest Inference**: DecisionTree (linfa) - 0.000003s
- **🎯 Best Accuracy**: SVM (Binary) (linfa) - 1.0000
- **💾 Most Memory Efficient**: SVM (Linear) (scikit-learn) - 0.16 MB


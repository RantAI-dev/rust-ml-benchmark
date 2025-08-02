# ML Benchmark Report: Iris

## Dataset Information

- **Name**: iris
- **Size**: small (150 samples)
- **Features**: 4
- **Classes**: 3
- **Timestamp**: 2025-07-27 19:09:10

## Neural Networks

| Library | Algorithm | Training (s) | Inference (s) | Accuracy | Memory (MB) |
|---------|-----------|--------------|---------------|----------|-------------|
| scikit-learn | Neural Network | 0.127972±nan | 0.000191±nan | 1.0000±nan | 1.75 |

## Other

| Library | Algorithm | Training (s) | Inference (s) | Accuracy | Memory (MB) |
|---------|-----------|--------------|---------------|----------|-------------|
| linfa | DecisionTree | 0.000103±nan | 0.000003±nan | 0.6897±nan | 1788.00 |
| smartcore | SVC (One-vs-Rest) | 0.018307±nan | 0.001000±nan | 0.8333±nan | 640.00 |

## Support Vector Machines

| Library | Algorithm | Training (s) | Inference (s) | Accuracy | Memory (MB) |
|---------|-----------|--------------|---------------|----------|-------------|
| linfa | SVM (Binary) | 0.000995±nan | 0.000035±nan | 1.0000±nan | 844.00 |
| scikit-learn | SVM | 0.001079±nan | 0.000176±nan | 1.0000±nan | 0.40 |
| scikit-learn | SVM (Linear) | 0.000812±nan | 0.000162±nan | 1.0000±nan | 0.16 |

## Tree-Based Models

| Library | Algorithm | Training (s) | Inference (s) | Accuracy | Memory (MB) |
|---------|-----------|--------------|---------------|----------|-------------|
| scikit-learn | Decision Tree | 0.000767±nan | 0.000099±nan | 1.0000±nan | 0.28 |
| scikit-learn | Random Forest | 0.087730±nan | 0.004114±nan | 1.0000±nan | 0.79 |
| smartcore | Decision Tree | 0.000102±nan | 0.000006±nan | 0.7333±nan | 1904.00 |
| smartcore | Random Forest | 0.002427±nan | 0.000198±nan | 0.7333±nan | 1028.00 |

## Performance Champions

- **🏃 Fastest Training**: Decision Tree (smartcore) - 0.000102s
- **⚡ Fastest Inference**: DecisionTree (linfa) - 0.000003s
- **🎯 Best Accuracy**: SVM (Binary) (linfa) - 1.0000
- **💾 Most Memory Efficient**: SVM (Linear) (scikit-learn) - 0.16 MB


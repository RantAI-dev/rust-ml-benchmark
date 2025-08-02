# ML Benchmark Report: Iris

## Dataset Information

- **Name**: iris
- **Size**: small (150 samples)
- **Features**: 4
- **Classes**: 3
- **Timestamp**: 2025-07-27 18:58:56

## Neural Networks

## Other

## Support Vector Machines

## Tree-Based Models

| Library | Algorithm | Training (s) | Inference (s) | Accuracy | Memory (MB) |
|---------|-----------|--------------|---------------|----------|-------------|
| scikit-learn | Decision Tree | 0.000756±0.000003 | 0.000112±0.000026 | 1.0000±0.0000 | 0.27 |

| scikit-learn | Random Forest | 0.091370±0.001074 | 0.003227±0.000071 | 1.0000±0.0000 | 0.65 |

| smartcore | Decision Tree | 0.000162±nan | 0.000007±nan | 0.7333±nan | 2292.00 |

| smartcore | Random Forest | 0.002361±nan | 0.000198±nan | 0.7333±nan | 984.00 |

## Performance Champions

- **🏃 Fastest Training**: DecisionTree (linfa) - 0.000115s
- **⚡ Fastest Inference**: DecisionTree (linfa) - 0.000003s
- **🎯 Best Accuracy**: SVM (Binary) (linfa) - 1.0000
- **💾 Most Memory Efficient**: SVM (Linear) (scikit-learn) - 0.08 MB


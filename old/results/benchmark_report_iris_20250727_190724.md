# ML Benchmark Report: Iris

## Dataset Information

- **Name**: iris
- **Size**: small (150 samples)
- **Features**: 4
- **Classes**: 3
- **Timestamp**: 2025-07-27 19:07:24

## Neural Networks

## Other

## Support Vector Machines

## Tree-Based Models

| Library | Algorithm | Training (s) | Inference (s) | Accuracy | Memory (MB) |
|---------|-----------|--------------|---------------|----------|-------------|
| scikit-learn | Decision Tree | 0.001336±nan | 0.000412±nan | 1.0000±nan | 0.17 |

| scikit-learn | Random Forest | 0.086209±nan | 0.003599±nan | 1.0000±nan | 0.50 |

| smartcore | Decision Tree | 0.000098±nan | 0.000007±nan | 0.7333±nan | 2328.00 |

| smartcore | Random Forest | 0.002510±nan | 0.000198±nan | 0.7333±nan | 980.00 |

## Performance Champions

- **🏃 Fastest Training**: Decision Tree (smartcore) - 0.000098s
- **⚡ Fastest Inference**: DecisionTree (linfa) - 0.000002s
- **🎯 Best Accuracy**: SVM (Binary) (linfa) - 1.0000
- **💾 Most Memory Efficient**: SVM (Linear) (scikit-learn) - 0.00 MB


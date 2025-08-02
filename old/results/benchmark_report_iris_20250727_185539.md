# ML Benchmark Report: Iris

## Dataset Information

- **Name**: iris
- **Size**: small (150 samples)
- **Features**: 4
- **Classes**: 3
- **Timestamp**: 2025-07-27 18:55:39

## Neural Networks

## Support Vector Machines

## Tree-Based Models

| Library | Algorithm | Training (s) | Inference (s) | Accuracy | Memory (MB) |
|---------|-----------|--------------|---------------|----------|-------------|
| scikit-learn | Decision Tree | 0.000762±0.000003 | 0.000110±0.000020 | 1.0000±0.0000 | 0.15 |

| scikit-learn | Random Forest | 0.086421±0.002953 | 0.003144±0.000039 | 1.0000±0.0000 | 0.58 |

## Performance Champions

- **🏃 Fastest Training**: Decision Tree (scikit-learn) - 0.000762s
- **⚡ Fastest Inference**: Decision Tree (scikit-learn) - 0.000110s
- **🎯 Best Accuracy**: Decision Tree (scikit-learn) - 1.0000
- **💾 Most Memory Efficient**: SVM (Linear) (scikit-learn) - 0.00 MB


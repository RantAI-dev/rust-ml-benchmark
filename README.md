# Rust vs. Python for ML: A Performance Benchmark

This repository contains a clean, simple benchmark suite for comparing the performance of Rust-based machine learning libraries against their established Python counterparts. The focus is on classical machine learning algorithms across different dataset sizes.

## 🎯 Simple & Clean Benchmark Suite

### **Dataset-Centric Organization**

The benchmark suite is organized by dataset size for fair comparison:

- **🌼 Small Dataset**: Iris (150 samples, 4 features) - Fast iteration and development
- **🏦 Large Dataset**: Bank Marketing (41,188 samples, 20 features) - Production-scale performance

### **Library Coverage**

- **🐍 Python**: scikit-learn (industry standard)
- **🦀 Rust**: SmartCore and Linfa (emerging Rust ML libraries)

### **Algorithm Standardization**

All algorithms are standardized across libraries for fair comparison:
- **SVM**: Support Vector Machines (various kernels)
- **Decision Tree**: Tree-based classification
- **Random Forest**: Ensemble tree methods
- **Neural Network**: Multi-layer perceptrons

#### Quick Start

```bash
# Run all benchmarks (both datasets)
./run_benchmarks.sh

# Run single dataset benchmark
python3 scripts/benchmark_runner.py --dataset data/iris.csv --runs 5

# Run large dataset benchmark
python3 scripts/benchmark_runner.py --dataset data/bank-additional/bank-additional-full.csv --runs 3
```

## Repository Structure

```
rust-ml-benchmark/
├── data/                        # Input datasets
│   ├── iris.csv                 # 🌼 Small dataset (150 samples)
│   └── bank-additional/         # 🏦 Large dataset (41K samples)
├── results/                     # Output reports and plots
├── scripts/                     # Benchmark source code
│   ├── benchmark_runner.py      # 🎯 Main benchmark orchestrator
│   ├── sklearn_runner.py        # 🐍 Python (scikit-learn) runner
│   ├── smartcore_bencher/       # 🦀 SmartCore Rust benchmarks
│   └── linfa_bencher/           # 🦀 Linfa Rust benchmarks
├── run_benchmarks.sh            # 🚀 One-command benchmark runner
└── README.md                    # This file
```

### Benchmark Features

| Feature | Description |
|---------|-------------|
| **Dataset-Centric** | Organized by dataset size (small vs large) |
| **Algorithm Standardization** | Consistent naming across libraries |
| **Multiple Runs** | Statistical significance with warmup |
| **Resource Monitoring** | Memory usage tracking |
| **Fair Comparison** | Same algorithms compared across libraries |
| **Clean Output** | Simple reports and visualizations |

## How to Run

### Prerequisites

- Rust & Cargo
- Python 3.8+ with `venv`

### Setup

1. Clone the repository:

```bash
git clone <your-repo-url>
cd rust-ml-benchmark
```

2. Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install Python dependencies:

```bash
pip install pandas scikit-learn matplotlib seaborn psutil
```

### 🚀 Benchmark Execution

#### **One-Command (Recommended)**

```bash
# Run all benchmarks (both datasets)
./run_benchmarks.sh
```

#### **Manual Execution**

```bash
# Run single dataset benchmark
python3 scripts/benchmark_runner.py --dataset data/iris.csv --runs 5

# Run large dataset benchmark
python3 scripts/benchmark_runner.py --dataset data/bank-additional/bank-additional-full.csv --runs 3
```

## 📊 Sample Results

### 🌼 Iris Dataset (Small - 150 samples, 4 features)

#### Support Vector Machines

| **Library** | **Algorithm** | **Training (s)** | **Inference (s)** | **Accuracy** | **Memory (MB)** |
|-------------|---------------|------------------|-------------------|--------------|-----------------|
| **scikit-learn** | SVM | 0.001716±nan | 0.000343±nan | 1.0000±nan | 0.39 |
| **scikit-learn** | SVM (Linear) | 0.001110±nan | 0.000196±nan | 1.0000±nan | 0.16 |
| **linfa** | SVM (Binary) | 0.001045±nan | 0.000035±nan | 1.0000±nan | 0.71 |

#### Tree-Based Models

| **Library** | **Algorithm** | **Training (s)** | **Inference (s)** | **Accuracy** | **Memory (MB)** |
|-------------|---------------|------------------|-------------------|--------------|-----------------|
| **scikit-learn** | Decision Tree | 0.000992±nan | 0.000135±nan | 1.0000±nan | 0.31 |
| **scikit-learn** | Random Forest | 0.083311±nan | 0.003076±nan | 1.0000±nan | 0.75 |
| **smartcore** | Decision Tree | 0.000120±nan | 0.000006±nan | 0.7333±nan | 2.22 |
| **smartcore** | Random Forest | 0.002361±nan | 0.000228±nan | 0.7333±nan | 1.04 |
| **linfa** | DecisionTree | 0.000107±nan | 0.000003±nan | 0.6897±nan | 2.12 |

#### Neural Networks

| **Library** | **Algorithm** | **Training (s)** | **Inference (s)** | **Accuracy** | **Memory (MB)** |
|-------------|---------------|------------------|-------------------|--------------|-----------------|
| **scikit-learn** | Neural Network | 0.134045±nan | 0.000191±nan | 1.0000±nan | 1.75 |

#### Performance Champions (Iris Dataset)

- **🏃 Fastest Training**: Decision Tree (smartcore) - 0.000102s
- **⚡ Fastest Inference**: DecisionTree (linfa) - 0.000003s
- **🎯 Best Accuracy**: SVM (Binary) (linfa) - 1.0000
- **💾 Most Memory Efficient**: SVM (Linear) (scikit-learn) - 0.16 MB

### 🏦 Bank Dataset (Large - 41,188 samples, 20 features)

*Benchmark results will be added after running the full dataset comparison. The bank dataset is significantly larger and will provide insights into production-scale performance differences between Rust and Python ML libraries.*

#### Expected Algorithms to Test:
- **Support Vector Machines**: Linear SVM, RBF SVM
- **Tree-Based Models**: Decision Trees, Random Forests
- **Neural Networks**: Multi-layer Perceptrons

#### Expected Metrics:
- Training time (seconds)
- Inference time (seconds)
- Accuracy scores
- Memory usage (MB)
- CPU utilization (%)
- Peak memory consumption

## 📈 Output Files

The benchmark generates:

- **📝 `benchmark_report_*.md`**: Comprehensive markdown reports with results by algorithm category
- **📊 `benchmark_plots_*.png`**: Performance visualizations (training time, inference time, accuracy, memory usage)
- **📄 `benchmark_data_*.csv`**: Raw benchmark data for further analysis

## 🔬 Technical Details

### Algorithm Standardization

The benchmark standardizes algorithm names across libraries:

- **SVM variations**: `SVC`, `SVM (One-vs-Rest)`, `SVM (Binary)`, `LinearSVC` → `SVM`, `SVM (Binary)`, `SVM (Linear)`
- **Tree variations**: `DecisionTreeClassifier`, `Decision Tree` → `Decision Tree`
- **Forest variations**: `RandomForestClassifier`, `Random Forest` → `Random Forest`
- **Neural networks**: `MLPClassifier`, `MLP` → `Neural Network`

### Dataset Handling

- **Iris dataset**: Standard CSV format, last column is target
- **Bank dataset**: Semicolon-separated, categorical features encoded
- **Preprocessing**: Standard scaling for SVM/MLP, raw data for trees

### Performance Metrics

- **Training time**: Model fitting duration (seconds)
- **Inference time**: Prediction duration (seconds)
- **Accuracy**: Classification accuracy on test set
- **Memory usage**: Peak memory consumption during training (MB)

## 🎯 Key Findings

### Small Dataset (Iris) - Latest Results
- **🏃 Fastest training**: Decision Tree (smartcore) - 0.000102s
- **⚡ Fastest inference**: DecisionTree (linfa) - 0.000003s
- **🎯 Best accuracy**: SVM (Binary) (linfa) - 100%
- **💾 Most memory efficient**: SVM (Linear) (scikit-learn) - 0.16 MB

### Large Dataset (Bank) - Coming Soon
- **Expected insights**: Production-scale performance comparison
- **Focus areas**: Training time, memory efficiency, CPU utilization
- **Target**: Understanding Rust vs Python trade-offs at scale

## Contributing

When adding new benchmarks:

1. **Keep it simple**: Focus on core ML algorithms
2. **Standardize names**: Use consistent algorithm naming
3. **Dataset-centric**: Organize by dataset size
4. **Fair comparison**: Compare same algorithms across libraries

## License

This project is licensed under the MIT License.
#!/bin/bash

# Rust vs Python ML Benchmark Runner
# Simple script to run benchmarks on both datasets

set -e

echo "🚀 Rust vs Python ML Benchmark Suite"
echo "====================================="

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Install required packages if not already installed
echo "📦 Checking dependencies..."
pip install -q pandas scikit-learn matplotlib seaborn psutil

# Create results directory
mkdir -p results

# Run benchmarks on small dataset (iris)
echo ""
echo "🌼 Running benchmarks on IRIS dataset (small)..."
python3 scripts/benchmark_runner.py \
    --dataset data/iris.csv \
    --output-dir results \
    --runs 5 \
    --warmup 2

# Run benchmarks on large dataset (bank)
echo ""
echo "🏦 Running benchmarks on BANK dataset (large)..."
python3 scripts/benchmark_runner.py \
    --dataset data/bank-additional/bank-additional-full.csv \
    --output-dir results \
    --runs 3 \
    --warmup 1

echo ""
echo "✅ All benchmarks complete!"
echo "📁 Results saved in: results/"
echo ""
echo "📊 Generated files:"
ls -la results/*.md results/*.png 2>/dev/null || echo "No report files found yet" 
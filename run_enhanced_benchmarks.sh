#!/bin/bash

# Enhanced ML Benchmark Suite - Integration Script
# Run all enhanced benchmarks with configurable parameters

set -e

# Default parameters
DATASET="${DATASET:-data/bank-additional/bank-additional-full.csv}"
RUNS="${RUNS:-5}"
WARMUP="${WARMUP:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_step() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Enhanced ML Benchmark Suite - Run comprehensive benchmarks

OPTIONS:
    -d, --dataset     Dataset path (default: data/bank-additional/bank-additional-full.csv)
    -r, --runs        Number of measurement runs (default: 5)
    -w, --warmup      Number of warmup runs (default: 1)
    -o, --output-dir  Output directory (default: results)
    -p, --python-only Run only Python benchmarks
    -R, --rust-only   Run only Rust benchmarks
    -a, --analyze     Run analysis after benchmarks
    -h, --help        Show this help

EXAMPLES:
    # Run all benchmarks with defaults
    $0

    # Run with more statistical samples
    $0 --runs 10 --warmup 3

    # Run only Python benchmark
    $0 --python-only --runs 15

    # Run with custom dataset
    $0 --dataset data/iris.csv --runs 5

    # Run and generate analysis
    $0 --runs 10 --analyze

ENVIRONMENT VARIABLES:
    DATASET     Override default dataset path
    RUNS        Override default run count
    WARMUP      Override default warmup count
    OUTPUT_DIR  Override default output directory
EOF
}

# Parse command line arguments
PYTHON_ONLY=false
RUST_ONLY=false
ANALYZE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -r|--runs)
            RUNS="$2"
            shift 2
            ;;
        -w|--warmup)
            WARMUP="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -p|--python-only)
            PYTHON_ONLY=true
            shift
            ;;
        -R|--rust-only)
            RUST_ONLY=true
            shift
            ;;
        -a|--analyze)
            ANALYZE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ ! -f "$DATASET" ]]; then
    print_error "Dataset file not found: $DATASET"
    exit 1
fi

if [[ ! -d "$OUTPUT_DIR" ]]; then
    mkdir -p "$OUTPUT_DIR"
    print_step "Created output directory: $OUTPUT_DIR"
fi

# Main execution
print_header "🚀 Enhanced ML Benchmark Suite"
echo "Configuration:"
echo "  📊 Dataset: $DATASET"
echo "  🔄 Runs: $RUNS (+ $WARMUP warmup)"
echo "  📁 Output: $OUTPUT_DIR"
echo "  🧪 Mode: $(if $PYTHON_ONLY; then echo "Python only"; elif $RUST_ONLY; then echo "Rust only"; else echo "All libraries"; fi)"
echo

# Timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Python benchmarks
if ! $RUST_ONLY; then
    print_header "🐍 Enhanced Python Benchmark (scikit-learn)"
    
    PYTHON_OUTPUT="$OUTPUT_DIR/enhanced_sklearn_${TIMESTAMP}.txt"
    
    # Detect if this is an iris dataset and use appropriate script
    if [[ "$DATASET" == *"iris"* ]]; then
        PYTHON_SCRIPT="scripts/enhanced_sklearn_iris.py"
    else
        PYTHON_SCRIPT="scripts/enhanced_sklearn_iris.py"  # For now, use iris script for all
    fi
    
    if python3 "$PYTHON_SCRIPT" \
        --dataset "$DATASET" \
        --output "$PYTHON_OUTPUT" \
        --runs "$RUNS" \
        --warmup "$WARMUP"; then
        print_step "Python benchmark completed: $PYTHON_OUTPUT"
    else
        print_error "Python benchmark failed"
        exit 1
    fi
    echo
fi

# Rust benchmarks
if ! $PYTHON_ONLY; then
    # Detect if this is an iris dataset
    if [[ "$DATASET" == *"iris"* ]]; then
        ENHANCED_BINARY="enhanced_iris_wrapper"
        DATASET_TYPE="iris"
    else
        ENHANCED_BINARY="enhanced_wrapper"
        DATASET_TYPE="bank"
    fi
    
    # SmartCore benchmark
    print_header "🦀 Enhanced Rust Benchmark (SmartCore)"
    
    SMARTCORE_OUTPUT="$OUTPUT_DIR/enhanced_smartcore_${TIMESTAMP}.txt"
    ABSOLUTE_DATASET=$(realpath "$DATASET")
    
    cd scripts/smartcore_bencher
    if cargo run --release --bin "$ENHANCED_BINARY" -- \
        --dataset "$ABSOLUTE_DATASET" \
        --output "../../$SMARTCORE_OUTPUT" \
        --runs "$RUNS" \
        --warmup "$WARMUP"; then
        print_step "SmartCore benchmark completed: $SMARTCORE_OUTPUT"
    else
        print_error "SmartCore benchmark failed"
        cd ../..
        exit 1
    fi
    cd ../..
    echo
    
    # Linfa benchmark
    print_header "🦀 Enhanced Rust Benchmark (Linfa)"
    
    LINFA_OUTPUT="$OUTPUT_DIR/enhanced_linfa_${TIMESTAMP}.txt"
    
    cd scripts/linfa_bencher
    if cargo run --release --bin "$ENHANCED_BINARY" -- \
        --dataset "$ABSOLUTE_DATASET" \
        --output "../../$LINFA_OUTPUT" \
        --runs "$RUNS" \
        --warmup "$WARMUP"; then
        print_step "Linfa benchmark completed: $LINFA_OUTPUT"
    else
        print_error "Linfa benchmark failed"
        cd ../..
        exit 1
    fi
    cd ../..
    echo
fi

# Analysis
if $ANALYZE; then
    print_header "📊 Enhanced Benchmark Analysis"
    
    # Find the most recent results for analysis
    if ! $RUST_ONLY && [[ -f "$PYTHON_OUTPUT" ]]; then
        ANALYSIS_INPUT="$PYTHON_OUTPUT"
        ANALYSIS_OUTPUT="$OUTPUT_DIR/enhanced_analysis_${TIMESTAMP}"
        
        if python3 scripts/enhanced_analysis.py \
            --results "$ANALYSIS_INPUT" \
            --output "$ANALYSIS_OUTPUT"; then
            print_step "Analysis completed: $ANALYSIS_OUTPUT"
        else
            print_warning "Analysis failed, but benchmarks completed successfully"
        fi
    else
        print_warning "Analysis requires Python benchmark results"
    fi
    echo
fi

# Summary
print_header "✅ Enhanced Benchmark Suite Complete"
echo "Results saved in: $OUTPUT_DIR"
echo "Timestamp: $TIMESTAMP"
echo

if ! $RUST_ONLY; then
    echo "📊 Python Results: $PYTHON_OUTPUT"
fi

if ! $PYTHON_ONLY; then
    echo "🦀 SmartCore Results: $SMARTCORE_OUTPUT"
    echo "🦀 Linfa Results: $LINFA_OUTPUT"
fi

if $ANALYZE && [[ -n "$ANALYSIS_OUTPUT" ]]; then
    echo "📈 Analysis: $ANALYSIS_OUTPUT*"
fi

echo
print_step "All enhanced benchmarks completed successfully!"

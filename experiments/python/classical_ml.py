import time
import json
import argparse
import os
import pandas as pd
import psutil # For memory tracking
import numpy as np # For percentile calculations
import threading

from sklearn.datasets import fetch_california_housing, load_digits, make_blobs
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
from sklearn.model_selection import train_test_split

# Define the path for processed data
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')

class ResourceMonitor:
    """Monitor CPU and memory usage during benchmark execution."""
    
    def __init__(self):
        self.cpu_samples = []
        self.memory_samples = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start monitoring system resources."""
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring and return statistics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
        if not self.cpu_samples or not self.memory_samples:
            return {
                "cpu_utilization_mean_percent": 0,
                "cpu_utilization_peak_percent": 0,
                "memory_mean_mb": 0,
                "memory_peak_mb": 0
            }
            
        return {
            "cpu_utilization_mean_percent": np.mean(self.cpu_samples),
            "cpu_utilization_peak_percent": np.max(self.cpu_samples),
            "memory_mean_mb": np.mean(self.memory_samples),
            "memory_peak_mb": np.max(self.memory_samples)
        }
        
    def _monitor_loop(self):
        """Background monitoring loop."""
        process = psutil.Process(os.getpid())
        # Initial call to cpu_percent() to establish baseline
        process.cpu_percent()
        time.sleep(0.1)  # Wait before first real measurement
        
        while self.monitoring:
            try:
                # Get process-specific CPU usage
                cpu_percent = process.cpu_percent()
                # Also get system-wide CPU usage as backup
                system_cpu = psutil.cpu_percent(interval=None)
                memory_mb = process.memory_info().rss / (1024 * 1024)
                
                # Use the higher of process-specific or proportional system CPU
                effective_cpu = max(cpu_percent, system_cpu / psutil.cpu_count())
                
                self.cpu_samples.append(effective_cpu)
                self.memory_samples.append(memory_mb)
                
                time.sleep(0.05)  # Sample every 50ms for better granularity
            except:
                break

def get_process_memory():
    """Returns the memory usage of the current process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def prepare_datasets():
    """
    Checks if the datasets exist, and if not, downloads and saves them as CSV files.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    # --- California Housing for Regression ---
    housing_path = os.path.join(DATA_DIR, 'california_housing.csv')
    if not os.path.exists(housing_path):
        print("Preparing California Housing dataset...")
        housing = fetch_california_housing()
        df = pd.DataFrame(housing.data, columns=housing.feature_names)
        df['target'] = housing.target
        df.to_csv(housing_path, index=False)
        print(f"Saved to {housing_path}")
    # --- Digits for Classification ---
    digits_path = os.path.join(DATA_DIR, 'digits.csv')
    if not os.path.exists(digits_path):
        print("Preparing Digits dataset...")
        digits = load_digits()
        df = pd.DataFrame(digits.data, columns=[f'pixel_{i}' for i in range(digits.data.shape[1])])
        df['target'] = digits.target
        df.to_csv(digits_path, index=False)
        print(f"Saved to {digits_path}")

def run_linear_regression():
    """Benchmarks Linear Regression with comprehensive metrics."""
    monitor = ResourceMonitor()
    
    # Load data
    df = pd.read_csv(os.path.join(DATA_DIR, 'california_housing.csv'))
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    
    # --- Benchmark Training ---
    mem_before_train = get_process_memory()
    monitor.start_monitoring()
    
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    train_duration = time.perf_counter() - start_time
    
    resource_stats = monitor.stop_monitoring()
    mem_after_train = get_process_memory()
    
    # --- Benchmark Inference ---
    # Single sample latency measurements
    latencies = []
    latency_test_set = X_test.head(500)  # Use subset for latency measurement
    
    for i in range(len(latency_test_set)):
        single_sample = latency_test_set.iloc[[i]]
        inference_start_time = time.perf_counter()
        model.predict(single_sample)
        latencies.append(time.perf_counter() - inference_start_time)
    
    # Batch inference throughput
    batch_start_time = time.perf_counter()
    predictions = model.predict(X_test)
    batch_duration = time.perf_counter() - batch_start_time
    
    # --- Evaluate Quality ---
    score = r2_score(y_test, predictions)
    
    return {
        # Performance Metrics
        "total_training_time_seconds": train_duration,
        "training_throughput_samples_per_sec": len(X_train) / train_duration,
        "inference_throughput_samples_per_sec": len(X_test) / batch_duration,
        "inference_latency_mean_seconds": np.mean(latencies),
        "inference_latency_p50_seconds": np.percentile(latencies, 50),
        "inference_latency_p90_seconds": np.percentile(latencies, 90),
        "inference_latency_p99_seconds": np.percentile(latencies, 99),
        
        # Resource Utilization Metrics
        "cpu_utilization_mean_percent": resource_stats["cpu_utilization_mean_percent"],
        "cpu_utilization_peak_percent": resource_stats["cpu_utilization_peak_percent"],
        "memory_mean_mb": resource_stats["memory_mean_mb"],
        "memory_peak_mb": resource_stats["memory_peak_mb"],
        "memory_delta_mb": mem_after_train - mem_before_train,
        
        # Quality and Correctness Metrics
        "model_accuracy": score,
        "metric_type": "r2_score",
        
        # Additional metadata
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "features": X.shape[1]
    }

def run_svm():
    """Benchmarks Support Vector Machine classification with comprehensive metrics."""
    monitor = ResourceMonitor()
    
    # Load data
    df = pd.read_csv(os.path.join(DATA_DIR, 'digits.csv'))
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = SVC(kernel='linear', random_state=42)
    
    # --- Benchmark Training ---
    mem_before_train = get_process_memory()
    monitor.start_monitoring()
    
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    train_duration = time.perf_counter() - start_time
    
    resource_stats = monitor.stop_monitoring()
    mem_after_train = get_process_memory()
    
    # --- Benchmark Inference ---
    # Single sample latency measurements
    latencies = []
    latency_test_set = X_test.head(100)  # Smaller subset for SVM due to slower inference
    
    for i in range(len(latency_test_set)):
        single_sample = X_test.iloc[[i]]
        inference_start_time = time.perf_counter()
        model.predict(single_sample)
        latencies.append(time.perf_counter() - inference_start_time)
    
    # Batch inference throughput
    batch_start_time = time.perf_counter()
    predictions = model.predict(X_test)
    batch_duration = time.perf_counter() - batch_start_time
    
    # --- Evaluate Quality ---
    score = accuracy_score(y_test, predictions)
    
    return {
        # Performance Metrics
        "total_training_time_seconds": train_duration,
        "training_throughput_samples_per_sec": len(X_train) / train_duration,
        "inference_throughput_samples_per_sec": len(X_test) / batch_duration,
        "inference_latency_mean_seconds": np.mean(latencies),
        "inference_latency_p50_seconds": np.percentile(latencies, 50),
        "inference_latency_p90_seconds": np.percentile(latencies, 90),
        "inference_latency_p99_seconds": np.percentile(latencies, 99),
        
        # Resource Utilization Metrics
        "cpu_utilization_mean_percent": resource_stats["cpu_utilization_mean_percent"],
        "cpu_utilization_peak_percent": resource_stats["cpu_utilization_peak_percent"],
        "memory_mean_mb": resource_stats["memory_mean_mb"],
        "memory_peak_mb": resource_stats["memory_peak_mb"],
        "memory_delta_mb": mem_after_train - mem_before_train,
        
        # Quality and Correctness Metrics
        "model_accuracy": score,
        "metric_type": "accuracy",
        
        # Additional metadata
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "features": X.shape[1],
        "classes": len(np.unique(y))
    }

def run_kmeans():
    """Benchmarks K-Means clustering with comprehensive metrics."""
    monitor = ResourceMonitor()
    
    # Generate synthetic data
    X, y_true = make_blobs(n_samples=10000, centers=8, n_features=20, random_state=42)
    model = KMeans(n_clusters=8, random_state=42, n_init=10)
    
    # --- Benchmark Training (fit) ---
    mem_before_train = get_process_memory()
    monitor.start_monitoring()
    
    start_time = time.perf_counter()
    model.fit(X)
    train_duration = time.perf_counter() - start_time
    
    resource_stats = monitor.stop_monitoring()
    mem_after_train = get_process_memory()
    
    # --- Benchmark Inference (predict) ---
    # Single sample latency measurements
    latencies = []
    latency_test_set = X[:100]  # Use first 100 samples for latency measurement
    
    for i in range(len(latency_test_set)):
        single_sample = latency_test_set[i:i+1]
        inference_start_time = time.perf_counter()
        model.predict(single_sample)
        latencies.append(time.perf_counter() - inference_start_time)
    
    # Batch inference throughput
    batch_start_time = time.perf_counter()
    labels = model.predict(X)
    batch_duration = time.perf_counter() - batch_start_time
    
    # --- Evaluate Quality ---
    score = silhouette_score(X, labels)
    
    return {
        # Performance Metrics
        "total_training_time_seconds": train_duration,
        "training_throughput_samples_per_sec": len(X) / train_duration,
        "inference_throughput_samples_per_sec": len(X) / batch_duration,
        "inference_latency_mean_seconds": np.mean(latencies),
        "inference_latency_p50_seconds": np.percentile(latencies, 50),
        "inference_latency_p90_seconds": np.percentile(latencies, 90),
        "inference_latency_p99_seconds": np.percentile(latencies, 99),
        
        # Resource Utilization Metrics
        "cpu_utilization_mean_percent": resource_stats["cpu_utilization_mean_percent"],
        "cpu_utilization_peak_percent": resource_stats["cpu_utilization_peak_percent"],
        "memory_mean_mb": resource_stats["memory_mean_mb"],
        "memory_peak_mb": resource_stats["memory_peak_mb"],
        "memory_delta_mb": mem_after_train - mem_before_train,
        
        # Quality and Correctness Metrics
        "model_accuracy": score,
        "metric_type": "silhouette_score",
        
        # Additional metadata
        "training_samples": len(X),
        "features": X.shape[1],
        "clusters": 8
    }

def main():
    parser = argparse.ArgumentParser(description="Run classical ML benchmarks for Python.")
    parser.add_argument(
        '--algorithm',
        type=str,
        required=False,
        choices=['linear_regression', 'svm', 'kmeans'],
        help='The algorithm to benchmark.'
    )
    parser.add_argument(
        '--prepare',
        action='store_true',
        help='If set, downloads and prepares the datasets.'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path for benchmark results (optional).'
    )
    args = parser.parse_args()

    if args.prepare:
        prepare_datasets()
        return
    
    if not args.algorithm:
        parser.error("The --algorithm argument is required when not using --prepare.")

    # Add timestamp and system info
    results = {
        "language": "python",
        "algorithm": args.algorithm,
        "timestamp": time.time(),
        "system_info": {
            "cpu_count": psutil.cpu_count(),
            "total_memory_mb": psutil.virtual_memory().total / (1024 * 1024)
        },
        "benchmark_results": None
    }

    print(f"Running {args.algorithm} benchmark...")
    
    if args.algorithm == 'linear_regression':
        results["benchmark_results"] = run_linear_regression()
    elif args.algorithm == 'svm':
        results["benchmark_results"] = run_svm()
    elif args.algorithm == 'kmeans':
        results["benchmark_results"] = run_kmeans()

    # Output results
    output_json = json.dumps(results, indent=4)
    print(output_json)
    
    # Save to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_json)
        print(f"\nResults saved to {args.output}")

if __name__ == '__main__':
    main()
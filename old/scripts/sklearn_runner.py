#!/usr/bin/env python3
"""
Simple scikit-learn benchmark runner
"""

import json
import time
import psutil
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def benchmark_algorithm(X_train, X_test, y_train, y_test, model, model_name):
    """Benchmark a single algorithm."""
    # Record baseline memory
    baseline_memory = get_memory_usage()
    
    # Training
    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - train_start
    
    # Record peak training memory
    peak_training_memory = get_memory_usage()
    
    # Inference
    inference_start = time.perf_counter()
    predictions = model.predict(X_test)
    inference_time = time.perf_counter() - inference_start
    
    # Record peak inference memory
    peak_inference_memory = get_memory_usage()
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    return {
        'library': 'scikit-learn',
        'model': model_name,
        'train_time_sec': train_time,
        'inference_time_sec': inference_time,
        'accuracy': accuracy,
        'memory_usage_mb': peak_training_memory - baseline_memory
    }

def main():
    parser = argparse.ArgumentParser(description='scikit-learn benchmark runner')
    parser.add_argument('--dataset', required=True, help='Path to dataset CSV')
    parser.add_argument('--output', required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    # Load dataset
    df = pd.read_csv(args.dataset)
    
    # Prepare data (assuming last column is target)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features for SVM and MLP
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = []
    
    # Define algorithms to test
    algorithms = [
        (SVC(random_state=42), 'SVC'),
        (LinearSVC(random_state=42), 'LinearSVC'),
        (DecisionTreeClassifier(random_state=42), 'DecisionTreeClassifier'),
        (RandomForestClassifier(n_estimators=100, random_state=42), 'RandomForestClassifier'),
        (MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500), 'MLPClassifier')
    ]
    
    # Run benchmarks
    for model, name in algorithms:
        try:
            # Use scaled data for SVM and MLP, raw data for trees
            if name in ['SVC', 'LinearSVC', 'MLPClassifier']:
                result = benchmark_algorithm(X_train_scaled, X_test_scaled, y_train, y_test, model, name)
            else:
                result = benchmark_algorithm(X_train, X_test, y_train, y_test, model, name)
            
            results.append(result)
            
        except Exception as e:
            print(f"Error benchmarking {name}: {e}")
            continue
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main() 
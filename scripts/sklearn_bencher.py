import argparse
import time
import pandas as pd
import psutil
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def get_memory_usage_mb():
    """Returns the memory usage of the current process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_benchmark(model, X_train, y_train, X_test, y_test):
    """A generic function to train, test, and time a model."""
    results = {}
    model_name = model.__class__.__name__
    
    # --- Training ---
    mem_before_train = get_memory_usage_mb()
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    mem_after_train = get_memory_usage_mb()

    # --- Inference ---
    start_time = time.time()
    predictions = model.predict(X_test)
    inference_time = time.time() - start_time
    
    accuracy = accuracy_score(y_test, predictions)

    results = {
        'library': 'scikit-learn',
        'model': model_name,
        'train_time_sec': round(train_time, 4),
        'inference_time_sec': round(inference_time, 4),
        'accuracy': round(accuracy, 4),
        'memory_usage_mb': round(mem_after_train - mem_before_train, 4)
    }
    return results

def main():
    parser = argparse.ArgumentParser(description="Scikit-Learn Benchmark")
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset CSV file.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output results file.')
    args = parser.parse_args()
    
    print(f"Loading data from {args.dataset}...")
    df = pd.read_csv(args.dataset)
    # Assuming the last column is the target and needs to be encoded
    X = df.iloc[:, :-1]
    y = pd.factorize(df.iloc[:, -1])[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models_to_test = [
        RandomForestClassifier(n_estimators=100, random_state=42),
        SVC(gamma='auto', random_state=42),
        MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
    ]
    
    all_results = []
    for model in models_to_test:
        print(f"Benching {model.__class__.__name__}...")
        result = run_benchmark(model, X_train, y_train, X_test, y_test)
        all_results.append(result)
        print(result)

    with open(args.output, 'w') as f:
        for res in all_results:
            f.write(str(res) + '\n')
            
    print(f"\nBenchmark complete. Results saved to {args.output}")

if __name__ == '__main__':
    main()
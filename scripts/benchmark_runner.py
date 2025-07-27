#!/usr/bin/env python3
"""
Rust vs Python ML Benchmark Runner
Simple, clean benchmark for comparing ML library performance across datasets.
"""

import json
import time
import psutil
import argparse
import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class MLBenchmark:
    def __init__(self, dataset_path, output_dir="results"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load dataset info
        self.dataset_info = self._get_dataset_info()
        
        # Standardize algorithm names
        self.algorithm_map = {
            # SVM variations
            'SVC': 'SVM',
            'SVM (One-vs-Rest)': 'SVM',
            'SVM (Binary: Setosa vs Others)': 'SVM (Binary)',
            'MultiClassSVC': 'SVM',
            'LinearSVC': 'SVM (Linear)',
            'SVM (Gaussian/RBF)': 'SVM (RBF)',
            
            # Tree variations
            'DecisionTreeClassifier': 'Decision Tree',
            'Decision Tree': 'Decision Tree',
            'RandomForestClassifier': 'Random Forest',
            'Random Forest': 'Random Forest',
            
            # Neural networks
            'MLPClassifier': 'Neural Network',
            'MLP': 'Neural Network',
        }
        
        # Algorithm categories
        self.categories = {
            'SVM': 'Support Vector Machines',
            'SVM (Binary)': 'Support Vector Machines',
            'SVM (Linear)': 'Support Vector Machines',
            'SVM (RBF)': 'Support Vector Machines',
            'Decision Tree': 'Tree-Based Models',
            'Random Forest': 'Tree-Based Models',
            'Neural Network': 'Neural Networks',
        }
    
    def _get_dataset_info(self):
        """Get dataset information."""
        df = pd.read_csv(self.dataset_path)
        return {
            'name': self.dataset_path.stem,
            'n_samples': len(df),
            'n_features': len(df.columns) - 1,  # Assuming last column is target
            'n_classes': df.iloc[:, -1].nunique(),
            'size_category': 'small' if len(df) < 1000 else 'large'
        }
    
    def standardize_algorithm_name(self, name):
        """Standardize algorithm names across libraries."""
        return self.algorithm_map.get(name, name)
    
    def get_algorithm_category(self, name):
        """Get algorithm category."""
        return self.categories.get(name, 'Other')
    
    def run_python_benchmark(self, runs=5, warmup=2):
        """Run scikit-learn benchmarks."""
        print(f"🐍 Running Python (scikit-learn) benchmarks on {self.dataset_info['name']}...")
        
        results = []
        for run in range(warmup + runs):
            start_time = time.time()
            
            # Run Python benchmark
            cmd = [
                'python3', 'scripts/sklearn_runner.py',
                '--dataset', str(self.dataset_path),
                '--output', str(self.output_dir / f'python_run_{run}.json')
            ]
            
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode == 0:
                # Load results
                with open(self.output_dir / f'python_run_{run}.json', 'r') as f:
                    run_results = json.load(f)
                
                if run >= warmup:  # Only count non-warmup runs
                    results.extend(run_results)
            
            # Clean up
            (self.output_dir / f'python_run_{run}.json').unlink(missing_ok=True)
        
        return results
    
    def run_rust_benchmarks(self, runs=5, warmup=2):
        """Run Rust benchmarks (smartcore and linfa)."""
        results = []
        
        # Determine which binary to use based on dataset
        if self.dataset_info['name'] == 'bank-additional-full':
            smartcore_binary = 'smartcore_bencher'
            linfa_binary = 'linfa_bencher'
        else:
            smartcore_binary = 'iris_bencher'
            linfa_binary = 'iris_bencher'
        
        # Run SmartCore benchmarks
        print(f"🦀 Running SmartCore benchmarks on {self.dataset_info['name']}...")
        smartcore_results = self._run_rust_library('smartcore', runs, warmup, smartcore_binary)
        results.extend(smartcore_results)
        
        # Run Linfa benchmarks
        print(f"🦀 Running Linfa benchmarks on {self.dataset_info['name']}...")
        linfa_results = self._run_rust_library('linfa', runs, warmup, linfa_binary)
        results.extend(linfa_results)
        
        return results
    
    def _run_rust_library(self, library, runs, warmup, binary_name):
        """Run benchmarks for a specific Rust library."""
        results = []
        
        # Run Rust benchmark with new parameters
        cmd = [
            'cargo', 'run', '--release', 
            '--manifest-path', f'scripts/{library}_bencher/Cargo.toml',
            '--bin', binary_name,
            '--', '--dataset', str(self.dataset_path),
            '--output', str(self.output_dir / f'{library}_results.json')
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode == 0:
            # Load results
            with open(self.output_dir / f'{library}_results.json', 'r') as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
            
            # Clean up
            (self.output_dir / f'{library}_results.json').unlink(missing_ok=True)
        
        return results
    
    def analyze_results(self, all_results):
        """Analyze and organize benchmark results."""
        if not all_results:
            print("❌ No results to analyze!")
            return None, None
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Standardize algorithm names
        df['algorithm'] = df['model'].apply(self.standardize_algorithm_name)
        df['category'] = df['algorithm'].apply(self.get_algorithm_category)
        
        # Calculate statistics
        agg_columns = {
            'train_time_sec': ['mean', 'std', 'min', 'max'],
            'inference_time_sec': ['mean', 'std', 'min', 'max'],
            'accuracy': ['mean', 'std', 'min', 'max'],
            'memory_usage_mb': ['mean', 'std', 'min', 'max']
        }
        
        # Add CPU and peak memory if available
        if 'cpu_usage_percent' in df.columns:
            agg_columns['cpu_usage_percent'] = ['mean', 'std', 'min', 'max']
        if 'peak_memory_mb' in df.columns:
            agg_columns['peak_memory_mb'] = ['mean', 'std', 'min', 'max']
        
        stats = df.groupby(['library', 'algorithm']).agg(agg_columns).round(6)
        
        # Flatten column names
        stats.columns = ['_'.join(col).strip() for col in stats.columns]
        stats = stats.reset_index()
        
        return df, stats
    
    def create_report(self, df, stats):
        """Create comprehensive benchmark report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"benchmark_report_{self.dataset_info['name']}_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# ML Benchmark Report: {self.dataset_info['name'].title()}\n\n")
            
            # Dataset info
            f.write("## Dataset Information\n\n")
            f.write(f"- **Name**: {self.dataset_info['name']}\n")
            f.write(f"- **Size**: {self.dataset_info['size_category']} ({self.dataset_info['n_samples']:,} samples)\n")
            f.write(f"- **Features**: {self.dataset_info['n_features']}\n")
            f.write(f"- **Classes**: {self.dataset_info['n_classes']}\n")
            f.write(f"- **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Results by category
            for category in sorted(df['category'].unique()):
                f.write(f"## {category}\n\n")
                
                # Get data for this category
                category_data = stats[stats['algorithm'].isin(
                    df[df['category'] == category]['algorithm'].unique()
                )]
                
                if len(category_data) > 0:
                    # Create table header
                    header = "| Library | Algorithm | Training (s) | Inference (s) | Accuracy | Memory (MB)"
                    separator = "|---------|-----------|--------------|---------------|----------|-------------"
                    
                    # Add CPU and peak memory columns if available
                    if 'cpu_usage_percent_mean' in stats.columns:
                        header += " | CPU (%)"
                        separator += "|---------"
                    if 'peak_memory_mb_mean' in stats.columns:
                        header += " | Peak Memory (MB)"
                        separator += "|------------------"
                    
                    header += " |\n"
                    separator += "|\n"
                    
                    f.write(header)
                    f.write(separator)
                    
                    for _, row in category_data.iterrows():
                        f.write(f"| {row['library']} | {row['algorithm']} | ")
                        f.write(f"{row['train_time_sec_mean']:.6f}±{row['train_time_sec_std']:.6f} | ")
                        f.write(f"{row['inference_time_sec_mean']:.6f}±{row['inference_time_sec_std']:.6f} | ")
                        f.write(f"{row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f} | ")
                        f.write(f"{row['memory_usage_mb_mean']:.2f}")
                        
                        # Add CPU usage if available
                        if 'cpu_usage_percent_mean' in stats.columns:
                            f.write(f" | {row['cpu_usage_percent_mean']:.2f}")
                        
                        # Add peak memory if available
                        if 'peak_memory_mb_mean' in stats.columns:
                            f.write(f" | {row['peak_memory_mb_mean']:.2f}")
                        
                        f.write(" |\n")
                    
                    f.write("\n")
            
            # Performance champions
            f.write("## Performance Champions\n\n")
            
            fastest_training = stats.loc[stats['train_time_sec_mean'].idxmin()]
            fastest_inference = stats.loc[stats['inference_time_sec_mean'].idxmin()]
            best_accuracy = stats.loc[stats['accuracy_mean'].idxmax()]
            most_efficient = stats.loc[stats['memory_usage_mb_mean'].idxmin()]
            
            f.write(f"- **🏃 Fastest Training**: {fastest_training['algorithm']} ({fastest_training['library']}) - {fastest_training['train_time_sec_mean']:.6f}s\n")
            f.write(f"- **⚡ Fastest Inference**: {fastest_inference['algorithm']} ({fastest_inference['library']}) - {fastest_inference['inference_time_sec_mean']:.6f}s\n")
            f.write(f"- **🎯 Best Accuracy**: {best_accuracy['algorithm']} ({best_accuracy['library']}) - {best_accuracy['accuracy_mean']:.4f}\n")
            f.write(f"- **💾 Most Memory Efficient**: {most_efficient['algorithm']} ({most_efficient['library']}) - {most_efficient['memory_usage_mb_mean']:.2f} MB\n\n")
        
        print(f"📝 Report saved to: {report_file}")
        return report_file
    
    def create_visualizations(self, df, stats):
        """Create performance visualizations."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.output_dir / f"benchmark_plots_{self.dataset_info['name']}_{timestamp}.png"
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'ML Benchmark Results: {self.dataset_info["name"].title()} Dataset', fontsize=16, fontweight='bold')
        
        # 1. Training Time Comparison
        ax1 = axes[0, 0]
        for library in df['library'].unique():
            lib_data = stats[stats['library'] == library]
            x_pos = range(len(lib_data))
            ax1.bar([x + x_pos[i] * 0.25 for i, x in enumerate(x_pos)], 
                   lib_data['train_time_sec_mean'],
                   yerr=lib_data['train_time_sec_std'],
                   capsize=5, alpha=0.8, label=library)
        
        ax1.set_xlabel('Algorithms')
        ax1.set_ylabel('Training Time (s)')
        ax1.set_title('Training Time by Library')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Inference Time Comparison
        ax2 = axes[0, 1]
        for library in df['library'].unique():
            lib_data = stats[stats['library'] == library]
            x_pos = range(len(lib_data))
            ax2.bar([x + x_pos[i] * 0.25 for i, x in enumerate(x_pos)], 
                   lib_data['inference_time_sec_mean'],
                   yerr=lib_data['inference_time_sec_std'],
                   capsize=5, alpha=0.8, label=library)
        
        ax2.set_xlabel('Algorithms')
        ax2.set_ylabel('Inference Time (s)')
        ax2.set_title('Inference Time by Library')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Accuracy Comparison
        ax3 = axes[1, 0]
        for library in df['library'].unique():
            lib_data = stats[stats['library'] == library]
            x_pos = range(len(lib_data))
            ax3.bar([x + x_pos[i] * 0.25 for i, x in enumerate(x_pos)], 
                   lib_data['accuracy_mean'],
                   yerr=lib_data['accuracy_std'],
                   capsize=5, alpha=0.8, label=library)
        
        ax3.set_xlabel('Algorithms')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Accuracy by Library')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance vs Accuracy Scatter
        ax4 = axes[1, 1]
        colors = {'scikit-learn': 'blue', 'smartcore': 'red', 'linfa': 'green'}
        
        for library in df['library'].unique():
            lib_data = stats[stats['library'] == library]
            ax4.scatter(lib_data['train_time_sec_mean'], lib_data['accuracy_mean'],
                       s=100, alpha=0.7, label=library, c=colors.get(library, 'gray'))
        
        ax4.set_xlabel('Training Time (s)')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Training Time vs Accuracy')
        ax4.set_xscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plots saved to: {plot_file}")
        return plot_file
    
    def run_benchmark(self, runs=5, warmup=2):
        """Run complete benchmark suite."""
        print(f"🚀 Starting ML Benchmark: {self.dataset_info['name']} Dataset")
        print(f"📊 Dataset: {self.dataset_info['n_samples']:,} samples, {self.dataset_info['n_features']} features")
        print(f"🔄 Runs: {runs} measurement + {warmup} warmup")
        print("-" * 60)
        
        # Run benchmarks
        python_results = self.run_python_benchmark(runs, warmup)
        rust_results = self.run_rust_benchmarks(runs, warmup)
        
        all_results = python_results + rust_results
        
        if not all_results:
            print("❌ No benchmark results obtained!")
            return None, None
        
        # Analyze results
        print("\n📈 Analyzing results...")
        df, stats = self.analyze_results(all_results)
        
        if df is not None and stats is not None:
            # Create outputs
            report_file = self.create_report(df, stats)
            plot_file = self.create_visualizations(df, stats)
            
            print(f"\n✅ Benchmark complete!")
            print(f"📝 Report: {report_file}")
            print(f"📊 Plots: {plot_file}")
        
        return df, stats

def main():
    parser = argparse.ArgumentParser(description='Rust vs Python ML Benchmark Runner')
    parser.add_argument('--dataset', required=True, help='Path to dataset CSV file')
    parser.add_argument('--output-dir', default='results', help='Output directory for results')
    parser.add_argument('--runs', type=int, default=5, help='Number of measurement runs')
    parser.add_argument('--warmup', type=int, default=2, help='Number of warmup runs')
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = MLBenchmark(args.dataset, args.output_dir)
    benchmark.run_benchmark(args.runs, args.warmup)

if __name__ == "__main__":
    main() 
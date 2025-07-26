import ast
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_results_file(file_path):
    """Safely parses a results file where each line is a JSON object."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(ast.literal_eval(line))
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Could not parse line in {file_path}: {line}\nError: {e}")
    return data

def generate_plots(df, output_dir):
    """Generates and saves bar charts for the benchmark results."""
    # Group data for plotting
    plot_data = df.pivot(index='model', columns='library', values=['Training Time (s)', 'Inference Time (s)', 'Accuracy'])
    
    # Plot Training Time (Log Scale)
    ax = plot_data['Training Time (s)'].plot(kind='bar', figsize=(10, 6), logy=True)
    plt.title('Training Time Comparison (Log Scale)')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Model')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "training_time.png")
    plt.close()

    # Plot Inference Time (Log Scale)
    ax = plot_data['Inference Time (s)'].plot(kind='bar', figsize=(10, 6), logy=True)
    plt.title('Inference Time Comparison (Log Scale)')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Model')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "inference_time.png")
    plt.close()

    # Plot Accuracy
    ax = plot_data['Accuracy'].plot(kind='bar', figsize=(10, 6))
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    ax.set_ylim([0, 1.05]) # Set y-axis from 0 to 1.05
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy.png")
    plt.close()

    print(f"\nGenerated plots in {output_dir}")

def main():
    """Main function to generate tables and plots."""
    results_dir = Path("results")
    files_to_process = [
        results_dir / "sklearn_results.txt",
        results_dir / "linfa_results.txt",
        results_dir / "smartcore_results.txt",
    ]

    all_data = []
    for file_path in files_to_process:
        if file_path.exists():
            print(f"Processing {file_path}...")
            all_data.extend(parse_results_file(file_path))
        else:
            print(f"Warning: File not found, skipping: {file_path}")

    if not all_data:
        print("No data found to generate a report. Exiting.")
        return

    # --- Use Pandas to structure and format the data ---
    df = pd.DataFrame(all_data)
    
    # Simplify model names for cleaner tables/plots
    df['model'] = df['model'].replace({
        'RandomForestClassifier': 'Random Forest',
        'DecisionTree': 'Decision Tree',
        'MLPClassifier': 'MLP',
        'SVM (One-vs-Rest)': 'SVM',
        'MultiClassSVC': 'SVM'
    })

    df = df.set_index(['library', 'model'])
    df = df[['train_time_sec', 'inference_time_sec', 'accuracy']]
    df.columns = ["Training Time (s)", "Inference Time (s)", "Accuracy"]
    
    # --- Generate Outputs ---
    
    # Generate Plots
    generate_plots(df.reset_index(), results_dir)

    # Generate and save Markdown Table
    md_table = df.to_markdown(floatfmt=".5f")
    md_output_path = results_dir / "benchmark_table.md"
    with open(md_output_path, 'w') as f:
        f.write(md_table)
    print(f"Markdown table saved to {md_output_path}")

    # Generate and save LaTeX Table
    latex_table = df.to_latex(
        float_format="%.5f",
        caption="Performance Comparison of Classical ML Libraries.",
        label="tab:classical_ml_results",
        bold_rows=True
    )
    latex_output_path = results_dir / "benchmark_table.tex"
    with open(latex_output_path, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to {latex_output_path}")


if __name__ == '__main__':
    main()
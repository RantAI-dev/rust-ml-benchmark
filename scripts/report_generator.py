import ast
from pathlib import Path
import pandas as pd

def parse_results_file(file_path):
    """Safely parses a results file where each line is a stringified dict."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    # Use ast.literal_eval for safe parsing of Python literals
                    data.append(ast.literal_eval(line))
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Could not parse line in {file_path}: {line}\nError: {e}")
    return data

def main():
    """Main function to generate the LaTeX table."""
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

    # Set a multi-level index for clear grouping by library and model
    df = df.set_index(['library', 'model'])

    # Select and rename columns for the final table
    df = df[['train_time_sec', 'inference_time_sec', 'accuracy', 'memory_usage_mb']]
    df.columns = ["Training Time (s)", "Inference Time (s)", "Accuracy", "Memory Usage (MB)"]

    # Generate the LaTeX table string
    # float_format="%.5f" formats numbers to 5 decimal places
    latex_table = df.to_latex(
        float_format="%.5f",
        caption="Performance Comparison of Classical ML Libraries on the Iris Dataset.",
        label="tab:classical_ml_results",
        bold_rows=True,
        column_format="llrrrr" # l=left-aligned text, r=right-aligned number
    )

    print("\n" + "="*20 + " Generated LaTeX Table " + "="*20 + "\n")
    print(latex_table)

    # Save the table to a file
    output_path = results_dir / "benchmark_table.tex"
    with open(output_path, 'w') as f:
        f.write(latex_table)

    print(f"\nLaTeX table saved to {output_path}")

if __name__ == '__main__':
    main()
use clap::Parser;
use csv::StringRecord;
use ndarray::{Array1, Array2};
use linfa::prelude::*;
use linfa_preprocessing::linear_scaling::LinearScaler;
use linfa_svm::Svm;
use linfa_trees::DecisionTree;
use sysinfo::{Process, System};
use std::{collections::HashMap, error::Error, fs::File, io::Write, time::Instant};

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(short, long)] dataset: String,
    #[arg(short, long)] output:  String,
}

#[derive(serde::Serialize, Debug)]
struct BenchResult {
    library:            String,
    model:              String,
    train_time_sec:     f64,
    inference_time_sec: f64,
    accuracy:           f64,
    memory_usage_mb:    f64,
}

fn get_memory_usage_mb() -> f64 {
    let mut sys = System::new_all();
    sys.refresh_processes();
    let pid = sysinfo::get_current_pid().unwrap();
    if let Some(proc_) = sys.process(pid) {
        let memory_bytes = proc_.memory();
        let memory_mb = memory_bytes as f64 / 1024.0 / 1024.0; // Convert bytes to MB
        return memory_mb;
    }
    0.0
}

// Process Iris dataset record: 4 numeric features + 1 categorical target
fn process_iris_record(
    record: &StringRecord,
    class_map: &mut HashMap<String, usize>,
) -> (Vec<f64>, usize) {
    let mut features = Vec::with_capacity(4);
    
    // All first 4 columns are numeric features
    for i in 0..4 {
        features.push(record[i].parse().unwrap_or(0.0));
    }
    
    // Last column is the class (variety)
    let class_name = record[4].to_string();
    let next_id = class_map.len();
    let class_id = *class_map.entry(class_name).or_insert(next_id);
    
    (features, class_id)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    println!("Loading & preprocessing Iris dataset from {}...", &args.dataset);
    let mut rdr = csv::ReaderBuilder::new()
        .from_path(&args.dataset)?;

    let mut raw_features = Vec::new();
    let mut raw_targets = Vec::new();
    let mut class_map = HashMap::new();

    // Skip header and process records
    for res in rdr.records().skip(1) {
        let rec = res?;
        let (features, target) = process_iris_record(&rec, &mut class_map);
        raw_features.push(features);
        raw_targets.push(target);
    }

    println!("Loaded {} samples with {} classes", raw_features.len(), class_map.len());
    for (class_name, class_id) in &class_map {
        println!("  Class {}: {}", class_id, class_name);
    }

    // Build ndarray of shape (n_samples, 4)
    let n_samples = raw_features.len();
    let n_features = raw_features[0].len();
    let flat: Vec<f64> = raw_features.into_iter().flatten().collect();
    let records_arr = Array2::from_shape_vec((n_samples, n_features), flat)?;
    let targets_arr: Array1<usize> = Array1::from(raw_targets);

    // Wrap in Dataset and split 80/20
    let dataset = Dataset::new(records_arr, targets_arr);
    let (train_data, test_data) = dataset.split_with_ratio(0.8);

    // Scale features to zero mean, unit variance
    println!("Scaling features...");
    let scaler = LinearScaler::standard().fit(&train_data)?;
    let scaled_train_data = scaler.transform(train_data.clone());
    let scaled_test_data = scaler.transform(test_data.clone());

    let mut all_results = Vec::new();

    // --- Decision Tree ---
    println!("Benching DecisionTree...");
    let mem0 = get_memory_usage_mb();
    let t0 = Instant::now();
    let dt_model = DecisionTree::params()
        .max_depth(Some(10))
        .fit(&train_data)?;  // Use unscaled data for Decision Tree
    let train_dt = t0.elapsed().as_secs_f64();
    let mem1 = get_memory_usage_mb();

    let t1 = Instant::now();
    let dt_pred = dt_model.predict(test_data.records());  // Use unscaled data
    let inf_dt = t1.elapsed().as_secs_f64();
    let acc_dt: f64 = dt_pred
        .confusion_matrix(test_data.targets())?  // Use unscaled data
        .accuracy()
        .into();

    all_results.push(BenchResult {
        library: "linfa".into(),
        model: "DecisionTree".into(),
        train_time_sec: train_dt,
        inference_time_sec: inf_dt,
        accuracy: acc_dt,
        memory_usage_mb: mem1 - mem0,
    });

    // --- SVM (Binary Classification) ---
    // For multi-class Iris, we'll convert to binary (class 0 vs others)
    // This is similar to the approach used in the existing main.rs
    println!("Benching SVM (Binary Classification: Setosa vs Others)...");
    let mem0 = get_memory_usage_mb();
    let t0 = Instant::now();

    // Binary classification: convert multiclass to binary (class 0 = Setosa vs others)
    let binary_targets = scaled_train_data.targets().mapv(|x| x == 0);
    let binary_train_data = Dataset::new(scaled_train_data.records().clone(), binary_targets);
    
    let svm_model = Svm::<f64, bool>::params()
        .gaussian_kernel(0.5)
        .pos_neg_weights(1.0, 1.0)
        .check()?
        .fit(&binary_train_data)?;
    let train_svm = t0.elapsed().as_secs_f64();
    let mem1 = get_memory_usage_mb();

    let t1 = Instant::now();
    let svm_pred = svm_model.predict(scaled_test_data.records());
    let inf_svm = t1.elapsed().as_secs_f64();
    
    // Convert binary predictions back to accuracy calculation
    let binary_test_targets = scaled_test_data.targets().mapv(|x| x == 0);
    
    let correct = svm_pred.iter()
        .zip(binary_test_targets.iter())
        .filter(|(pred, actual)| pred == actual)
        .count();
    let acc_svm = correct as f64 / svm_pred.len() as f64;

    all_results.push(BenchResult {
        library: "linfa".into(),
        model: "SVM (Binary: Setosa vs Others)".into(),
        train_time_sec: train_svm,
        inference_time_sec: inf_svm,
        accuracy: acc_svm,
        memory_usage_mb: mem1 - mem0,
    });

    // Write JSON lines
    let mut file = File::create(&args.output)?;
    for r in &all_results {
        println!("{:?}", r);
        writeln!(file, "{}", serde_json::to_string(r)?)?;
    }
    println!("\nBenchmark complete. Results saved to {}", &args.output);
    Ok(())
}
use clap::Parser;
use csv::StringRecord;
use ndarray::{Array1, Array2};
use linfa::prelude::*;
use linfa_preprocessing::linear_scaling::LinearScaler;
use linfa_svm::Svm;
use linfa_trees::DecisionTree;
use serde::{Serialize, Deserialize};

// Professional resource monitoring with sysinfo
use sysinfo::{Process, System};

use std::{collections::HashMap, error::Error, fs::File, io::Write, time::Instant};

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(short, long)] dataset: String,
    #[arg(short, long)] output: String,
    #[arg(long, default_value = "5")] runs: usize,
    #[arg(long, default_value = "2")] warmup: usize,
}

#[derive(Serialize, Debug)]
struct BenchResult {
    library: String,
    model: String,
    train_time_sec: f64,
    inference_time_sec: f64,
    accuracy: f64,
    memory_usage_mb: f64,
    cpu_usage_percent: f64,
    peak_memory_mb: f64,
}

fn get_process_stats(sys: &mut System) -> (f32, u64) {
    sys.refresh_processes();
    let pid = sysinfo::get_current_pid().unwrap();
    if let Some(proc_) = sys.process(pid) {
        let cpu = proc_.cpu_usage(); // percent of one core
        let mem = proc_.memory();     // bytes of RSS
        return (cpu, mem);
    }
    (0.0, 0)
}

/// Turn one CSV line into 20 numeric features + a bool target.
fn process_bank_record(
    record: &StringRecord,
    cat_maps: &mut [HashMap<String, usize>],
) -> (Vec<f64>, bool) {
    let mut features = Vec::with_capacity(20);

    // numeric columns
    for &idx in &[0, 10, 11, 12, 13, 15, 16, 17, 18, 19] {
        features.push(record[idx].parse().unwrap_or(0.0));
    }

    // categorical columns → precompute map.len() *before* entry() to avoid double borrow
    for (i, &idx) in [1,2,3,4,5,6,7,8,9,14].iter().enumerate() {
        let v = record[idx].to_string();
        let map = &mut cat_maps[i];
        let next_id = map.len();                  // immutable borrow happens here
        let id = *map.entry(v).or_insert(next_id); // then mutable borrow
        features.push(id as f64);
    }

    // compare two &str's by using record.get(...)
    let target = record.get(20).unwrap() == "yes";
    (features, target)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    
    // Initialize system monitoring
    let mut sys = System::new_all();
    
    println!("Loading & preprocessing {}...", &args.dataset);
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b';')
        .from_path(&args.dataset)?;

    let mut raw_features = Vec::new();
    let mut raw_targets = Vec::new();
    let mut cat_maps = vec![HashMap::new(); 10];

    // skip header
    for rec in rdr.records().skip(1) {
        let rec = rec?;
        let (feat, tgt) = process_bank_record(&rec, &mut cat_maps);
        raw_features.push(feat);
        raw_targets.push(tgt);
    }

    // build an ndarray of shape (n_samples, 20)
    let n_samples = raw_features.len();
    let n_features = raw_features[0].len();
    let flat: Vec<f64> = raw_features.into_iter().flatten().collect();
    let records_arr = Array2::from_shape_vec((n_samples, n_features), flat)?;

    // bool → usize (0/1), then into Array1<usize>
    let numeric_targets: Vec<usize> = raw_targets
        .into_iter()
        .map(|b| if b { 1 } else { 0 })
        .collect();
    let targets_arr: Array1<usize> = Array1::from(numeric_targets);

    // wrap in a Dataset and split 80/20
    let dataset = Dataset::new(records_arr, targets_arr);
    let (train_data, test_data) = dataset.split_with_ratio(0.8);

    // scale features to zero mean, unit variance
    println!("Scaling features...");
    let scaler = LinearScaler::standard().fit(&train_data)?;
    let scaled_train_data = scaler.transform(train_data.clone());
    let scaled_test_data = scaler.transform(test_data.clone());

    let mut all_results = Vec::new();

    // Benchmark Decision Tree
    println!("Benchmarking DecisionTree...");
    let mut train_times = Vec::new();
    let mut inference_times = Vec::new();
    let mut memory_usages = Vec::new();
    let mut cpu_usages = Vec::new();
    let mut accuracies = Vec::new();
    
    // Warmup runs
    for _ in 0..args.warmup {
        let _dt_model = DecisionTree::params()
            .max_depth(Some(10))
            .fit(&train_data)?;
    }
    
    // Measurement runs
    for _ in 0..args.runs {
        // Get baseline stats
        let (_cpu_before, mem_before) = get_process_stats(&mut sys);
        
        // Training
        let train_start = Instant::now();
        let dt_model = DecisionTree::params()
            .max_depth(Some(10))
            .fit(&train_data)?;
        let train_time = train_start.elapsed().as_secs_f64();
        
        // Get peak stats during training
        let (cpu_peak, mem_peak) = get_process_stats(&mut sys);
        
        // Inference
        let inference_start = Instant::now();
        let dt_pred = dt_model.predict(test_data.records());
        let inference_time = inference_start.elapsed().as_secs_f64();
        
        // Calculate accuracy
        let confusion_matrix = dt_pred.confusion_matrix(test_data.targets())?;
        let acc: f64 = confusion_matrix.accuracy().into();
        
        // Store results
        train_times.push(train_time);
        inference_times.push(inference_time);
        memory_usages.push((mem_peak.saturating_sub(mem_before)) as f64 / 1024.0 / 1024.0); // Convert bytes to MB
        cpu_usages.push(cpu_peak as f64);
        accuracies.push(acc);
    }
    
    // Calculate statistics
    let train_time_mean = train_times.iter().sum::<f64>() / args.runs as f64;
    let inference_time_mean = inference_times.iter().sum::<f64>() / args.runs as f64;
    let memory_mean = memory_usages.iter().sum::<f64>() / args.runs as f64;
    let cpu_mean = cpu_usages.iter().sum::<f64>() / args.runs as f64;
    let accuracy_mean = accuracies.iter().sum::<f64>() / args.runs as f64;
    
    all_results.push(BenchResult {
        library: "linfa".into(),
        model: "DecisionTree".into(),
        train_time_sec: train_time_mean,
        inference_time_sec: inference_time_mean,
        accuracy: accuracy_mean,
        memory_usage_mb: memory_mean,
        cpu_usage_percent: cpu_mean,
        peak_memory_mb: memory_mean,
    });

    // Benchmark SVM (Gaussian/RBF)
    println!("Benchmarking SVM (Gaussian/RBF)...");
    let mut train_times = Vec::new();
    let mut inference_times = Vec::new();
    let mut memory_usages = Vec::new();
    let mut cpu_usages = Vec::new();
    let mut accuracies = Vec::new();
    
    // Warmup runs
    for _ in 0..args.warmup {
        // Binary classification for SVM works better - convert multiclass to binary
        let binary_targets = scaled_train_data.targets().mapv(|x| x > 0);
        let binary_train_data = Dataset::new(scaled_train_data.records().clone(), binary_targets);
        
        let _svm_model = Svm::<f64, bool>::params()
            .gaussian_kernel(0.5)
            .pos_neg_weights(1.0, 1.0)
            .check()?
            .fit(&binary_train_data)?;
    }
    
    // Measurement runs
    for _ in 0..args.runs {
        // Get baseline stats
        let (_cpu_before, mem_before) = get_process_stats(&mut sys);
        
        // Training
        let train_start = Instant::now();
        
        // Binary classification for SVM works better - convert multiclass to binary
        let binary_targets = scaled_train_data.targets().mapv(|x| x > 0);
        let binary_train_data = Dataset::new(scaled_train_data.records().clone(), binary_targets);
        
        let svm_model = Svm::<f64, bool>::params()
            .gaussian_kernel(0.5)
            .pos_neg_weights(1.0, 1.0)
            .check()?
            .fit(&binary_train_data)?;
        let train_time = train_start.elapsed().as_secs_f64();
        
        // Get peak stats during training
        let (cpu_peak, mem_peak) = get_process_stats(&mut sys);
        
        // Inference
        let inference_start = Instant::now();
        let svm_pred = svm_model.predict(scaled_test_data.records());
        let inference_time = inference_start.elapsed().as_secs_f64();
        
        // Convert binary predictions back to multiclass format for accuracy calculation
        let svm_pred_multiclass: Array1<usize> = svm_pred.mapv(|x| if x { 1 } else { 0 });
        let binary_test_targets = scaled_test_data.targets().mapv(|x| if x > 0 { 1 } else { 0 });
        
        let correct = svm_pred_multiclass.iter()
            .zip(binary_test_targets.iter())
            .filter(|(pred, actual)| pred == actual)
            .count();
        let acc = correct as f64 / svm_pred_multiclass.len() as f64;
        
        // Store results
        train_times.push(train_time);
        inference_times.push(inference_time);
        memory_usages.push((mem_peak.saturating_sub(mem_before)) as f64 / 1024.0 / 1024.0); // Convert bytes to MB
        cpu_usages.push(cpu_peak as f64);
        accuracies.push(acc);
    }
    
    // Calculate statistics
    let train_time_mean = train_times.iter().sum::<f64>() / args.runs as f64;
    let inference_time_mean = inference_times.iter().sum::<f64>() / args.runs as f64;
    let memory_mean = memory_usages.iter().sum::<f64>() / args.runs as f64;
    let cpu_mean = cpu_usages.iter().sum::<f64>() / args.runs as f64;
    let accuracy_mean = accuracies.iter().sum::<f64>() / args.runs as f64;
    
    all_results.push(BenchResult {
        library: "linfa".into(),
        model: "SVM (Gaussian/RBF)".into(),
        train_time_sec: train_time_mean,
        inference_time_sec: inference_time_mean,
        accuracy: accuracy_mean,
        memory_usage_mb: memory_mean,
        cpu_usage_percent: cpu_mean,
        peak_memory_mb: memory_mean,
    });

    // write JSON‐lines
    let mut file = File::create(&args.output)?;
    for r in &all_results {
        println!("{:?}", r);
        writeln!(file, "{}", serde_json::to_string(r)?)?;
    }
    println!("\nBenchmark complete. Results saved to {}", &args.output);
    Ok(())
}

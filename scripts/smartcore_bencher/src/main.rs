use clap::Parser;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use csv::StringRecord;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::svm::{Kernels, svc::{SVC, SVCParameters}};
use smartcore::metrics::accuracy;
use serde::{Serialize, Deserialize};

// Professional resource monitoring with sysinfo
use sysinfo::{Process, System};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
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

// This helper function processes bank dataset records
fn process_bank_record(
    record: &StringRecord,
    cat_maps: &mut [HashMap<String,usize>],
) -> (Vec<f64>, bool) {
    let mut features = Vec::with_capacity(20);
    // numeric cols
    for &idx in &[0,10,11,12,13,15,16,17,18,19] {
        features.push(record[idx].parse().unwrap_or(0.0));
    }
    // categorical cols
    for (i, &idx) in [1,2,3,4,5,6,7,8,9,14].iter().enumerate() {
        let v = record[idx].to_string();
        let map = &mut cat_maps[i];
        let next_id = map.len();
        let id = *map.entry(v).or_insert(next_id);
        features.push(id as f64);
    }
    let target = record.get(20).unwrap() == "yes";
    (features, target)
}

// Manual implementation of standard scaling
fn apply_standard_scaling(train_x_vec: &[Vec<f64>], test_x_vec: &[Vec<f64>]) -> (DenseMatrix<f64>, DenseMatrix<f64>) {
    let n_train = train_x_vec.len();
    let n_test = test_x_vec.len();
    let n_features = train_x_vec[0].len();
    
    // Calculate mean and std from training data
    let mut means = vec![0.0; n_features];
    let mut stds = vec![0.0; n_features];
    
    // Calculate means
    for j in 0..n_features {
        let mut sum = 0.0;
        for i in 0..n_train {
            sum += train_x_vec[i][j];
        }
        means[j] = sum / n_train as f64;
    }
    
    // Calculate standard deviations
    for j in 0..n_features {
        let mut var_sum = 0.0;
        for i in 0..n_train {
            let diff = train_x_vec[i][j] - means[j];
            var_sum += diff * diff;
        }
        stds[j] = (var_sum / n_train as f64).sqrt();
        if stds[j] < 1e-8 { stds[j] = 1.0; } // Avoid division by zero
    }
    
    // Scale training data
    let mut scaled_train = vec![vec![0.0; n_features]; n_train];
    for i in 0..n_train {
        for j in 0..n_features {
            scaled_train[i][j] = (train_x_vec[i][j] - means[j]) / stds[j];
        }
    }
    
    // Scale test data using training statistics
    let mut scaled_test = vec![vec![0.0; n_features]; n_test];
    for i in 0..n_test {
        for j in 0..n_features {
            scaled_test[i][j] = (test_x_vec[i][j] - means[j]) / stds[j];
        }
    }
    
    (DenseMatrix::from_2d_vec(&scaled_train).unwrap(), DenseMatrix::from_2d_vec(&scaled_test).unwrap())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    
    // Initialize system monitoring
    let mut sys = System::new_all();
    
    println!("Loading & preprocessing {}", &args.dataset);
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b';')
        .from_path(&args.dataset)?;

    let mut records: Vec<Vec<f64>> = Vec::new();
    let mut targets_bool: Vec<bool> = Vec::new();
    let mut cat_maps = vec![HashMap::new(); 10];

    for res in rdr.records().skip(1) {
        let rec = res?;
        let (feat, tgt) = process_bank_record(&rec, &mut cat_maps);
        records.push(feat);
        targets_bool.push(tgt);
    }

    let n = records.len();
    let n_train = (n as f64 * 0.8) as usize;

    let train_x_vec: Vec<Vec<f64>> = records[..n_train].to_vec();
    let test_x_vec: Vec<Vec<f64>> = records[n_train..].to_vec();

    let train_y_bool = &targets_bool[..n_train];
    let test_y_bool = &targets_bool[n_train..];

    let train_x: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&train_x_vec)?;
    let test_x: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&test_x_vec)?;

    // Apply standard scaling
    println!("Scaling features...");
    let (scaled_train_x, scaled_test_x) = apply_standard_scaling(&train_x_vec, &test_x_vec);

    // Convert boolean targets to usize for smartcore models
    let train_y: Vec<usize> = train_y_bool.iter().map(|&b| if b {1} else {0}).collect();
    let test_y: Vec<usize> = test_y_bool.iter().map(|&b| if b {1} else {0}).collect();

    let mut all_results = Vec::new();

    // Benchmark Random Forest
    println!("Benchmarking RandomForestClassifier...");
    let mut train_times = Vec::new();
    let mut inference_times = Vec::new();
    let mut memory_usages = Vec::new();
    let mut cpu_usages = Vec::new();
    let mut accuracies = Vec::new();
    
    // Warmup runs
    for _ in 0..args.warmup {
        let _rfc = RandomForestClassifier::fit(
            &train_x,
            &train_y,
            RandomForestClassifierParameters::default(),
        )?;
    }
    
    // Measurement runs
    for _ in 0..args.runs {
        // Get baseline stats
        let (cpu_before, mem_before) = get_process_stats(&mut sys);
        
        // Training
        let train_start = Instant::now();
        let rfc = RandomForestClassifier::fit(
            &train_x,
            &train_y,
            RandomForestClassifierParameters::default(),
        )?;
        let train_time = train_start.elapsed().as_secs_f64();
        
        // Get peak stats during training
        let (cpu_peak, mem_peak) = get_process_stats(&mut sys);
        
        // Inference
        let inference_start = Instant::now();
        let rfc_pred: Vec<usize> = rfc.predict(&test_x)?;
        let inference_time = inference_start.elapsed().as_secs_f64();
        
        // Calculate accuracy
        let acc = accuracy(&test_y, &rfc_pred);
        
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
        library: "smartcore".into(),
        model: "RandomForestClassifier".into(),
        train_time_sec: train_time_mean,
        inference_time_sec: inference_time_mean,
        accuracy: accuracy_mean,
        memory_usage_mb: memory_mean,
        cpu_usage_percent: cpu_mean,
        peak_memory_mb: memory_mean,
    });

    // Benchmark SVM (Gaussian/RBF)
    println!("Benchmarking SVC (Gaussian/RBF)...");
    let mut train_times = Vec::new();
    let mut inference_times = Vec::new();
    let mut memory_usages = Vec::new();
    let mut cpu_usages = Vec::new();
    let mut accuracies = Vec::new();
    
    // Warmup runs
    for _ in 0..args.warmup {
        let svc_params = SVCParameters::default().with_kernel(Kernels::rbf().with_gamma(0.5));
        let _svc = SVC::fit(&scaled_train_x, &train_y, &svc_params)?;
    }
    
    // Measurement runs
    for _ in 0..args.runs {
        // Get baseline stats
        let (cpu_before, mem_before) = get_process_stats(&mut sys);
        
        // Training
        let train_start = Instant::now();
        let svc_params = SVCParameters::default().with_kernel(Kernels::rbf().with_gamma(0.5));
        let svc = SVC::fit(&scaled_train_x, &train_y, &svc_params)?;
        let train_time = train_start.elapsed().as_secs_f64();
        
        // Get peak stats during training
        let (cpu_peak, mem_peak) = get_process_stats(&mut sys);
        
        // Inference
        let inference_start = Instant::now();
        let svc_pred: Vec<usize> = svc.predict(&scaled_test_x)?.into_iter().map(|x| x as usize).collect();
        let inference_time = inference_start.elapsed().as_secs_f64();
        
        // Calculate accuracy
        let acc = accuracy(&test_y, &svc_pred);
        
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
        library: "smartcore".into(),
        model: "SVC (Gaussian/RBF)".into(),
        train_time_sec: train_time_mean,
        inference_time_sec: inference_time_mean,
        accuracy: accuracy_mean,
        memory_usage_mb: memory_mean,
        cpu_usage_percent: cpu_mean,
        peak_memory_mb: memory_mean,
    });

    // Write results
    let mut file = File::create(&args.output)?;
    for r in &all_results {
        println!("{:?}", r);
        writeln!(file, "{}", serde_json::to_string(r)?)?;
    }
    println!("\nBenchmark complete. Results saved to {}", &args.output);

    Ok(())
}

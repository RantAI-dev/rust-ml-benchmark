use clap::Parser;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use psutil::process::Process;

// SmartCore imports
use smartcore::linalg::basic::matrix::DenseMatrix;            // DenseMatrix::from_2d_vec → Result<DenseMatrix,_> 
use smartcore::linalg::basic::arrays::Array;                 // brings `.shape()` into scope 
use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::svm::{Kernels, svc::{MultiClassSVC, SVCParameters}};
use smartcore::metrics::accuracy;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    dataset: String,
    #[arg(short, long)]
    output: String,
}

#[derive(serde::Serialize, Debug)]
struct BenchResult {
    library: String,
    model: String,
    train_time_sec: f64,
    inference_time_sec: f64,
    accuracy: f64,
    memory_usage_mb: f64,
}

fn get_memory_usage_mb() -> f64 {
    let proc = Process::current().unwrap();
    proc.memory_info().unwrap().rss() as f64 / (1024.0 * 1024.0)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    println!("Loading data from {}...", &args.dataset);
    let mut reader = csv::Reader::from_path(&args.dataset)?;
    let mut records: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<String>  = Vec::new();

    for result in reader.records() {
        let record = result?;
        let features: Vec<f64> = record
            .iter()
            .take(4)
            .map(|v| v.parse().unwrap())
            .collect();
        let label = record.get(4).unwrap().to_string();
        records.push(features);
        targets.push(label);
    }

    // Map string labels → usize so they satisfy `Ord`
    let mut class_map = HashMap::new();
    let mut numeric_targets: Vec<usize> = Vec::new();
    for t in &targets {
        let next = class_map.len();
        let id = *class_map.entry(t.clone()).or_insert(next);
        numeric_targets.push(id);
    }

    // Build full DenseMatrix (unwrap the Result!)
    let full_x = DenseMatrix::from_2d_vec(&records)?;           // 
    let (n_samples, _) = full_x.shape();                       // `.shape()` now in scope 

    // 80/20 split
    let n_train = (n_samples as f64 * 0.8) as usize;
    let mut train_x_vec = Vec::with_capacity(n_train);
    let mut train_y_vec: Vec<usize> = Vec::with_capacity(n_train);
    let mut test_x_vec  = Vec::with_capacity(n_samples - n_train);
    let mut test_y_vec:  Vec<usize> = Vec::with_capacity(n_samples - n_train);

    for i in 0..n_samples {
        if i < n_train {
            train_x_vec.push(records[i].clone());
            train_y_vec.push(numeric_targets[i]);
        } else {
            test_x_vec.push(records[i].clone());
            test_y_vec.push(numeric_targets[i]);
        }
    }

    // Convert splits into DenseMatrix (unwrap Results)
    let train_x = DenseMatrix::from_2d_vec(&train_x_vec)?;     // 
    let test_x  = DenseMatrix::from_2d_vec(&test_x_vec)?;      // 
    let train_y = train_y_vec;
    let test_y  = test_y_vec;

    let mut all_results = Vec::new();

    // --- Random Forest ---
    println!("Benching RandomForestClassifier...");
    let mem0 = get_memory_usage_mb();
    let t0 = Instant::now();
    let rfc = RandomForestClassifier::fit(
        &train_x,
        &train_y,
        RandomForestClassifierParameters::default(),
    )?;
    let train_time = t0.elapsed().as_secs_f64();
    let mem1 = get_memory_usage_mb();

    let t1 = Instant::now();
    let rfc_pred: Vec<usize> = rfc.predict(&test_x)?;          // returns Vec<usize>
    let inf_time = t1.elapsed().as_secs_f64();

    all_results.push(BenchResult {
        library: "smartcore".into(),
        model:   "RandomForestClassifier".into(),
        train_time_sec:     train_time,
        inference_time_sec: inf_time,
        accuracy:           accuracy(&test_y, &rfc_pred),
        memory_usage_mb:    mem1 - mem0,
    });

    // --- Multi-class SVM ---
    println!("Benching MultiClassSVC...");
    let mem0 = get_memory_usage_mb();
    let t0 = Instant::now();

    let svc_params = SVCParameters::default()
        .with_c(1.0)
        .with_kernel(Kernels::rbf().with_gamma(0.5));
    let msvc = MultiClassSVC::fit(&train_x, &train_y, &svc_params)?;
    let train_time = t0.elapsed().as_secs_f64();
    let mem1 = get_memory_usage_mb();

    let t1 = Instant::now();
    let svc_pred: Vec<usize> = msvc.predict(&test_x)?.into_iter().map(|x| x as usize).collect();
    let inf_time = t1.elapsed().as_secs_f64();

    all_results.push(BenchResult {
        library: "smartcore".into(),
        model:   "MultiClassSVC".into(),
        train_time_sec:     train_time,
        inference_time_sec: inf_time,
        accuracy:           accuracy(&test_y, &svc_pred),
        memory_usage_mb:    mem1 - mem0,
    });

    // --- Write results ---
    let mut file = File::create(&args.output)?;
    for r in &all_results {
        println!("{:?}", r);
        // Serialize the result to a JSON string and write it
        let json_string = serde_json::to_string(r)?;
        writeln!(file, "{}", json_string)?;
    }
    println!("\nBenchmark complete. Results saved to {}", args.output);
    Ok(())
}

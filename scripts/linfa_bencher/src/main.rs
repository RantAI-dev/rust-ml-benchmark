use clap::Parser;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use ndarray::{Array1, Array2};

use linfa::prelude::*;
use linfa::dataset::Pr;
use linfa::composing::platt_scaling::PlattParams;
use linfa_svm::Svm;
use linfa_trees::DecisionTree;
use psutil::process::Process;

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

    // ** Annotate your Vecs so Rust knows the element types upfront **
    let mut records: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<String>  = Vec::new();

    for result in reader.records() {
        let record = result?;

        // ** Annotate `features` as Vec<f64> so .collect() can infer correctly **
        let features: Vec<f64> = record
            .iter()
            .take(4)
            .map(|v| v.parse().unwrap())
            .collect();

        let label = record.get(4).unwrap().to_string();

        records.push(features);
        targets.push(label);
    }

    // Map string labels → usize
    let mut class_map = HashMap::new();
    let mut numeric_targets = Vec::new();
    for t in &targets {
        let next = class_map.len();
        let id = *class_map.entry(t.clone()).or_insert(next);
        numeric_targets.push(id);
    }

    // Build ndarray data structures
    let n_features = records[0].len();
    let flat: Vec<f64> =
        records.clone().into_iter().flatten().collect();
    let records_arr =
        Array2::from_shape_vec((flat.len() / n_features, n_features), flat)?;
    let targets_arr = Array1::from(numeric_targets);

    let dataset = Dataset::new(records_arr.clone(), targets_arr.clone());
    let (train_data, test_data) = dataset.split_with_ratio(0.8);

    let mut all_results = Vec::new();

    // --- Decision Tree (unchanged) ---
    println!("Benching DecisionTree...");
    let dt_params = DecisionTree::params().max_depth(Some(10));
    let mem0 = get_memory_usage_mb();
    let t0 = Instant::now();
    let dt_model = dt_params.fit(&train_data)?;
    let dt_train = t0.elapsed().as_secs_f64();
    let mem1 = get_memory_usage_mb();
    let t1 = Instant::now();
    let dt_pred = dt_model.predict(test_data.records());
    let dt_inf = t1.elapsed().as_secs_f64();
    let dt_acc: f64 =
        dt_pred.confusion_matrix(test_data.targets())?.accuracy().into();
    all_results.push(BenchResult {
        library: "linfa".into(),
        model: "DecisionTree".into(),
        train_time_sec: dt_train,
        inference_time_sec: dt_inf,
        accuracy: dt_acc,
        memory_usage_mb: mem1 - mem0,
    });

    // --- SVM (One‑vs‑Rest Multi‑class) ---
    println!("Benching SVM (One-vs-Rest)...");
    let (train_rec, train_tgt) = (train_data.records(), train_data.targets());
    let mem0 = get_memory_usage_mb();
    let t0 = Instant::now();

    let n_classes = train_tgt.iter().max().unwrap() + 1; // no `*` deref error
    let mut models = Vec::with_capacity(n_classes);
    for cid in 0..n_classes {
        let binary_tg = train_tgt.mapv(|t| t == cid); // drop the stray `*`
        let ds = Dataset::new(train_rec.clone(), binary_tg);

        let model = Svm::<f64, Pr>::params()
            .gaussian_kernel(0.2)                          // RBF γ=0.2
            .pos_neg_weights(1.0, 1.0)                     // C=1.0
            .with_platt_params(PlattParams::default())     // probability
            .check()?
            .fit(&ds)?;
        models.push(model);
    }

    let svm_train = t0.elapsed().as_secs_f64();
    let mem1 = get_memory_usage_mb();

    let (test_rec, test_tgt) = (test_data.records(), test_data.targets());
    let t1 = Instant::now();
    let mut correct = 0;
    for (i, sample) in test_rec.outer_iter().enumerate() {
        let mut best = 0;
        let mut best_p = std::f32::NEG_INFINITY;
        for (cid, m) in models.iter().enumerate() {
            let p: Pr = m.predict(sample.view());
            let p_f32: f32 = *p;                         // Deref to inner f32
            if p_f32 > best_p {
                best_p = p_f32;
                best   = cid;
            }
        }
        if best == test_tgt[i] {
            correct += 1;
        }
    }
    let svm_inf = t1.elapsed().as_secs_f64();
    let svm_acc = correct as f64 / test_rec.nrows() as f64;

    all_results.push(BenchResult {
        library: "linfa".into(),
        model: "SVM (One-vs-Rest)".into(),
        train_time_sec: svm_train,
        inference_time_sec: svm_inf,
        accuracy: svm_acc,
        memory_usage_mb: mem1 - mem0,
    });

    // --- Write results ---
    let mut file = File::create(&args.output)?;
    for res in &all_results {
        println!("{:?}", res);
        // Serialize the result to a JSON string and write it
        let json_string = serde_json::to_string(res)?;
        writeln!(file, "{}", json_string)?;
    }
    println!("\nBenchmark complete. Results saved to {}", &args.output);

    Ok(())
}

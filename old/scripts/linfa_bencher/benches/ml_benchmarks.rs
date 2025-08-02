use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array1, Array2};
use linfa::prelude::*;
use linfa_preprocessing::linear_scaling::LinearScaler;
use linfa_svm::Svm;
use linfa_trees::DecisionTree;
use std::collections::HashMap;
use csv::StringRecord;

// Helper functions from main.rs
fn process_bank_record(
    record: &StringRecord,
    cat_maps: &mut [HashMap<String, usize>],
) -> (Vec<f64>, bool) {
    let mut features = Vec::with_capacity(20);

    // numeric columns
    for &idx in &[0, 10, 11, 12, 13, 15, 16, 17, 18, 19] {
        features.push(record[idx].parse().unwrap_or(0.0));
    }

    // categorical columns
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

fn load_bank_dataset() -> (Dataset<f64, usize>, Dataset<f64, usize>) {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b';')
        .from_path("../../data/bank-additional/bank-additional-full.csv")
        .unwrap();

    let mut raw_features = Vec::new();
    let mut raw_targets = Vec::new();
    let mut cat_maps = vec![HashMap::new(); 10];

    // skip header
    for rec in rdr.records().skip(1) {
        let rec = rec.unwrap();
        let (feat, tgt) = process_bank_record(&rec, &mut cat_maps);
        raw_features.push(feat);
        raw_targets.push(tgt);
    }

    // build an ndarray of shape (n_samples, 20)
    let n_samples = raw_features.len();
    let n_features = raw_features[0].len();
    let flat: Vec<f64> = raw_features.into_iter().flatten().collect();
    let records_arr = Array2::from_shape_vec((n_samples, n_features), flat).unwrap();

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
    let scaler = LinearScaler::standard().fit(&train_data).unwrap();
    let scaled_train_data = scaler.transform(train_data.clone());
    let scaled_test_data = scaler.transform(test_data.clone());

    (scaled_train_data, scaled_test_data)
}

fn bench_decision_tree(c: &mut Criterion) {
    let (train_data, test_data) = load_bank_dataset();
    
    c.bench_function("linfa_decision_tree_train", |b| {
        b.iter(|| {
            DecisionTree::params()
                .max_depth(Some(10))
                .fit(&train_data)
                .unwrap()
        })
    });

    let model = DecisionTree::params()
        .max_depth(Some(10))
        .fit(&train_data)
        .unwrap();

    c.bench_function("linfa_decision_tree_inference", |b| {
        b.iter(|| {
            model.predict(test_data.records())
        })
    });
}

fn bench_svm_rbf(c: &mut Criterion) {
    let (train_data, test_data) = load_bank_dataset();
    
    c.bench_function("linfa_svm_rbf_train", |b| {
        b.iter(|| {
            // Binary classification for SVM works better - convert multiclass to binary
            let binary_targets = train_data.targets().mapv(|x| x > 0);
            let binary_train_data = Dataset::new(train_data.records().clone(), binary_targets);
            
            Svm::<f64, bool>::params()
                .gaussian_kernel(0.5)
                .pos_neg_weights(1.0, 1.0)
                .check()
                .unwrap()
                .fit(&binary_train_data)
                .unwrap()
        })
    });

    // Binary classification for SVM works better - convert multiclass to binary
    let binary_targets = train_data.targets().mapv(|x| x > 0);
    let binary_train_data = Dataset::new(train_data.records().clone(), binary_targets);
    
    let model = Svm::<f64, bool>::params()
        .gaussian_kernel(0.5)
        .pos_neg_weights(1.0, 1.0)
        .check()
        .unwrap()
        .fit(&binary_train_data)
        .unwrap();

    c.bench_function("linfa_svm_rbf_inference", |b| {
        b.iter(|| {
            model.predict(test_data.records())
        })
    });
}

criterion_group!(benches, bench_decision_tree, bench_svm_rbf);
criterion_main!(benches); 
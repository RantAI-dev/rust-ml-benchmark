use criterion::{criterion_group, criterion_main, Criterion};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::svm::{Kernels, svc::{SVC, SVCParameters}};
use smartcore::metrics::accuracy;
use std::collections::HashMap;
use csv::StringRecord;

// Helper functions from main.rs
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

fn load_bank_dataset() -> (DenseMatrix<f64>, DenseMatrix<f64>, Vec<usize>, Vec<usize>) {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b';')
        .from_path("../../data/bank-additional/bank-additional-full.csv")
        .unwrap();

    let mut records: Vec<Vec<f64>> = Vec::new();
    let mut targets_bool: Vec<bool> = Vec::new();
    let mut cat_maps = vec![HashMap::new(); 10];

    for res in rdr.records().skip(1) {
        let rec = res.unwrap();
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

    let train_x: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&train_x_vec).unwrap();
    let test_x: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&test_x_vec).unwrap();

    // Apply standard scaling
    let (scaled_train_x, scaled_test_x) = apply_standard_scaling(&train_x_vec, &test_x_vec);

    // Convert boolean targets to usize for smartcore models
    let train_y: Vec<usize> = train_y_bool.iter().map(|&b| if b {1} else {0}).collect();
    let test_y: Vec<usize> = test_y_bool.iter().map(|&b| if b {1} else {0}).collect();

    (scaled_train_x, scaled_test_x, train_y, test_y)
}

fn bench_random_forest(c: &mut Criterion) {
    let (train_x, test_x, train_y, test_y) = load_bank_dataset();
    
    c.bench_function("smartcore_random_forest_train", |b| {
        b.iter(|| {
            RandomForestClassifier::fit(
                &train_x,
                &train_y,
                RandomForestClassifierParameters::default(),
            ).unwrap()
        })
    });

    let model = RandomForestClassifier::fit(
        &train_x,
        &train_y,
        RandomForestClassifierParameters::default(),
    ).unwrap();

    c.bench_function("smartcore_random_forest_inference", |b| {
        b.iter(|| {
            model.predict(&test_x).unwrap()
        })
    });
}

fn bench_svm_rbf(c: &mut Criterion) {
    let (train_x, test_x, train_y, test_y) = load_bank_dataset();
    
    c.bench_function("smartcore_svm_rbf_train", |b| {
        b.iter(|| {
            let svc_params = SVCParameters::default().with_kernel(Kernels::rbf().with_gamma(0.5));
            SVC::fit(&train_x, &train_y, &svc_params).unwrap()
        })
    });

    let svc_params = SVCParameters::default().with_kernel(Kernels::rbf().with_gamma(0.5));
    let model = SVC::fit(&train_x, &train_y, &svc_params).unwrap();

    c.bench_function("smartcore_svm_rbf_inference", |b| {
        b.iter(|| {
            model.predict(&test_x).unwrap()
        })
    });
}

criterion_group!(benches, bench_random_forest, bench_svm_rbf);
criterion_main!(benches); 
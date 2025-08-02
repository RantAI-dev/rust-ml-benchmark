use clap::Parser;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::process::Command;
use std::time::Instant;

use chrono::{DateTime, Local};
use gethostname::gethostname;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(short, long)] 
    dataset: String,
    #[arg(short, long)] 
    output: String,
    #[arg(long, default_value = "5")] 
    runs: usize,
    #[arg(long, default_value = "1")] 
    warmup: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct SingleBenchResult {
    library: String,
    model: String,
    train_time_sec: f64,
    inference_time_sec: f64,
    accuracy: f64,
    memory_usage_mb: f64,
}

#[derive(Serialize, Debug)]
struct BenchmarkStats {
    mean: f64,
    std_dev: f64,
    min_val: f64,
    max_val: f64,
    count: usize,
}

impl BenchmarkStats {
    fn from_values(values: &[f64]) -> Self {
        let count = values.len();
        let mean = values.iter().sum::<f64>() / count as f64;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();
        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        BenchmarkStats { mean, std_dev, min_val, max_val, count }
    }
}

#[derive(Serialize, Debug, Clone)]
struct SystemInfo {
    hostname: String,
    cpu_model: String,
    cpu_cores: usize,
    total_memory_gb: f64,
    os_version: String,
    rust_version: String,
    smartcore_version: String,
    timestamp: DateTime<Local>,
}

#[derive(Serialize, Debug, Clone)]
struct DatasetInfo {
    name: String,
    path: String,
    preprocessing_notes: String,
}

#[derive(Serialize, Debug)]
struct EnhancedBenchResult {
    system_info: SystemInfo,
    dataset_info: DatasetInfo,
    library: String,
    model: String,
    
    // Statistical results from multiple runs
    train_time_stats: BenchmarkStats,
    inference_time_stats: BenchmarkStats,
    accuracy_stats: BenchmarkStats,
    memory_usage_stats: BenchmarkStats,
    
    // Methodology
    runs: usize,
    warmup_runs: usize,
}

fn collect_system_info() -> SystemInfo {
    let hostname = gethostname().into_string().unwrap_or_else(|_| "unknown".to_string());
    
    // Get CPU info
    let cpu_model = std::fs::read_to_string("/proc/cpuinfo")
        .unwrap_or_default()
        .lines()
        .find(|line| line.starts_with("model name"))
        .and_then(|line| line.split(':').nth(1))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    
    let cpu_cores = num_cpus::get();
    
    // Get memory info (in GB)
    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let total_memory_gb = meminfo
        .lines()
        .find(|line| line.starts_with("MemTotal:"))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|kb| kb.parse::<f64>().ok())
        .map(|kb| kb / 1024.0 / 1024.0)
        .unwrap_or(0.0);
    
    // Get OS version
    let os_version = Command::new("uname")
        .arg("-a")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .unwrap_or_else(|| "unknown".to_string())
        .trim()
        .to_string();
    
    // Get Rust version
    let rust_version = Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .unwrap_or_else(|| "unknown".to_string())
        .trim()
        .to_string();
    
    SystemInfo {
        hostname,
        cpu_model,
        cpu_cores,
        total_memory_gb,
        os_version,
        rust_version,
        smartcore_version: "0.4.0".to_string(),
        timestamp: Local::now(),
    }
}

fn run_single_benchmark(dataset: &str) -> Result<Vec<SingleBenchResult>, Box<dyn Error>> {
    let temp_output = format!("/tmp/smartcore_single_{}.json", std::process::id());
    
    let status = Command::new("cargo")
        .args(&["run", "--release", "--bin", "smartcore_bencher", "--", 
                "--dataset", dataset, "--output", &temp_output])
        .status()?;
    
    if !status.success() {
        return Err("SmartCore benchmark failed".into());
    }
    
    // Read results
    let content = std::fs::read_to_string(&temp_output)?;
    let mut results = Vec::new();
    
    for line in content.lines() {
        if let Ok(result) = serde_json::from_str::<SingleBenchResult>(line) {
            results.push(result);
        }
    }
    
    // Clean up temp file
    let _ = std::fs::remove_file(&temp_output);
    
    Ok(results)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    
    println!("🚀 Enhanced SmartCore Benchmark Suite");
    println!("Running {} warmup + {} measurement runs...", args.warmup, args.runs);
    
    let system_info = collect_system_info();
    let dataset_info = DatasetInfo {
        name: "Bank Marketing".to_string(),
        path: args.dataset.clone(),
        preprocessing_notes: "Standard scaling, categorical encoding".to_string(),
    };
    
    // Warmup runs
    println!("\n🔥 Warmup phase ({} runs)...", args.warmup);
    for i in 0..args.warmup {
        print!("  Warmup run {}/{}... ", i + 1, args.warmup);
        std::io::stdout().flush().unwrap();
        let _ = run_single_benchmark(&args.dataset)?;
        println!("✓");
    }
    
    // Measurement runs
    println!("\n📊 Measurement phase ({} runs)...", args.runs);
    let mut all_runs = Vec::new();
    
    for i in 0..args.runs {
        print!("  Run {}/{}... ", i + 1, args.runs);
        std::io::stdout().flush().unwrap();
        
        let start = Instant::now();
        let run_results = run_single_benchmark(&args.dataset)?;
        let elapsed = start.elapsed();
        
        println!("✓ ({:.2}s)", elapsed.as_secs_f64());
        all_runs.push(run_results);
    }
    
    // Aggregate results by model
    let mut model_results = std::collections::HashMap::new();
    
    for run in &all_runs {
        for result in run {
            let key = format!("{}_{}", result.library, result.model);
            model_results.entry(key).or_insert_with(Vec::new).push(result.clone());
        }
    }
    
    // Generate enhanced results
    let mut enhanced_results = Vec::new();
    
    for (_, results) in model_results {
        if results.is_empty() { continue; }
        
        let first = &results[0];
        
        // Extract timing stats
        let train_times: Vec<f64> = results.iter().map(|r| r.train_time_sec).collect();
        let inference_times: Vec<f64> = results.iter().map(|r| r.inference_time_sec).collect();
        let accuracies: Vec<f64> = results.iter().map(|r| r.accuracy).collect();
        let memory_usages: Vec<f64> = results.iter().map(|r| r.memory_usage_mb).collect();
        
        enhanced_results.push(EnhancedBenchResult {
            system_info: system_info.clone(),
            dataset_info: dataset_info.clone(),
            library: first.library.clone(),
            model: first.model.clone(),
            train_time_stats: BenchmarkStats::from_values(&train_times),
            inference_time_stats: BenchmarkStats::from_values(&inference_times),
            accuracy_stats: BenchmarkStats::from_values(&accuracies),
            memory_usage_stats: BenchmarkStats::from_values(&memory_usages),
            runs: args.runs,
            warmup_runs: args.warmup,
        });
    }
    
    // Write enhanced results
    let mut file = File::create(&args.output)?;
    
    // Write header
    writeln!(file, "# Enhanced SmartCore Benchmark Results")?;
    writeln!(file, "# Generated: {}", Local::now())?;
    writeln!(file, "# System: {} cores, {:.1}GB RAM", 
             system_info.cpu_cores, system_info.total_memory_gb)?;
    writeln!(file, "# Methodology: {} runs + {} warmup", args.runs, args.warmup)?;
    writeln!(file)?;
    
    // Write results
    for result in &enhanced_results {
        writeln!(file, "{}", serde_json::to_string_pretty(result)?)?;
        writeln!(file)?;
    }
    
    // Print summary
    println!("\n📈 Enhanced Benchmark Summary:");
    println!("├─ System: {}", system_info.hostname);
    println!("├─ CPU: {} ({} cores)", system_info.cpu_model, system_info.cpu_cores);
    println!("├─ Memory: {:.1}GB", system_info.total_memory_gb);
    println!("├─ Methodology: {} runs + {} warmup", args.runs, args.warmup);
    println!("└─ Models tested: {}", enhanced_results.len());
    
    for result in &enhanced_results {
        println!("\n🔸 {} - {}:", result.library, result.model);
        println!("  ├─ Training: {:.4}±{:.4}s", 
                 result.train_time_stats.mean, result.train_time_stats.std_dev);
        println!("  ├─ Inference: {:.6}±{:.6}s", 
                 result.inference_time_stats.mean, result.inference_time_stats.std_dev);
        println!("  ├─ Accuracy: {:.4}±{:.4}", 
                 result.accuracy_stats.mean, result.accuracy_stats.std_dev);
        println!("  └─ Memory: {:.2}±{:.2}MB", 
                 result.memory_usage_stats.mean, result.memory_usage_stats.std_dev);
    }
    
    println!("\n✅ Enhanced benchmark complete! Results saved to: {}", args.output);
    
    Ok(())
}

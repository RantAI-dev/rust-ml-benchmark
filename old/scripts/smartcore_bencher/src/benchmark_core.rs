use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use psutil::process::Process;

/// Statistics for multiple benchmark runs
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BenchmarkStats {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub count: usize,
}

impl BenchmarkStats {
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self {
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                count: 0,
            };
        }

        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = if samples.len() > 1 {
            samples.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / (samples.len() - 1) as f64
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        Self {
            mean,
            std_dev,
            min: samples.iter().cloned().fold(f64::INFINITY, f64::min),
            max: samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            count: samples.len(),
        }
    }
}

/// System resource information
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SystemInfo {
    pub hostname: String,
    pub os_version: String,
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub total_memory_gb: f64,
    pub rust_version: String,
    pub smartcore_version: String,
    pub timestamp: String,
}

/// Resource usage measurement
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ResourceUsage {
    pub memory_rss_mb: f64,
    pub memory_vms_mb: f64,
    pub cpu_percent: f32,
}

/// Comprehensive benchmark result with statistics
#[derive(Serialize, Deserialize, Debug)]
pub struct EnhancedBenchResult {
    pub system_info: SystemInfo,
    pub library: String,
    pub model: String,
    pub dataset_info: DatasetInfo,
    
    // Timing statistics
    pub preprocessing_time_stats: BenchmarkStats,
    pub training_time_stats: BenchmarkStats,
    pub inference_time_stats: BenchmarkStats,
    
    // Accuracy statistics
    pub accuracy_stats: BenchmarkStats,
    
    // Resource usage
    pub baseline_resources: ResourceUsage,
    pub peak_training_resources: ResourceUsage,
    pub peak_inference_resources: ResourceUsage,
}

/// Dataset metadata
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DatasetInfo {
    pub name: String,
    pub n_samples: usize,
    pub n_features: usize,
    pub n_classes: usize,
    pub train_samples: usize,
    pub test_samples: usize,
}

/// Benchmark configuration
#[derive(Clone)]
pub struct BenchmarkConfig {
    pub n_runs: usize,
    pub warmup_runs: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            n_runs: 10,
            warmup_runs: 2,
        }
    }
}

/// System information collector
pub struct SystemInfoCollector;

impl SystemInfoCollector {
    pub fn collect() -> SystemInfo {
        let hostname = gethostname::gethostname()
            .into_string()
            .unwrap_or_else(|_| "unknown".to_string());
        
        let os_version = Self::get_os_version();
        let (cpu_model, cpu_cores) = Self::get_cpu_info();
        let total_memory_gb = Self::get_total_memory_gb();
        let rust_version = Self::get_rust_version();
        let smartcore_version = env!("CARGO_PKG_VERSION").to_string();
        let timestamp = chrono::Utc::now().to_rfc3339();

        SystemInfo {
            hostname,
            os_version,
            cpu_model,
            cpu_cores,
            total_memory_gb,
            rust_version,
            smartcore_version,
            timestamp,
        }
    }

    fn get_os_version() -> String {
        match std::process::Command::new("uname").arg("-a").output() {
            Ok(output) => String::from_utf8_lossy(&output.stdout).trim().to_string(),
            Err(_) => "unknown".to_string(),
        }
    }

    fn get_cpu_info() -> (String, usize) {
        // Try to get CPU info from /proc/cpuinfo on Linux
        if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
            let mut model = "unknown".to_string();
            let mut cores = 1;
            
            for line in content.lines() {
                if line.starts_with("model name") {
                    if let Some(name) = line.split(':').nth(1) {
                        model = name.trim().to_string();
                    }
                } else if line.starts_with("processor") {
                    cores += 1;
                }
            }
            
            return (model, cores);
        }

        // Fallback: try lscpu
        if let Ok(output) = std::process::Command::new("lscpu").output() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut model = "unknown".to_string();
            let mut cores = 1;

            for line in stdout.lines() {
                if line.starts_with("Model name:") {
                    model = line.split(':').nth(1).unwrap_or("unknown").trim().to_string();
                } else if line.starts_with("CPU(s):") {
                    if let Ok(c) = line.split(':').nth(1).unwrap_or("1").trim().parse() {
                        cores = c;
                    }
                }
            }

            return (model, cores);
        }

        // Final fallback
        ("unknown".to_string(), num_cpus::get())
    }

    fn get_total_memory_gb() -> f64 {
        match psutil::memory::virtual_memory() {
            Ok(mem) => mem.total() as f64 / (1024.0 * 1024.0 * 1024.0),
            Err(_) => 0.0,
        }
    }

    fn get_rust_version() -> String {
        match std::process::Command::new("rustc").arg("--version").output() {
            Ok(output) => String::from_utf8_lossy(&output.stdout).trim().to_string(),
            Err(_) => "unknown".to_string(),
        }
    }
}

/// Resource monitor for tracking CPU and memory usage
pub struct ResourceMonitor {
    process: Process,
}

impl ResourceMonitor {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            process: Process::current()?,
        })
    }

    pub fn measure(&mut self) -> Result<ResourceUsage, Box<dyn std::error::Error>> {
        let memory_info = self.process.memory_info()?;
        let cpu_percent = self.process.cpu_percent()?;

        Ok(ResourceUsage {
            memory_rss_mb: memory_info.rss() as f64 / (1024.0 * 1024.0),
            memory_vms_mb: memory_info.vms() as f64 / (1024.0 * 1024.0),
            cpu_percent,
        })
    }
}

/// Timer for high-precision measurements
pub struct BenchmarkTimer {
    start: Option<Instant>,
}

impl BenchmarkTimer {
    pub fn new() -> Self {
        Self { start: None }
    }

    pub fn start(&mut self) {
        self.start = Some(Instant::now());
    }

    pub fn elapsed_secs(&self) -> f64 {
        match self.start {
            Some(start) => start.elapsed().as_secs_f64(),
            None => 0.0,
        }
    }

    pub fn restart(&mut self) -> f64 {
        let elapsed = self.elapsed_secs();
        self.start();
        elapsed
    }
}

/// Trait for benchmarkable ML models
pub trait BenchmarkableModel<X, Y> {
    type Model;

    fn train(&self, x: &X, y: &Y) -> Result<Self::Model, Box<dyn std::error::Error>>;
    fn predict(&self, model: &Self::Model, x: &X) -> Result<Vec<usize>, Box<dyn std::error::Error>>;
    fn name(&self) -> &str;
}

/// Core benchmarking engine
pub struct BenchmarkEngine {
    config: BenchmarkConfig,
    monitor: ResourceMonitor,
}

impl BenchmarkEngine {
    pub fn new(config: BenchmarkConfig) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            config,
            monitor: ResourceMonitor::new()?,
        })
    }

    pub fn run_benchmark<X, Y, M>(
        &mut self,
        model_impl: &M,
        train_x: &X,
        train_y: &Y,
        test_x: &X,
        test_y: &[usize],
        dataset_info: DatasetInfo,
    ) -> Result<EnhancedBenchResult, Box<dyn std::error::Error>>
    where
        M: BenchmarkableModel<X, Y>,
    {
        let system_info = SystemInfoCollector::collect();
        let baseline_resources = self.monitor.measure()?;

        // Warmup runs
        for _ in 0..self.config.warmup_runs {
            let _ = model_impl.train(train_x, train_y)?;
        }

        // Collect timing samples
        let mut training_times = Vec::new();
        let mut inference_times = Vec::new();
        let mut accuracies = Vec::new();
        let mut peak_training_resources = baseline_resources.clone();
        let mut peak_inference_resources = baseline_resources.clone();

        for run in 0..self.config.n_runs {
            println!("  Run {}/{}", run + 1, self.config.n_runs);

            // Training phase
            let mut timer = BenchmarkTimer::new();
            timer.start();
            let model = model_impl.train(train_x, train_y)?;
            let training_time = timer.elapsed_secs();
            training_times.push(training_time);

            // Check peak training resources
            let current_resources = self.monitor.measure()?;
            if current_resources.memory_rss_mb > peak_training_resources.memory_rss_mb {
                peak_training_resources = current_resources;
            }

            // Inference phase
            timer.start();
            let predictions = model_impl.predict(&model, test_x)?;
            let inference_time = timer.elapsed_secs();
            inference_times.push(inference_time);

            // Check peak inference resources
            let current_resources = self.monitor.measure()?;
            if current_resources.memory_rss_mb > peak_inference_resources.memory_rss_mb {
                peak_inference_resources = current_resources;
            }

            // Calculate accuracy
            let correct = predictions.iter()
                .zip(test_y.iter())
                .filter(|(pred, actual)| pred == actual)
                .count();
            let accuracy = correct as f64 / test_y.len() as f64;
            accuracies.push(accuracy);
        }

        Ok(EnhancedBenchResult {
            system_info,
            library: "smartcore".to_string(),
            model: model_impl.name().to_string(),
            dataset_info,
            preprocessing_time_stats: BenchmarkStats::from_samples(&[]), // To be filled by caller
            training_time_stats: BenchmarkStats::from_samples(&training_times),
            inference_time_stats: BenchmarkStats::from_samples(&inference_times),
            accuracy_stats: BenchmarkStats::from_samples(&accuracies),
            baseline_resources,
            peak_training_resources,
            peak_inference_resources,
        })
    }
}

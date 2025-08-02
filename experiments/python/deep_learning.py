import time
import json
import argparse
import os
import threading
import psutil
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Attempt to import and initialize pynvml for GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except (ImportError, pynvml.NVMLError):
    NVML_AVAILABLE = False

# --- Helper Class for Resource Monitoring ---
class ResourceMonitor:
    """A thread-safe class to monitor CPU, System RAM, and GPU resources."""
    def __init__(self, device):
        self.cpu_samples = []
        self.memory_samples = []
        self.gpu_util_samples = []
        self.gpu_mem_samples = []
        self.monitoring = False
        self.monitor_thread = None
        self.device = device
        self.gpu_handle = None

        if self.device.type == 'cuda' and NVML_AVAILABLE:
            # Assumes device index 0, adjust if using multi-GPU
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def start(self):
        self.monitoring = True
        self.cpu_samples, self.memory_samples = [], []
        self.gpu_util_samples, self.gpu_mem_samples = [], []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        # Prevent errors if no samples were collected
        if not self.cpu_samples: self.cpu_samples.append(0)
        if not self.memory_samples: self.memory_samples.append(0)
        if not self.gpu_util_samples: self.gpu_util_samples.append(0)
        if not self.gpu_mem_samples: self.gpu_mem_samples.append(0)

        return {
            "cpu_utilization_mean_percent": np.mean(self.cpu_samples),
            "cpu_utilization_peak_percent": np.max(self.cpu_samples),
            "system_memory_mean_mb": np.mean(self.memory_samples),
            "system_memory_peak_mb": np.max(self.memory_samples),
            "gpu_utilization_mean_percent": np.mean(self.gpu_util_samples),
            "gpu_utilization_peak_percent": np.max(self.gpu_util_samples),
            "gpu_memory_mean_mb": np.mean(self.gpu_mem_samples),
            "gpu_memory_peak_mb": np.max(self.gpu_mem_samples),
        }

    def _monitor_loop(self):
        process = psutil.Process(os.getpid())
        while self.monitoring:
            try:
                self.cpu_samples.append(process.cpu_percent(interval=0.1))
                self.memory_samples.append(process.memory_info().rss / (1024 * 1024))
                if self.gpu_handle:
                    gpu_info = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    self.gpu_util_samples.append(gpu_info.gpu)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    self.gpu_mem_samples.append(mem_info.used / (1024 * 1024))
                time.sleep(0.1) # Adjust sample rate if needed
            except (psutil.NoSuchProcess, pynvml.NVMLError):
                break

# --- Model and Data Functions ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')

class StandardCNN(nn.Module):
    def __init__(self):
        super(StandardCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512), nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x): return self.network(x)

def get_data_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

# --- Main Benchmark Function ---
def run_deep_learning_benchmark(epochs, batch_size, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    monitor = ResourceMonitor(device)
    
    model = StandardCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader, test_loader = get_data_loaders(batch_size)

    print(f"Starting Deep Learning benchmark on device: {device}")
    monitor.start()
    total_start_time = time.perf_counter()
    
    epoch_durations, convergence_curve = [], []
    for epoch in range(epochs):
        epoch_start_time = time.perf_counter()
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_duration = time.perf_counter() - epoch_start_time
        epoch_loss = running_loss / len(train_loader)
        epoch_durations.append(epoch_duration)
        convergence_curve.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Duration: {epoch_duration:.2f}s")

    total_duration = time.perf_counter() - total_start_time
    resource_stats = monitor.stop()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    
    num_samples = len(train_loader.dataset)
    throughput = num_samples * epochs / total_duration

    return {
        "performance": {
            "total_training_time_seconds": total_duration,
            "time_per_epoch_seconds": epoch_durations,
            "training_throughput_samples_per_sec": throughput,
        },
        "resources": resource_stats,
        "quality": {
            "model_accuracy": accuracy,
            "convergence_curve_loss": convergence_curve,
        },
        "metadata": {
            "device_used": str(device),
            "epochs": epochs,
            "batch_size": batch_size,
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Run Deep Learning benchmarks for Python.")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for Adam optimizer.')
    parser.add_argument('--output', type=str, help='Output file path for benchmark results.')
    args = parser.parse_args()

    results = {
        "language": "python",
        "framework": "pytorch",
        "timestamp": time.time(),
        "benchmark_results": run_deep_learning_benchmark(args.epochs, args.batch_size, args.lr)
    }

    output_json = json.dumps(results, indent=4)
    if args.output:
        with open(args.output, 'w') as f: f.write(output_json)
        print(f"\nBenchmark complete. Results saved to {args.output}")
    else:
        print(output_json)

if __name__ == '__main__':
    main()

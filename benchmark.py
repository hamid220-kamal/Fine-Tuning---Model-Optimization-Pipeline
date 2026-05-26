import os
import time
import json
import psutil
from serve import LocalMicroServer

def measure_directory_size(path=".") -> float:
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024) # MB

def run_benchmark():
    print("Initializing Real-Time Efficiency Benchmarker...")
    print("Checking constraints...")
    
    server = LocalMicroServer()
    process = psutil.Process(os.getpid())
    
    iterations = 50
    total_time = 0
    total_tokens = 0
    valid_json_count = 0
    peak_ram_mb = 0
    
    prompts = ["Help my account is locked!", "I need a refund right now", "How do I upgrade my plan?"]
    
    print(f"Starting {iterations} iterations of zero-GPU inference...")
    
    for i in range(iterations):
        prompt = prompts[i % len(prompts)]
        
        start_time = time.time()
        output = server.generate_json_with_rejection_sampling(prompt)
        end_time = time.time()
        
        duration = end_time - start_time
        total_time += duration
        
        tokens_generated = len(output.split())
        total_tokens += tokens_generated
        
        current_ram_mb = process.memory_info().rss / (1024 * 1024)
        if current_ram_mb > peak_ram_mb:
            peak_ram_mb = current_ram_mb
            
        try:
            json.loads(output)
            valid_json_count += 1
        except json.JSONDecodeError:
            pass
            
    avg_tps = total_tokens / total_time if total_time > 0 else 0
    storage_mb = measure_directory_size()
    reliability = (valid_json_count / iterations) * 100
    
    print("\n# 🚀 Pipeline Benchmark Results\n")
    print("| Metric | Value | Status |")
    print("|---|---|---|")
    print(f"| ⚡ Inference Speed | {avg_tps:.2f} tokens/sec | CPU-Optimized |")
    print(f"| 🧠 Peak Memory (RAM) | {peak_ram_mb:.2f} MB | Mmap Engaged |")
    print(f"| 💾 Storage Footprint | {storage_mb:.2f} MB | {'✅ < 1GB' if storage_mb < 1024 else '❌ > 1GB'} |")
    print(f"| 🎯 JSON Reliability | {reliability:.1f}% | Structured Alignment |")

if __name__ == "__main__":
    run_benchmark()

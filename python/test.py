#!/usr/bin/env python3
"""
NT-Xent Loss CUDA Extension Test Suite
This module provides comprehensive testing for the NT-Xent CUDA implementation.
"""

import torch
import numpy as np
from time import perf_counter
import os
import sys
from typing import Dict, Tuple, Any, List
import logging
import json
from pathlib import Path
import torch.cuda.amp as amp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GPUMemoryTracker:
    def __init__(self):
        self.memory_stats = []
        
    def log_memory(self, step: str):
        stats = {
            'step': step,
            'allocated': torch.cuda.memory_allocated() / 1024**2,
            'cached': torch.cuda.memory_reserved() / 1024**2
        }
        self.memory_stats.append(stats)
        logger.info(f"Memory at {step}: {stats['allocated']:.1f}MB allocated, {stats['cached']:.1f}MB cached")
        
    def save_report(self, output_path: str):
        with open(output_path, 'w') as f:
            json.dump(self.memory_stats, f, indent=2)

def verify_gpu_requirements() -> None:
    """
    Verify that CUDA is available and the GPU meets minimum requirements.
    Raises RuntimeError if requirements are not met.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    cuda_capability = torch.cuda.get_device_capability()
    if cuda_capability[0] < 7:  # Minimum Volta architecture
        raise RuntimeError(
            f"GPU compute capability {cuda_capability[0]}.{cuda_capability[1]} "
            "is too low. Minimum requirement is 7.0 (Volta)"
        )

def test_numerical_stability(ntxent, batch_size: int, dim: int) -> bool:
    """Test numerical stability with edge cases"""
    device = torch.device('cuda')
    z = torch.randn(batch_size, dim, device=device)
    z = torch.nn.functional.normalize(z)
    
    # Test with various scales
    scales = [1e-5, 1.0, 1e5]
    temps = [0.01, 0.07, 1.0]
    
    for scale in scales:
        for temp in temps:
            scaled_z = z * scale
            try:
                loss = ntxent.forward(scaled_z, temp)
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"Numerical instability detected: scale={scale}, temp={temp}")
                    return False
            except Exception as e:
                logger.error(f"Error in numerical stability test: {e}")
                return False
    
    return True

def run_performance_test(
    ntxent,
    batch_size: int,
    dim: int,
    num_runs: int = 100,
    use_amp: bool = False
) -> Dict[str, float]:
    """Run performance tests with optional AMP"""
    device = torch.device('cuda')
    z = torch.randn(batch_size, dim, device=device)
    z = torch.nn.functional.normalize(z)
    
    if use_amp:
        scaler = amp.GradScaler()
    
    # Warmup
    for _ in range(10):
        if use_amp:
            with amp.autocast():
                _ = ntxent.forward(z, 0.07)
        else:
            _ = ntxent.forward(z, 0.07)
    
    torch.cuda.synchronize()
    
    times = []
    memory_usage = []
    
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = perf_counter()
        
        if use_amp:
            with amp.autocast():
                loss = ntxent.forward(z, 0.07)
        else:
            loss = ntxent.forward(z, 0.07)
            
        torch.cuda.synchronize()
        times.append(perf_counter() - start)
        memory_usage.append(torch.cuda.memory_allocated())
    
    return {
        'mean_time': np.mean(times) * 1000,  # Convert to ms
        'std_time': np.std(times) * 1000,
        'min_time': np.min(times) * 1000,
        'max_time': np.max(times) * 1000,
        'mean_memory': np.mean(memory_usage) / 1024**2,  # Convert to MB
        'max_memory': np.max(memory_usage) / 1024**2
    }

def test_ntxent() -> Dict[Tuple[int, int], Dict[str, float]]:
    memory_tracker = GPUMemoryTracker()
    
    try:
        # Load CUDA extension
        ntxent = torch.ops.ntxent_cuda
        memory_tracker.log_memory("Extension loaded")
        
        # Test parameters
        batch_sizes = [32, 64, 128, 256, 512]
        dimensions = [64, 128, 256, 512]
        results = {}
        
        # Test numerical stability
        logger.info("Testing numerical stability...")
        if not test_numerical_stability(ntxent, 128, 256):
            raise RuntimeError("Numerical stability test failed")
        memory_tracker.log_memory("After stability test")
        
        # Run performance tests
        for batch_size in batch_sizes:
            for dim in dimensions:
                logger.info(f"\nTesting with batch_size={batch_size}, dim={dim}")
                
                # Test with and without AMP
                fp32_results = run_performance_test(ntxent, batch_size, dim, use_amp=False)
                amp_results = run_performance_test(ntxent, batch_size, dim, use_amp=True)
                
                results[(batch_size, dim)] = {
                    'fp32': fp32_results,
                    'amp': amp_results
                }
                
                memory_tracker.log_memory(f"After test (B={batch_size}, D={dim})")
                
                # Log results
                logger.info(
                    f"FP32: {fp32_results['mean_time']:.2f}±{fp32_results['std_time']:.2f}ms, "
                    f"Memory: {fp32_results['mean_memory']:.1f}MB"
                )
                logger.info(
                    f"AMP:  {amp_results['mean_time']:.2f}±{amp_results['std_time']:.2f}ms, "
                    f"Memory: {amp_results['mean_memory']:.1f}MB"
                )
        
        # Save memory report
        memory_tracker.save_report('memory_profile.json')
        return results
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        # Print system info
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"PyTorch: {torch.__version__}")
        
        initial_memory = torch.cuda.memory_allocated()
        results = test_ntxent()
        
        # Save results
        output_dir = Path('benchmark_results')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = Path.ctime(Path()).replace(' ', '_').replace(':', '-')
        result_file = output_dir / f'results_{timestamp}.json'
        
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nResults saved to {result_file}")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}", exc_info=True)
        sys.exit(1)
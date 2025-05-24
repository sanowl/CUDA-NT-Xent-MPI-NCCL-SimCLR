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
from typing import Dict, Tuple, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def test_ntxent() -> Dict[Tuple[int, int], Dict[str, float]]:
    try:
        # Load CUDA extension
        ntxent = torch.ops.ntxent_cuda
    except AttributeError as e:
        print("Error: Could not load CUDA extension. Make sure it's properly built.")
        sys.exit(1)

    # Test parameters
    batch_sizes = [32, 64, 128]
    dimensions = [64, 128, 256]
    temperature = 0.07
    num_runs = 10

    # Enable CUDA timing
    torch.cuda.synchronize()
    
    results = {}
    
    for B in batch_sizes:
        for D in dimensions:
            times = []
            print(f"\nTesting with batch size {B}, dimension {D}")
            
            # Generate random normalized vectors
            z = torch.randn(B, D, device='cuda')
            z = torch.nn.functional.normalize(z, dim=1)
            
            # Warmup run
            _ = ntxent.forward(z, temperature)
            torch.cuda.synchronize()
            
            # Benchmark runs
            for i in range(num_runs):
                start = perf_counter()
                loss = ntxent.forward(z, temperature)
                torch.cuda.synchronize()
                end = perf_counter()
                
                times.append((end - start) * 1000)  # Convert to ms
                
                # Basic correctness checks
                assert not torch.isnan(loss), f"NaN loss detected at iteration {i}"
                assert loss.item() > 0, f"Invalid negative loss: {loss.item()}"
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            print(f"Average forward time: {avg_time:.2f} ± {std_time:.2f} ms")
            print(f"Loss value: {loss.item():.4f}")
            
            results[(B, D)] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'loss': loss.item()
            }
    
    return results

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        sys.exit(1)
        
    print(f"Running on GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    try:
        results = test_ntxent()
        
        # Print summary
        print("\nTest Summary:")
        print("-" * 60)
        for (B, D), metrics in results.items():
            print(f"Batch={B}, Dim={D}:")
            print(f"  Time: {metrics['avg_time']:.2f} ± {metrics['std_time']:.2f} ms")
            print(f"  Loss: {metrics['loss']:.4f}")
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        sys.exit(1)
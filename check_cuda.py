#!/usr/bin/env python3
"""
Script to check if CUDA is installed correctly and available for use.
"""

import sys

def check_cuda():
    """Check CUDA installation and availability."""
    print("=" * 60)
    print("CUDA Installation Check")
    print("=" * 60)
    
    # Check PyTorch installation
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch is not installed")
        return False
    
    # Check CUDA availability in PyTorch
    cuda_available = torch.cuda.is_available()
    print(f"✓ CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("\n✗ CUDA is not available. Possible reasons:")
        print("  - CUDA drivers are not installed")
        print("  - PyTorch was installed without CUDA support")
        print("  - No compatible GPU found")
        return False
    
    # Get CUDA version
    cuda_version = torch.version.cuda
    print(f"✓ CUDA version: {cuda_version}")
    
    # Get cuDNN version
    try:
        cudnn_version = torch.backends.cudnn.version()
        print(f"✓ cuDNN version: {cudnn_version}")
    except:
        print("✗ cuDNN version: Not available")
    
    # Check GPU count
    gpu_count = torch.cuda.device_count()
    print(f"✓ Number of GPUs: {gpu_count}")
    
    # Get GPU information
    print("\n" + "-" * 60)
    print("GPU Information:")
    print("-" * 60)
    
    for i in range(gpu_count):
        print(f"\nGPU {i}:")
        gpu_name = torch.cuda.get_device_name(i)
        print(f"  Name: {gpu_name}")
        
        # Get GPU properties
        props = torch.cuda.get_device_properties(i)
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Multiprocessors: {props.multi_processor_count}")
        
        # Get current memory usage
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  Memory Allocated: {memory_allocated:.2f} GB")
        print(f"  Memory Reserved: {memory_reserved:.2f} GB")
    
    # Test CUDA operations
    print("\n" + "-" * 60)
    print("Testing CUDA Operations:")
    print("-" * 60)
    
    try:
        # Create a tensor on GPU
        test_tensor = torch.randn(1000, 1000).cuda()
        print("✓ Successfully created tensor on GPU")
        
        # Perform a simple operation
        result = torch.matmul(test_tensor, test_tensor)
        print("✓ Successfully performed matrix multiplication on GPU")
        
        # Clean up
        del test_tensor, result
        torch.cuda.empty_cache()
        print("✓ Successfully cleaned up GPU memory")
        
    except Exception as e:
        print(f"✗ Error testing CUDA operations: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All CUDA checks passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = check_cuda()
    sys.exit(0 if success else 1)


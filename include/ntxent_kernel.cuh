#pragma once
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>

namespace ntxent {

// Constants for kernel optimization
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS = 1024;
constexpr int MIN_BLOCKS_PER_SM = 2;
constexpr int MAX_SHARED_MEMORY = 48 * 1024;  // 48KB for modern GPUs

// Error checking utilities
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + \
                               cudaGetErrorString(err)); \
    } \
} while(0)

/**
 * Forward pass of the NT-Xent loss function
 * @param z Input tensor of shape (batch_size, dim)
 * @param T Temperature parameter for scaling
 * @returns Loss tensor
 * @throws std::runtime_error on CUDA errors
 */
torch::Tensor ntxent_forward_cuda(
    torch::Tensor z,
    double T,
    bool use_mixed_precision = false
);

/**
 * Backward pass of the NT-Xent loss function
 * @param z Input tensor from forward pass
 * @param softmax Softmax output from forward pass
 * @param grad_out Gradient of the loss
 * @param T Temperature parameter
 * @returns Tuple of gradients (grad_z, grad_internal)
 * @throws std::runtime_error on CUDA errors
 */
std::tuple<torch::Tensor, torch::Tensor> ntxent_backward_cuda(
    torch::Tensor z,
    torch::Tensor softmax,
    torch::Tensor grad_out,
    double T,
    bool use_mixed_precision = false
);

// CUDA Kernel declarations
namespace cuda {
    __global__ void row_max_kernel(
        const float* __restrict__ logits,
        float* __restrict__ row_max,
        const int rows,
        const int cols
    );

    __global__ void softmax_kernel(
        const float* __restrict__ logits,
        const float* __restrict__ row_max,
        float* __restrict__ softmax_output,
        const int rows,
        const int cols
    );

    __global__ void compute_loss_kernel(
        const float* __restrict__ softmax_output,
        float* __restrict__ loss,
        const int batch_size
    );
}

// Utility functions
namespace utils {
    inline int get_optimal_block_size(int problem_size) {
        int device_id;
        CUDA_CHECK(cudaGetDevice(&device_id));
        
        int max_threads_per_block;
        CUDA_CHECK(cudaDeviceGetAttribute(
            &max_threads_per_block,
            cudaDevAttrMaxThreadsPerBlock,
            device_id
        ));
        
        return std::min({
            nextPowerOf2(problem_size),
            MAX_THREADS,
            max_threads_per_block
        });
    }
    
    inline bool check_tensor_core_support() {
        int device_id;
        CUDA_CHECK(cudaGetDevice(&device_id));
        
        int major, minor;
        CUDA_CHECK(cudaDeviceGetAttribute(
            &major,
            cudaDevAttrComputeCapabilityMajor,
            device_id
        ));
        
        return major >= 7;  // Volta or newer
    }

    // cuBLAS handle singleton
    class CublasHandle {
    public:
        static cublasHandle_t get() {
            static CublasHandle instance;
            return instance.handle_;
        }
        
    private:
        CublasHandle() {
            if (cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("Failed to create cuBLAS handle");
            }
            if (check_tensor_core_support()) {
                cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH);
            }
        }
        
        ~CublasHandle() {
            cublasDestroy(handle_);
        }
        
        cublasHandle_t handle_;
    };
}

} // namespace ntxent

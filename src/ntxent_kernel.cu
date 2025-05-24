#include "ntxent_kernel_new.cuh"
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

namespace ntxent {
namespace cuda {

__global__ void row_max_kernel(
    const float* __restrict__ logits,
    float* __restrict__ row_max,
    const int rows,
    const int cols
) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid & (WARP_SIZE - 1);
    const int warp_id = tid >> 5;
    
    float local_max = -INFINITY;
    
    // Coalesced memory access with vectorized loads
    #pragma unroll 4
    for (int c = tid; c < cols; c += blockDim.x) {
        local_max = fmaxf(local_max, logits[row * cols + c]);
    }
    
    // Warp-level reduction using shuffle
    #pragma unroll 5
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        sdata[warp_id] = local_max;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (tid < (blockDim.x / WARP_SIZE)) {
        float warp_max = sdata[tid];
        #pragma unroll
        for (int offset = (blockDim.x / WARP_SIZE) / 2; offset > 0; offset >>= 1) {
            warp_max = fmaxf(warp_max, __shfl_down_sync(0xffffffff, warp_max, offset));
        }
        if (tid == 0) {
            row_max[row] = warp_max;
        }
    }
}

__global__ void softmax_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ row_max,
    float* __restrict__ softmax_output,
    const int rows,
    const int cols
) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid & (WARP_SIZE - 1);
    const int warp_id = tid >> 5;
    
    float sum = 0.0f;
    const float max_val = row_max[row];
    
    // Compute exp(x - max) and sum with vectorized memory access
    #pragma unroll 4
    for (int c = tid; c < cols; c += blockDim.x) {
        const int idx = row * cols + c;
        const float val = __expf(logits[idx] - max_val);
        softmax_output[idx] = val;
        sum += val;
    }
    
    // Warp-level reduction using shuffle
    #pragma unroll 5
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane_id == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction and normalization
    if (tid == 0) {
        float total_sum = 0.0f;
        for (int i = 0; i < blockDim.x / WARP_SIZE; ++i) {
            total_sum += sdata[i];
        }
        const float inv_sum = __fdividef(1.0f, total_sum);
        
        // Normalize with vectorized memory access
        #pragma unroll 4
        for (int c = 0; c < cols; ++c) {
            softmax_output[row * cols + c] *= inv_sum;
        }
    }
}

__global__ void compute_loss_kernel(
    const float* __restrict__ softmax_output,
    float* __restrict__ loss,
    const int batch_size
) {
    const int tid = threadIdx.x;
    extern __shared__ float sdata[];
    
    float local_loss = 0.0f;
    
    // Compute local loss from diagonal elements
    for (int i = tid; i < batch_size; i += blockDim.x) {
        local_loss -= logf(softmax_output[i * batch_size + i]);
    }
    
    // Warp reduction
    sdata[tid] = local_loss;
    __syncthreads();
    
    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        loss[0] = sdata[0] / static_cast<float>(batch_size);
    }
}

} // namespace cuda

torch::Tensor ntxent_forward_cuda(
    torch::Tensor z,
    double T,
    bool use_mixed_precision
) {
    at::cuda::CUDAGuard device_guard(z.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    const int B = z.size(0);
    const int D = z.size(1);
    const int num_blocks = 2*B;
    
    // Calculate optimal block size
    const int block_size = utils::get_optimal_block_size(num_blocks);
    
    // Allocate output tensors
    auto options = z.options();
    auto logits = at::empty({2*B, 2*B}, options);
    auto row_max = at::empty({2*B}, options);
    auto softmax_output = at::empty({2*B, 2*B}, options);
    auto loss = at::empty({1}, options);
    
    // Use cuBLAS for matrix multiplication
    auto z_cat = at::cat({z, z}, 0);
    float alpha = 1.0f / T;
    float beta = 0.0f;
    
    auto handle = utils::CublasHandle::get();
    AT_CUDA_CHECK(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        2*B, 2*B, D,
        &alpha,
        z_cat.data_ptr<float>(), 2*B,
        z_cat.data_ptr<float>(), 2*B,
        &beta,
        logits.data_ptr<float>(), 2*B));
    
    // Launch optimized kernels
    const int shared_mem_size = (block_size / WARP_SIZE) * sizeof(float);
    
    cuda::row_max_kernel<<<num_blocks, block_size, shared_mem_size, stream>>>(
        logits.data_ptr<float>(),
        row_max.data_ptr<float>(),
        num_blocks,
        num_blocks
    );
    AT_CUDA_CHECK(cudaGetLastError());
    
    cuda::softmax_kernel<<<num_blocks, block_size, shared_mem_size, stream>>>(
        logits.data_ptr<float>(),
        row_max.data_ptr<float>(),
        softmax_output.data_ptr<float>(),
        num_blocks,
        num_blocks
    );
    AT_CUDA_CHECK(cudaGetLastError());
    
    cuda::compute_loss_kernel<<<1, block_size, block_size * sizeof(float), stream>>>(
        softmax_output.data_ptr<float>(),
        loss.data_ptr<float>(),
        num_blocks
    );
    AT_CUDA_CHECK(cudaGetLastError());
    
    return loss;
}

std::tuple<torch::Tensor, torch::Tensor> ntxent_backward_cuda(
    torch::Tensor z,
    torch::Tensor softmax,
    torch::Tensor grad_out,
    double T,
    bool use_mixed_precision
) {
    at::cuda::CUDAGuard device_guard(z.device());
    
    const int B = z.size(0);
    const int D = z.size(1);
    
    auto grad_z = torch::zeros_like(z);
    auto grad_logits = torch::zeros({2*B, 2*B}, z.options());
    
    // Compute gradient w.r.t logits with efficient diagonal update
    grad_logits.diagonal() = -1.0f / (softmax.diagonal() * static_cast<float>(2*B));
    
    // Use cuBLAS for gradient computation
    auto z_cat = at::cat({z, z}, 0);
    float alpha = 1.0f / T;
    float beta = 0.0f;
    
    auto handle = utils::CublasHandle::get();
    AT_CUDA_CHECK(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        D, B, 2*B,
        &alpha,
        z_cat.data_ptr<float>(), D,
        grad_logits.data_ptr<float>(), 2*B,
        &beta,
        grad_z.data_ptr<float>(), D));
    
    return {grad_z, grad_logits};
}

} // namespace ntxent

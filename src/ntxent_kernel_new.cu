#include "ntxent_kernel_new.cuh"
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>

namespace ntxent {
namespace cuda {

using namespace cooperative_groups;

// Improved row_max_kernel with cooperative groups
__global__ void row_max_kernel(
    const float* __restrict__ logits,
    float* __restrict__ row_max,
    const int rows,
    const int cols
) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    auto warp = cooperative_groups::tiled_partition<32>(block);
    
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Use vectorized loads for better memory coalescing
    using float4_t = float4;
    const int items_per_thread = 4;
    float local_max = -INFINITY;
    
    // Vectorized memory access
    #pragma unroll 4
    for (int c = tid * items_per_thread; c < cols; c += blockDim.x * items_per_thread) {
        if (c + items_per_thread <= cols) {
            float4_t data = reinterpret_cast<const float4_t*>(logits + row * cols)[c / items_per_thread];
            local_max = fmaxf(local_max, fmaxf(fmaxf(data.x, data.y), fmaxf(data.z, data.w)));
        } else {
            for (int i = 0; i < items_per_thread && c + i < cols; ++i) {
                local_max = fmaxf(local_max, logits[row * cols + c + i]);
            }
        }
    }
    
    // Warp-level reduction using cooperative groups
    local_max = warp.shfl_down(local_max, 16);
    local_max = warp.shfl_down(local_max, 8);
    local_max = warp.shfl_down(local_max, 4);
    local_max = warp.shfl_down(local_max, 2);
    local_max = warp.shfl_down(local_max, 1);
    
    // First thread in each warp writes to shared memory
    __shared__ float warp_maxes[32];
    if (warp.thread_rank() == 0) {
        warp_maxes[warp.meta_group_rank()] = local_max;
    }
    block.sync();
    
    // Final reduction across warps
    if (warp.meta_group_rank() == 0 && warp.thread_rank() < (block.size() / 32)) {
        float warp_max = warp_maxes[warp.thread_rank()];
        warp_max = warp.shfl_down(warp_max, 16);
        warp_max = warp.shfl_down(warp_max, 8);
        warp_max = warp.shfl_down(warp_max, 4);
        warp_max = warp.shfl_down(warp_max, 2);
        warp_max = warp.shfl_down(warp_max, 1);
        
        if (warp.thread_rank() == 0) {
            row_max[row] = warp_max;
        }
    }
}

// Improved softmax kernel with shared memory optimization
__global__ void softmax_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ row_max,
    float* __restrict__ softmax_output,
    const int rows,
    const int cols
) {
    auto block = cooperative_groups::this_thread_block();
    auto warp = cooperative_groups::tiled_partition<32>(block);
    
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    extern __shared__ float shared_data[];
    float* sum_shared = shared_data;
    float* output_shared = &shared_data[blockDim.x];
    
    const float max_val = row_max[row];
    float sum = 0.0f;
    
    // Vectorized computation
    #pragma unroll 4
    for (int c = tid; c < cols; c += blockDim.x) {
        const float val = __expf(logits[row * cols + c] - max_val);
        output_shared[c] = val;
        sum += val;
    }
    
    // Warp-level reduction
    sum = warp_reduce_sum(warp, sum);
    
    if (warp.thread_rank() == 0) {
        sum_shared[warp.meta_group_rank()] = sum;
    }
    block.sync();
    
    // Final reduction and normalization
    if (tid == 0) {
        float total_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < block.size() / 32; ++i) {
            total_sum += sum_shared[i];
        }
        const float inv_sum = __fdividef(1.0f, total_sum);
        
        // Store normalized values with vectorized writes
        using float4_t = float4;
        const int vec_cols = cols / 4;
        
        float4_t* out_vec = reinterpret_cast<float4_t*>(softmax_output + row * cols);
        const float4_t* in_vec = reinterpret_cast<const float4_t*>(output_shared);
        
        #pragma unroll 4
        for (int i = 0; i < vec_cols; ++i) {
            float4_t tmp = in_vec[i];
            tmp.x *= inv_sum;
            tmp.y *= inv_sum;
            tmp.z *= inv_sum;
            tmp.w *= inv_sum;
            out_vec[i] = tmp;
        }
        
        // Handle remaining elements
        for (int i = vec_cols * 4; i < cols; ++i) {
            softmax_output[row * cols + i] = output_shared[i] * inv_sum;
        }
    }
}

// Helper function for warp-level sum reduction
__device__ __forceinline__ float warp_reduce_sum(
    cooperative_groups::thread_block_tile<32>& warp,
    float val
) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += warp.shfl_down(val, offset);
    }
    return val;
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

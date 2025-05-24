#pragma once
#include <torch/torch.h>
#include <random>

namespace test_utils {

inline torch::Tensor generate_random_embeddings(
    int batch_size, 
    int dim, 
    torch::Device device
) {
    auto z = torch::randn({batch_size, dim}, device);
    return torch::nn::functional::normalize(z);
}

inline void check_cuda_error(const char* file, int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) + " " +
            cudaGetErrorString(err)
        );
    }
}

#define CHECK_CUDA_ERROR() check_cuda_error(__FILE__, __LINE__)

} // namespace test_utils

#include <gtest/gtest.h>
#include <torch/torch.h>
#include "ntxent_kernel.cuh"
#include "test_utils.hpp"

class NTXentBackwardTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available());
        device = torch::Device(torch::kCUDA);
    }

    torch::Device device;
    const double temperature = 0.07;
    const int batch_size = 32;
    const int dim = 128;
};

TEST_F(NTXentBackwardTest, BasicBackward) {
    auto z = torch::randn({batch_size, dim}, 
        torch::TensorOptions().device(device).requires_grad(true));
    z = torch::nn::functional::normalize(z);
    
    auto outputs = ntxent_forward_cuda(z, temperature);
    auto softmax = std::get<0>(outputs);
    auto grad_out = torch::ones_like(softmax);
    
    auto [grad_z, grad_logits] = ntxent_backward_cuda(z, softmax, grad_out, temperature);
    
    EXPECT_FALSE(torch::isnan(grad_z).any().item<bool>());
    EXPECT_FALSE(torch::isnan(grad_logits).any().item<bool>());
}

TEST_F(NTXentBackwardTest, GradientNorm) {
    auto z = torch::randn({batch_size, dim}, 
        torch::TensorOptions().device(device).requires_grad(true));
    z = torch::nn::functional::normalize(z);
    
    auto outputs = ntxent_forward_cuda(z, temperature);
    auto softmax = std::get<0>(outputs);
    auto grad_out = torch::ones_like(softmax);
    
    auto [grad_z, _] = ntxent_backward_cuda(z, softmax, grad_out, temperature);
    
    // Check if gradients are reasonably sized
    auto grad_norm = torch::norm(grad_z);
    EXPECT_LT(grad_norm.item<float>(), 100.0);
    EXPECT_GT(grad_norm.item<float>(), 0.0);
}

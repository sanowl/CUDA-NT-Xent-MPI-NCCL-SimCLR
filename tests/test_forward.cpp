#include <gtest/gtest.h>
#include <torch/torch.h>
#include "ntxent_kernel.cuh"
#include "test_utils.hpp"

class NTXentForwardTest : public ::testing::Test {
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

TEST_F(NTXentForwardTest, BasicForward) {
    auto z = torch::randn({batch_size, dim}, device);
    z = torch::nn::functional::normalize(z);
    
    auto loss = ntxent_forward_cuda(z, temperature);
    
    EXPECT_GT(loss.item<float>(), 0.0);
    EXPECT_FALSE(torch::isnan(loss).item<bool>());
}

TEST_F(NTXentForwardTest, GradientCheck) {
    auto z = torch::randn({batch_size, dim}, 
        torch::TensorOptions().device(device).requires_grad(true));
    z = torch::nn::functional::normalize(z);
    
    auto loss = ntxent_forward_cuda(z, temperature);
    loss.backward();
    
    EXPECT_FALSE(torch::isnan(z.grad()).any().item<bool>());
}

TEST_F(NTXentForwardTest, DifferentBatchSizes) {
    std::vector<int> batch_sizes = {16, 32, 64, 128};
    
    for (int bs : batch_sizes) {
        auto z = torch::randn({bs, dim}, device);
        z = torch::nn::functional::normalize(z);
        
        auto loss = ntxent_forward_cuda(z, temperature);
        
        EXPECT_GT(loss.item<float>(), 0.0);
        EXPECT_FALSE(torch::isnan(loss).item<bool>());
    }
}

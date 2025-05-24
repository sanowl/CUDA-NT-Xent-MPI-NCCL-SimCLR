#include <torch/torch.h>
#include <chrono>
#include <vector>
#include <numeric>
#include <iostream>
#include <iomanip>
#include "ntxent_kernel.cuh"

struct BenchmarkResults {
    double mean_time;
    double std_dev;
    double min_time;
    double max_time;
};

BenchmarkResults run_benchmark(int batch_size, int dim, double temperature, int num_runs) {
    std::vector<double> timings;
    timings.reserve(num_runs);

    // Create input tensor
    auto z = torch::randn({batch_size, dim}, 
        torch::dtype(torch::kFloat32).device(torch::kCUDA));
    z = torch::nn::functional::normalize(z);

    // Warmup run
    auto warmup = ntxent_forward_cuda(z, temperature);
    cudaDeviceSynchronize();

    // Benchmark runs
    for (int i = 0; i < num_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto loss = ntxent_forward_cuda(z, temperature);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start).count() / 1000.0;
        timings.push_back(ms);
    }

    // Calculate statistics
    double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
    double mean = sum / timings.size();

    double sq_sum = std::inner_product(
        timings.begin(), timings.end(), timings.begin(), 0.0,
        std::plus<>(), [](double a, double b) { return (a - mean) * (b - mean); }
    );
    double std_dev = std::sqrt(sq_sum / timings.size());

    auto [min_it, max_it] = std::minmax_element(timings.begin(), timings.end());

    return {mean, std_dev, *min_it, *max_it};
}

int main(int argc, char* argv[]) {
    try {
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA is not available");
        }

        // Print system info
        std::cout << "GPU Device: " << torch::cuda::get_device_name() << "\n";
        std::cout << "CUDA Capability: " 
                  << torch::cuda::get_device_capability() << "\n\n";

        // Benchmark parameters
        const std::vector<int> batch_sizes = {32, 64, 128, 256, 512, 1024};
        const std::vector<int> dimensions = {64, 128, 256};
        const double temperature = 0.07;
        const int num_runs = 100;

        // Run benchmarks
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Batch Size | Dimension | Mean (ms) | Std Dev | Min | Max\n";
        std::cout << std::string(60, '-') << "\n";

        for (int B : batch_sizes) {
            for (int D : dimensions) {
                auto results = run_benchmark(B, D, temperature, num_runs);
                
                std::cout << std::setw(10) << B << " | "
                          << std::setw(9) << D << " | "
                          << std::setw(9) << results.mean_time << " | "
                          << std::setw(8) << results.std_dev << " | "
                          << std::setw(5) << results.min_time << " | "
                          << std::setw(5) << results.max_time << "\n";
            }
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
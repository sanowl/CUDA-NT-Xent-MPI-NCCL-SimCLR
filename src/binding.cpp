#include <torch/extension.h>
#include "ntxent_kernel.cuh"

auto fwd = ntxent_forward_cuda;
auto bwd = ntxent_backward_cuda;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fwd, "NT‑Xent forward (CUDA)");
    m.def("backward", &bwd, "NT‑Xent backward (CUDA)");
}   
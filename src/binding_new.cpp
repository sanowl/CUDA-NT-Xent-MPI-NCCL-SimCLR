#include <torch/extension.h>
#include "ntxent_kernel_new.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &ntxent::ntxent_forward_cuda,
          "NT-Xent forward (CUDA)",
          py::arg("z"),
          py::arg("T"),
          py::arg("use_mixed_precision") = false);
          
    m.def("backward", &ntxent::ntxent_backward_cuda,
          "NT-Xent backward (CUDA)",
          py::arg("z"),
          py::arg("softmax"),
          py::arg("grad_out"),
          py::arg("T"),
          py::arg("use_mixed_precision") = false);
          
    m.def("check_tensor_core_support", &ntxent::utils::check_tensor_core_support,
          "Check if Tensor Cores are supported on the current GPU");
}

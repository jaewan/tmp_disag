#include <torch/extension.h>
#include "remote_device.h"

PYBIND11_MODULE(remote_cuda_ext, m) {
    m.def("register_device", &remote_cuda::register_device, "Register remote CUDA device type with PyTorch");
    
    // Add a simple test function
    m.def("is_available", []() { return true; }, "Check if remote CUDA is available");
}

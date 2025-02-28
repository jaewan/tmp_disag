#include <torch/extension.h>
#include "remote_device.h"
#include "remote_dispatch.h"

// Create the Python module
PYBIND11_MODULE(remote_cuda_ext, m) {
    // Register our device
    remote_cuda::register_device();
    remote_cuda::register_dispatch_keys();

    // Expose the device type to Python
    m.attr("REMOTE_CUDA") = py::int_(static_cast<int>(remote_cuda::REMOTE_CUDA_TYPE));

    // -------- Add any additional bindings here --------
    // Register device with PyTorch
    m.def("register_device", &remote_cuda::register_device, "Register remote CUDA device type with PyTorch");
    
    // Register dispatcher keys
    m.def("register_dispatch_keys", &remote_cuda::register_dispatch_keys, "Register dispatcher keys for remote operations");
}

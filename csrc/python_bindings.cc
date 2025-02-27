#include <torch/extension.h>
#include "remote_device.h"

// Create the Python module
PYBIND11_MODULE(remote_cuda_ext, m) {
    // Register our device
    remote_cuda::register_device();

    // Expose the device type to Python
    m.attr("REMOTE_CUDA") = py::int_(static_cast<int>(remote_cuda::REMOTE_CUDA_TYPE));

    // Add any additional bindings here
}

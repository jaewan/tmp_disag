#include "remote_device.h"
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <iostream>

namespace remote_cuda {

// Register our device guard implementation
// This macro associates our guard implementation with our device type
C10_REGISTER_GUARD_IMPL(PrivateUse1, RemoteCUDAGuardImpl);

void register_device() {
  // Log that we're registering the device
  std::cout << "Registering Remote CUDA device (type ID: "
            << static_cast<int>(REMOTE_CUDA_TYPE)
            << ")" << std::endl;

  // We could add additional initialization here
  // For example, connecting to the remote server
}

}  // namespace remote_cuda

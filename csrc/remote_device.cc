#include "remote_device.h"
#include <iostream>

namespace remote_cuda {

// Register our device guard implementation
// This macro associates our guard implementation with our device type
C10_REGISTER_GUARD_IMPL(PrivateUse1, RemoteCUDAGuardImpl);


void register_device() {
	// Registers REMOTE_CUDA_TYPE as a valid and usable device type within PyTorch's core (C10) runtime. 
	// RuntimeDeviceTypeInit allows PyTorch to manage the memory and other resources associated with REMOTE_CUDA device. 
	// This is essential for proper memory management and to avoid conflicts with other devices.
  //c10::impl::RuntimeDeviceTypeInit(REMOTE_CUDA_TYPE);
	// Register device type
    if (!c10::DeviceType::is_valid_device_type(static_cast<int16_t>(REMOTE_CUDA_TYPE))) {
        c10::DeviceType::register_device_type(static_cast<int16_t>(REMOTE_CUDA_TYPE));
    }

    // Initialize device guard
    c10::impl::CUDAGuardImpl::initializeDeviceStatics();
    
    // Register our device guard implementation
    static RemoteCUDAGuardImpl remote_guard;
    c10::impl::DeviceGuardImplRegistry::registerDeviceGuardImpl(&remote_guard);

  std::cout << "Registering Remote CUDA device (type ID: "
            << static_cast<int>(REMOTE_CUDA_TYPE)
            << ")" << std::endl;

  // We could add additional initialization here
  // For example, connecting to the remote server
}

}  // namespace remote_cuda

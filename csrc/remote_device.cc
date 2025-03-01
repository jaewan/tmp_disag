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
	/*
	c10::impl::PyInterpreterStatus::registerDeviceGuardImplForDeviceType(
			REMOTE_CUDA_TYPE,
			&remote_guard);
			*/

	std::cout << "Registering Remote CUDA device (type ID: "
		<< static_cast<int>(REMOTE_CUDA_TYPE)
		<< ")" << std::endl;

	// We could add additional initialization here
	// For example, connecting to the remote server
}

} // end namespace remote_cuda

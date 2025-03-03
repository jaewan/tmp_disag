#include "remote_device.h"
#include <iostream>

namespace remote_cuda {

// Register our device guard implementation
// This macro associates our guard implementation with our device type
C10_REGISTER_GUARD_IMPL(PrivateUse1, RemoteCUDAGuardImpl);

void register_device() {
	SPDLOG_INFO("Remote CUDA device registration started");

	// We could add additional initialization here
	// For example, connecting to the remote server
}

} // end namespace remote_cuda

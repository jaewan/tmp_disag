import torch
from torch.utils.cpp_extension import load
import importlib.util
import os
import sys

# -------------- Try to load the pre-compiled extension -------------- #
try:
    # Find extension in the same directory as this file
    extension_path = os.path.join(os.path.dirname(__file__), '../remote_cuda_ext.so')
    if os.path.exists(extension_path):
        spec = importlib.util.spec_from_file_location("remote_cuda_ext", extension_path)
        _ext = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_ext)
        print("[DEBUG] Loaded from .so!!")
    else:
        # Fallback to JIT compilation
        print("[DEBUG] Falling back to JIT compilation!!")
        _ext = load(
            name="_remote_cuda_ext",
            sources=[
                "csrc/remote_device.cpp",
                "csrc/remote_dispatch.cpp",
                #"csrc/rpc_client.cpp", 
                #"csrc/memory_manager.cpp",
                "csrc/python_bindings.cpp"
            ],
            extra_include_paths=["./"],
            extra_cflags=["-O3"],
            verbose=True
        )
except Exception as e:
    print(f"Error loading remote_cuda extension: {e}")
    sys.exit(1)
# Constants
REMOTE_CUDA = torch.device("privateuseone")

# Initialize with a no-op function for initial build testing
def init(server_address="localhost:50051"):
    """
    Initialize connection to remote GPU server
    
    Args:
        server_address (str): Address of the remote server (default: "localhost:50051")
        **kwargs: Additional configuration options
            - connection_timeout_ms (int): Connection timeout in milliseconds
            - operation_timeout_ms (int): Operation timeout in milliseconds
            - enable_reconnect (bool): Enable automatic reconnection
            - max_reconnect_attempts (int): Maximum number of reconnection attempts
            - use_compression (bool): Enable data compression
    
    Returns:
        bool: True if connection was successful, False otherwise
    """
    return True

def is_available():
    """Check if remote CUDA is available"""
    # Placeholder implementation
    return True

# Make remote_cuda a module in torch
class RemoteCudaModule:
    def __init__(self):
        self.is_available = is_available
        self.__version__ = "0.1.0"

# Add module to torch namespace
torch.remote_cuda = RemoteCudaModule()

# Initialize the device with PyTorch
_ext.register_device()
_ext.register_dispatch_keys()

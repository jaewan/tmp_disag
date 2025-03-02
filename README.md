# Accelerator Disaggregation framework for Pytorch.

## Features
Custom Device Extension with Unified Dispatch
- Leverages PyTorch's dispatcher system to intercept operations at a high level (do not need to manually implement each tensor operation unlike pure custom device extension)
- Provides proper memory management through the custom device type
- Allows for a clean separation between PyTorch's API and your remote execution logic

## TODO
### Feature
- Operation mapping: map Pytorch ops to remote execution
- Distributed Future implementation
- Memory management strategies for remote device
- Register kernels for specific operations that need special handling on remote device

### Optimization
- Serialization efficiency
- Optimize data transfer to remote GPU using DPDK

### Project Management
- Separate tests to tests directory. Complete tests/BUILD working

### For Fully Open Source
- Make our device type unique. Currently we use PrivateUse1 for prototype. Register a unique device type (This will involve contributing to PyTorch itself)

#pragma once

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/DeviceType.h>

namespace remote_cuda {

// Our device type from remote_device.h
//extern constexpr c10::DeviceType REMOTE_CUDA_TYPE;

// Helper functions for dispatch
void update_stack_with_result(c10::Stack& stack, const at::Tensor& result);

// Register dispatch keys with PyTorch's dispatcher
void register_dispatch_keys();

// Handle specific operation types
//at::Tensor handle_binary_op(const char* op_name, const at::Tensor& self, const at::Tensor& other);
//at::Tensor handle_unary_op(const char* op_name, const at::Tensor& self);
//at::Tensor handle_view_op(const char* op_name, const at::Tensor& self, c10::ArrayRef<int64_t> sizes);

} // namespace remote_cuda

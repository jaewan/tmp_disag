#pragma once

#include "remote_device.h"

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace remote_cuda {

// Register dispatch keys with PyTorch's dispatcher
void register_dispatch_keys();

// Handle specific operation types
//at::Tensor handle_binary_op(const char* op_name, const at::Tensor& self, const at::Tensor& other);
//at::Tensor handle_unary_op(const char* op_name, const at::Tensor& self);
//at::Tensor handle_view_op(const char* op_name, const at::Tensor& self, c10::ArrayRef<int64_t> sizes);

} // namespace remote_cuda

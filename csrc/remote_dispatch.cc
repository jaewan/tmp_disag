#include "remote_dispatch.h"
#include "remote_device.h"
#include <iostream>
#include <unordered_set> //TODO(Jae) change this to absl later

namespace remote_cuda {

// Our device type
extern constexpr c10::DeviceType REMOTE_CUDA_TYPE;

// Set of operations that should NOT be transferred to the remote GPU
// Add more operations as needed, using their full schema name (aten::...)
const std::unordered_set<std::string> kLocalOps = {
    "aten::size",      // Device-agnostic: size of a tensor
    "aten::stride",    // Device-agnostic: stride of a tensor
    "aten::dim",       // Device-agnostic: dimension of a tensor
    "aten::is_cuda",   // Device-agnostic: check if tensor is on CUDA
    "aten::is_cpu",    // Device-agnostic: check if tensor is on CPU
    "aten::device",    // Device-agnostic: gets device of tensor
    "aten::numel",     // Device-agnostic: number of elements in the tensor
    "aten::is_contiguous", //Device-agnostic: check if contiguous

    "aten::item",      // CPU-specific: gets a number from a 1-element tensor
    "aten::to",        // CPU-specific: Moving data. We will reimplement a different way
    "aten::cpu",       // CPU-specific: Move to CPU
    "aten::numpy_T",   // CPU-specific: Convert Tensor to numpy

    "aten::print",     // CPU-specific: Printing
    "aten::println",   // CPU-specific: Printing a new line
    "aten::set_printoptions", // CPU-specific: set printing options

    "aten::empty.memory_format", //CPU-specific, allocate tensor based on cpu

    // Add more operations as needed
};

// Function to execute an operation on the remote server
at::Tensor execute_op_remotely(const c10::OperatorHandle& op, c10::Stack* stack) {
    std::string op_name = op.schema().name();
    std::string overload_name = op.schema().overload_name();

    // 1. Extract tensors and other necessary arguments from the stack
    at::ArrayRef<at::Tensor> tensors = extract_tensors(*stack);

    // 2. Serialize and send the operation and arguments to the remote server
    //at::Tensor result = rpc_client::execute_op(op_name.c_str(), overload_name.c_str(), tensors, *stack);

    // 3. Deserialize the result from the remote server
    update_stack_with_result(*stack, result);

    return result;
}

// Function to execute operation locally
void execute_op_locally(const c10::OperatorHandle& op, c10::Stack* stack) {
    std::cout << "Executing operation locally: " << op.schema().name() << std::endl;

    // Execute the operation locally using op.call(stack)
    op.call(stack);
}

void register_dispatch_keys() {
  auto dispatcher = at::globalContext().dispatchKeyExtractor();

  // Register a catch-all fallback for all operations on REMOTE_CUDA_TYPE
  dispatcher->registerFallback(
    c10::DispatchKeySet({REMOTE_CUDA_TYPE}),
    [](c10::Stack* stack) {
      // Get operator handle from the stack
      const c10::OperatorHandle& op = c10::Dispatcher::currentOp();
      const std::string& op_name = op.schema().name();

      // Check if the operation should be executed locally
      if (kLocalOps.count(op_name)) {
          execute_op_locally(op, stack);
      } else {
          execute_op_remotely(op, stack);
      }
    }
  );
}

} // namespace remote_cuda

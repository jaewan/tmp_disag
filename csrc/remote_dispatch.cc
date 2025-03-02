#include "remote_dispatch.h"

#include <iostream>
#include "absl/container/flat_hash_set.h"

namespace remote_cuda {

at::ArrayRef<at::Tensor> extract_tensors(c10::Stack& stack) {
    std::vector<at::Tensor> tensors;
    for (const c10::IValue& value : stack) { // Iterate through the stack.  Consider only extracting the tensors for now.  This logic needs refined.
        if (value.isTensor()) {
            tensors.push_back(value.toTensor());
        }
    }
    return at::ArrayRef<at::Tensor>(tensors); // Return as at::ArrayRef
}

// Set of operations that should NOT be transferred to the remote GPU
// Add more operations as needed, using their full schema name (aten::...)
const absl::flat_hash_set<std::string> kLocalOps = {
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
    std::cout << "[DEBUG] Executing operation from remote: " << op.schema().name() << std::endl;
    std::string op_name = op.schema().name();
    std::string overload_name = op.schema().overload_name();

    // 1. Extract tensors and other necessary arguments from the stack
    //at::ArrayRef<at::Tensor> tensors = extract_tensors(*stack);

    // 2. Serialize and send the operation and arguments to the remote server
    //at::Tensor result = rpc_client::execute_op(op_name.c_str(), overload_name.c_str(), tensors, *stack);
    at::Tensor result;

    // 3. Deserialize the result from the remote server
    //update_stack_with_result(*stack, result);
                // TODO: Implement remote execution logic here
    // 1. Serialize operation and arguments
    // 2. Send to remote server
    // 3. Receive and deserialize results
    // 4. Update stack with results

    return result;
}

// Function to execute operation locally
void execute_op_locally(const c10::OperatorHandle& op, c10::Stack* stack) {
		SPDLOG_INFO("[DEBUG] Executing operation locally {}",op.schema().name());

		// The correct way to call op locally. Figure out how to do this properly
    //auto kernel = c10::Dispatcher::singleton().findSchema(op.schema());
    //kernel.call(stack);
		//deprecated:op.call(stack) & op.callBoxed(stack) is not preferred in newer pytorch
		op.callBoxed(stack);
}

// Define a boxed fallback function outside the registerFallback call
void remote_cuda_fallback(const c10::OperatorHandle& op, c10::Stack* stack) {
		SPDLOG_INFO("[DEBUG] remote_cuda_fallback called");
    const std::string& op_name = op.schema().name();

    // Check if the operation should be executed locally
    if (kLocalOps.count(op_name)) {
        execute_op_locally(op, stack);
    } else {
        // Move stack to remote_cuda device
        for (c10::IValue& ivalue : *stack) {
            if (ivalue.isTensor()) {
                at::Tensor tensor = ivalue.toTensor();
                if (tensor.device().type() != c10::DeviceType::PrivateUse1) {
                    ivalue = tensor.to(c10::Device(c10::DeviceType::PrivateUse1, 0));
                }
            }
        }
        at::Tensor result = execute_op_remotely(op, stack);
        stack->clear();
        stack->push_back(result);
    }
}

void register_dispatch_keys() {
		SPDLOG_INFO("Register dispatch keys  called");
    auto& dispatcher = c10::Dispatcher::singleton();

    // Create a dispatch key for our device
    c10::DispatchKey remote_cuda_key = c10::DispatchKey::PrivateUse1;

		std::cout << "[DEBUG] register_dispatch_keys called" << std::endl;
    // Register a catch-all fallback for all operations on REMOTE_CUDA_TYPE
    dispatcher.registerFallback(
        remote_cuda_key,
        c10::KernelFunction::makeFromBoxedFunction<&remote_cuda_fallback>(),
        "remote_cuda_fallback"
    );
}

} // namespace remote_cuda

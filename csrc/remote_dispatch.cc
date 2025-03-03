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

//TODO(Jae) complete this
void* remote_allocate(size_t total_bytes){
	return malloc(total_bytes);
}

void register_dispatch_keys() {
	SPDLOG_INFO("Register dispatch keys  called");
	auto& dispatcher = c10::Dispatcher::singleton();

	// Register a catch-all fallback for all operations on REMOTE_CUDA_TYPE
	dispatcher.registerFallback(
			REMOTE_CUDA_KEY,
			c10::KernelFunction::makeFromBoxedFunction<&remote_cuda_fallback>(),
			"remote_cuda_fallback"
			);
}

//----------- Bare Minimum Operations -----------
at::Tensor handle_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride, c10::optional<at::ScalarType> dtype_opt, 
		c10::optional<c10::Layout> layout_opt, c10::optional<c10::Device> device_opt, 
		c10::optional<bool> pin_memory_opt) {
	// Ensure the device is of type REMOTE_CUDA_TYPE
	TORCH_CHECK(device_opt.has_value() && device_opt->type() == REMOTE_CUDA_TYPE, 
			"empty_strided: Expected device of type REMOTE_CUDA_TYPE");

	// 1. Determine the data type
	at::ScalarType scalar_type = dtype_opt.value_or(at::kFloat);

	// Validate layout
	TORCH_CHECK(layout_opt.value_or(c10::kStrided) == c10::kStrided, 
			"empty_strided: Only supports strided layout");
	//TODO For now, we don't handle pinned memory
	TORCH_CHECK(!pin_memory_opt.has_value() || !pin_memory_opt.value(), 
			"empty_strided: Pinned memory is not supported on remote_cuda");

	// 2. Calculate the total size in bytes
	int64_t num_elements = 1;
	for (int64_t dim : size) {
		num_elements *= dim;
	}
	size_t element_size = at::elementSize(scalar_type);
	size_t total_bytes = num_elements * element_size;

	// 3. Allocate memory on the remote device
	//    (This is where you'd communicate with your remote server)
	void* remote_ptr = remote_allocate(total_bytes);

	// 4. Construct a TensorOptions object
	at::TensorOptions options;
	options = options.dtype(scalar_type);
	options = options.layout(layout_opt.value_or(at::kStrided));
	options = options.device(device_opt.value_or(c10::Device(REMOTE_CUDA_TYPE, 0))); // Important: Specify your custom device

	// 5. Create a tensor from the remote memory
	at::Tensor tensor = at::from_blob(remote_ptr, size, stride, options);
	return tensor;
}

} // namespace remote_cuda


//TORCH_LIBRARY_IMPL(aten, c10::DispatchKey::PrivateUse1, m) {
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
		m.impl("empty_strided", remote_cuda::handle_empty_strided);
}

import torch
import remote_cuda

def main():
    print("Testing remote_cuda module:")
    print(f"Remote CUDA is available: {remote_cuda.is_available()}")
    print(f"REMOTE_CUDA device: {remote_cuda.REMOTE_CUDA}")

    
    # Create a simple tensor
    a = torch.tensor([1.0, 2.0, 3.0])
    print(f"Tensor on CPU: {a}")
    
    print("Basic torch integration test:")
    print(f"torch.remote_cuda.is_available(): {torch.remote_cuda.is_available()}")

if __name__ == "__main__":
    main()

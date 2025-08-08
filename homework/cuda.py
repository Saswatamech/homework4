import torch
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Get the name of the current (or first) GPU
    if num_gpus > 0:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU Name: {gpu_name}")

    # Get the CUDA version PyTorch was compiled with
    cuda_version = torch.version.cuda
    print(f"PyTorch CUDA version: {cuda_version}")
    #del BSQPatchAutoEncoder  # Delete unused variables (models, tensors, etc.)
    #del optimizer  # If applicable
    #gc.collect()
    torch.cuda.empty_cache()
else:
    print("CUDA is not available. PyTorch will run on CPU.")
import torch

def get_gpu_memory():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')

    device = torch.device('cuda')
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 3  # in GB
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 3  # in GB
    total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
    return {
        'total_GB': total,
        'allocated_GB': allocated,
        'reserved_GB': reserved,
        'free_GB': total - reserved
    }

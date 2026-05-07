"""Simple script to exercise available GPUs."""

import torch


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        total = torch.cuda.get_device_properties(i).total_mem / 1024**3
        print(f"GPU {i}: {name} ({total:.1f} GiB)")

    # Use GPU 0 for a basic matmul benchmark
    device = torch.device("cuda:0")
    size = 8192

    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    torch.cuda.synchronize()
    a @ b  # warmup
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    c = a @ b
    end.record()
    torch.cuda.synchronize()

    print(f"\nMatmul {size}x{size}: {start.elapsed_time(end):.2f} ms")

    used = torch.cuda.memory_allocated(device) / 1024**3
    print(f"GPU 0 memory used: {used:.2f} GiB")


if __name__ == "__main__":
    main()

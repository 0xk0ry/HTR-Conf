import time
import argparse
import torch
from contextlib import nullcontext
try:
    from torch.profiler import profile, ProfilerActivity
except Exception:
    profile = None
    ProfilerActivity = None

from model.HTR_VT import create_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels-last", action="store_true", help="Use channels_last memory format for model and inputs")
    parser.add_argument("--amp", action="store_true", help="Use autocast mixed precision on CUDA for inference")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile if available (PyTorch 2.x)")
    parser.add_argument("--iters", type=int, default=20, help="Number of timed iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--cudnn-benchmark", action="store_true", help="Enable cudnn.benchmark for potentially faster convs")
    parser.add_argument("--profile", action="store_true", help="Run a short profiler pass and print top ops by time")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(nb_cls=80, img_size=(args.width, args.height))
    # Move to device first, then optionally set channels_last on parameters/buffers
    model.eval().to(device)

    B = args.batch
    H, W = args.height, args.width
    x = torch.randn(B, 1, H, W, device=device)

    # cudnn autotuner may help with fixed input sizes
    if device.type == 'cuda' and args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # Optional torch.compile (PyTorch 2.x)
    if args.compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile failed: {e}")

    # Warmup
    with torch.no_grad():
        for _ in range(max(0, int(args.warmup))):
            if args.amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    _ = model(x)
            else:
                _ = model(x)

    # Optional profiling pass
    if args.profile and profile is not None:
        activities = [ProfilerActivity.CPU]
        if device.type == 'cuda':
            activities.append(ProfilerActivity.CUDA)
        amp_ctx = torch.cuda.amp.autocast() if (args.amp and device.type == 'cuda') else nullcontext()
        with profile(activities=activities, record_shapes=False, profile_memory=True) as prof:
            with torch.no_grad():
                for _ in range(5):
                    with amp_ctx:
                        _ = model(x)
        sort_key = 'self_cuda_time_total' if device.type == 'cuda' else 'self_cpu_time_total'
        print("\n=== Profiler: top 20 ops by {} ===".format('CUDA time' if device.type == 'cuda' else 'CPU time'))
        print(prof.key_averages().table(sort_by=sort_key, row_limit=20))
    elif args.profile and profile is None:
        print("Profiler not available in this PyTorch build; skipping --profile")

    iters = int(args.iters)

    # Memory accounting helpers
    def bytes_to_mb(v):
        return v / (1024 * 1024)

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    # Reset CUDA memory stats before timed run
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_alloc = torch.cuda.memory_allocated()
        start_reserved = torch.cuda.memory_reserved()

    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters):
            if args.amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    _ = model(x)
            else:
                _ = model(x)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    dt = (time.time() - t0) / iters

    print(f"Device: {device}, Batch: {B}, Input: {H}x{W}, channels_last={args.channels_last}, amp={args.amp} -> avg {dt*1000:.2f} ms/iter")

    # Report memory usage
    print("Model params: {:.2f} MB, buffers: {:.2f} MB".format(bytes_to_mb(param_bytes), bytes_to_mb(buffer_bytes)))
    if device.type == 'cuda':
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        end_alloc = torch.cuda.memory_allocated()
        end_reserved = torch.cuda.memory_reserved()
        print(
            "CUDA mem (MB) -> start alloc/res: {:.2f}/{:.2f}, peak alloc/res: {:.2f}/{:.2f}, end alloc/res: {:.2f}/{:.2f}".format(
                bytes_to_mb(start_alloc), bytes_to_mb(start_reserved),
                bytes_to_mb(peak_alloc), bytes_to_mb(peak_reserved),
                bytes_to_mb(end_alloc), bytes_to_mb(end_reserved),
            )
        )
    else:
        # CPU RSS via psutil if available
        try:
            import psutil, os
            rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            print(f"CPU RSS approx: {rss_mb:.2f} MB")
        except Exception:
            print("psutil not available; skipping CPU RSS report")

if __name__ == "__main__":
    main()

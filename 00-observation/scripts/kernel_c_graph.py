import torch
import time
import argparse

def busy_launch_graph(iters):
    a = torch.ones(32, device='cuda')
    torch.cuda.synchronize()

    # Warm-up (required before capture — CUDA graphs need a stable
    # memory/kernel state to record against)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            a.add_(1.0)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # Capture: record the op sequence once
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        a.add_(1.0)

    # Replay: fire the pre-recorded graph, no per-iter Python/launch overhead
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"C (graph): {elapsed:.4f}s over {iters} iters "
          f"({elapsed/iters*1e6:.2f} us/iter)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=2_000_000)
    args = parser.parse_args()
    busy_launch_graph(args.iters)

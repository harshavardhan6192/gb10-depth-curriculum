import torch
import time
import argparse

def busy_launch(iters):
    a = torch.ones(32, device='cuda')
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a.add_(1.0)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"C (launch): {elapsed:.4f}s over {iters} iters "
          f"({elapsed/iters*1e6:.2f} us/iter)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=2_000_000)
    args = parser.parse_args()
    busy_launch(args.iters)

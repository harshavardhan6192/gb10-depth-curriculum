import torch
import argparse

def matmul(n):
    a = torch.randn(n, n, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(n, n, device='cuda', dtype=torch.bfloat16)
    torch.cuda.synchronize()
    c = a @ b
    torch.cuda.synchronize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2048)
    args = parser.parse_args()
    matmul(args.n)

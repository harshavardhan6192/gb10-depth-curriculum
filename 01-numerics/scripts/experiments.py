import torch

def cancellation_test(dtype):
    a = torch.tensor(1e6, dtype=dtype)
    b = torch.tensor(1e-3, dtype=dtype)
    result = (a + b) - a
    print(f"{dtype}: (a+b)-a = {result.item()}")

def accumulate_test(accum_dtype, n=10000):
    val = torch.tensor(1.0, dtype=torch.bfloat16)
    acc = torch.tensor(0.0, dtype=accum_dtype)
    for _ in range(n):
        acc = (acc + val.to(accum_dtype)).to(accum_dtype)
    print(f"accumulate in {accum_dtype}: result={acc.item()}, expected={n}")

if __name__ == "__main__":
    print("--- Experiment 1: Catastrophic cancellation ---")
    for dtype in [torch.float64, torch.float32, torch.bfloat16, torch.float16]:
        cancellation_test(dtype)

    print("\n--- Experiment 3: Accumulation precision ---")
    accumulate_test(torch.float32)
    accumulate_test(torch.bfloat16)

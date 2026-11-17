import torch

def derive_properties(exp_bits: int, mant_bits: int, name: str):
    bias = 2**(exp_bits - 1) - 1
    max_exp = 2**exp_bits - 2
    min_exp = 1 - bias

    max_mantissa_frac = (2**mant_bits - 1) / 2**mant_bits
    max_val = (1 + max_mantissa_frac) * 2**(max_exp - bias)
    min_normal = 1.0 * 2**min_exp
    min_subnormal = (1 / 2**mant_bits) * 2**min_exp
    eps_theory = 2**(-mant_bits)

    print(f"--- {name} (exp={exp_bits}, mant={mant_bits}) ---")
    print(f"  max value:      {max_val:.6g}")
    print(f"  min normal:     {min_normal:.6g}")
    print(f"  min subnormal:  {min_subnormal:.6g}")
    print(f"  eps (theory):   {eps_theory:.6g}")

def find_epsilon(dtype):
    x = torch.tensor(1.0, dtype=dtype)
    eps = torch.tensor(1.0, dtype=dtype)
    while (x + eps != x):
        last_good = eps.clone()
        eps = eps / 2
    return last_good.item()

if __name__ == "__main__":
    derive_properties(8, 23, "FP32")
    derive_properties(8, 7,  "BF16")
    derive_properties(5, 10, "FP16")
    derive_properties(4, 3,  "FP8 E4M3")
    derive_properties(5, 2,  "FP8 E5M2")

    print("\n--- Empirical epsilon check ---")
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        print(dtype, find_epsilon(dtype))

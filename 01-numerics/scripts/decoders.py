import struct
import torch

def decode_fp32(x: float):
    bits = struct.unpack('>I', struct.pack('>f', x))[0]
    sign = (bits >> 31) & 1
    exp  = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    return sign, exp, mant

def decode_fp16(x: float):
    bits = struct.unpack('>H', struct.pack('>e', x))[0]
    sign = (bits >> 15) & 1
    exp  = (bits >> 10) & 0x1F
    mant = bits & 0x3FF
    return sign, exp, mant

def decode_bf16(x: float):
    t = torch.tensor(x, dtype=torch.bfloat16)
    bits = t.view(torch.uint16).item()
    sign = (bits >> 15) & 1
    exp  = (bits >> 7) & 0xFF
    mant = bits & 0x7F
    return sign, exp, mant

def decode_fp8_e4m3(x: float):
    t = torch.tensor(x, dtype=torch.float8_e4m3fn)
    bits = t.view(torch.uint8).item()
    sign = (bits >> 7) & 1
    exp  = (bits >> 3) & 0xF
    mant = bits & 0x7
    return sign, exp, mant

def decode_fp8_e5m2(x: float):
    t = torch.tensor(x, dtype=torch.float8_e5m2)
    bits = t.view(torch.uint8).item()
    sign = (bits >> 7) & 1
    exp  = (bits >> 2) & 0x1F
    mant = bits & 0x3
    return sign, exp, mant

if __name__ == "__main__":
    test_val = 1.5
    for name, fn in [("fp32", decode_fp32), ("fp16", decode_fp16),
                      ("bf16", decode_bf16), ("fp8_e4m3", decode_fp8_e4m3),
                      ("fp8_e5m2", decode_fp8_e5m2)]:
        print(name, fn(test_val))

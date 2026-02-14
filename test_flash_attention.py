"""
Correctness + benchmark for the FlashAttention kernel.
JIT-compiles flash_attention.cu and compares against PyTorch SDPA.

Usage: python test_flash_attention.py
       (first run takes 1-2min for JIT compile)
"""

import torch
import torch.nn.functional as F
import torch.utils.cpp_extension
import os

print("Compiling CUDA kernel (JIT)...")

cuda_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flash_attention.cu")

wrapper = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>

void flash_attention_forward(
    const half* Q, const half* K, const half* V, half* O,
    int batch_heads, int N, int d, float scale);

void naive_attention_forward(
    const half* Q, const half* K, const half* V, half* O,
    float* S_buf, int batch_heads, int N, int d, float scale);

torch::Tensor flash_attn_fwd(torch::Tensor Q, torch::Tensor K, torch::Tensor V, float scale) {
    int BH = Q.size(0), N = Q.size(1), D = Q.size(2);
    auto O = torch::zeros_like(Q);
    flash_attention_forward(
        (const half*)Q.data_ptr(), (const half*)K.data_ptr(),
        (const half*)V.data_ptr(), (half*)O.data_ptr(), BH, N, D, scale);
    return O;
}

torch::Tensor naive_attn_fwd(torch::Tensor Q, torch::Tensor K, torch::Tensor V, float scale) {
    int BH = Q.size(0), N = Q.size(1), D = Q.size(2);
    auto O = torch::zeros_like(Q);
    auto S = torch::zeros({BH, N, N}, torch::dtype(torch::kFloat32).device(Q.device()));
    naive_attention_forward(
        (const half*)Q.data_ptr(), (const half*)K.data_ptr(),
        (const half*)V.data_ptr(), (half*)O.data_ptr(),
        (float*)S.data_ptr(), BH, N, D, scale);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_fwd", &flash_attn_fwd);
    m.def("naive_attn_fwd", &naive_attn_fwd);
}
"""

wrapper_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_wrapper.cpp")
with open(wrapper_path, "w") as f:
    f.write(wrapper)

ext = torch.utils.cpp_extension.load(
    name="flash_attn_ext",
    sources=[wrapper_path, cuda_src],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89"],
    verbose=True,
)
print("Compile done.\n")


# -- correctness --

def test_correctness(bh=4, n=256, d=128):
    torch.manual_seed(42)
    Q = torch.randn(bh, n, d, device="cuda", dtype=torch.float16)
    K = torch.randn(bh, n, d, device="cuda", dtype=torch.float16)
    V = torch.randn(bh, n, d, device="cuda", dtype=torch.float16)
    scale = 1.0 / (d ** 0.5)

    ref = F.scaled_dot_product_attention(Q, K, V)

    torch.cuda.synchronize()
    flash_out = ext.flash_attn_fwd(Q, K, V, scale)
    torch.cuda.synchronize()
    naive_out = ext.naive_attn_fwd(Q, K, V, scale)
    torch.cuda.synchronize()

    d_flash = (ref - flash_out).abs()
    d_naive = (ref - naive_out).abs()

    flash_ok = torch.allclose(ref, flash_out, atol=1e-2, rtol=1e-2)
    naive_ok = torch.allclose(ref, naive_out, atol=1e-2, rtol=1e-2)

    print(f"  bh={bh} n={n} d={d}")
    print(f"    flash: max_err={d_flash.max().item():.6f}  mean={d_flash.mean().item():.6f}  {'PASS' if flash_ok else 'FAIL'}")
    print(f"    naive: max_err={d_naive.max().item():.6f}  mean={d_naive.mean().item():.6f}  {'PASS' if naive_ok else 'FAIL'}")

    return flash_ok and naive_ok


# -- benchmark --

def benchmark(bh=32, d=128, seq_lens=[512, 1024, 2048], runs=20, warmup=5):
    scale = 1.0 / (d ** 0.5)
    results = []

    for n in seq_lens:
        Q = torch.randn(bh, n, d, device="cuda", dtype=torch.float16)
        K = torch.randn(bh, n, d, device="cuda", dtype=torch.float16)
        V = torch.randn(bh, n, d, device="cuda", dtype=torch.float16)

        def time_fn(fn, label):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            times = []
            for _ in range(runs):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record(); fn(); e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
            avg = sum(times) / len(times)
            return avg

        # naive (skip if S_buf would OOM)
        s_buf_bytes = bh * n * n * 4
        free, _ = torch.cuda.mem_get_info()
        naive_ms = None
        if s_buf_bytes < free * 0.8:
            naive_ms = time_fn(lambda: ext.naive_attn_fwd(Q, K, V, scale), "naive")

        flash_ms = time_fn(lambda: ext.flash_attn_fwd(Q, K, V, scale), "flash")
        sdpa_ms = time_fn(lambda: F.scaled_dot_product_attention(Q, K, V), "sdpa")

        results.append({"n": n, "naive": naive_ms, "flash": flash_ms, "sdpa": sdpa_ms})

        naive_str = f"{naive_ms:.2f}" if naive_ms else "OOM"
        speedup = f"{naive_ms / flash_ms:.2f}x" if naive_ms else "-"
        print(f"  n={n:<6} naive={naive_str:<10} flash={flash_ms:<10.2f} sdpa={sdpa_ms:<10.2f} flash/naive={speedup}")

    return results


def dram_analysis(seq_lens=[512, 1024, 2048], bh=32, d=128):
    """Theoretical DRAM traffic comparison."""
    print(f"\n{'n':<8} {'standard (MB)':<16} {'tiled (MB)':<16} {'reduction':<12}")
    print("-" * 52)
    for n in seq_lens:
        std = bh * (3 * n * d * 2 + 2 * n * n * 4 + n * d * 2)
        n_qt = (n + 63) // 64
        n_kvt = (n + 63) // 64
        tiled = bh * (n * d * 2 + n_kvt * 64 * d * 2 * n_qt * 2 + n * d * 2)
        print(f"{n:<8} {std/1024**2:<16.1f} {tiled/1024**2:<16.1f} {(1 - tiled/std)*100:<12.1f}%")


if __name__ == "__main__":
    print("=" * 60)
    print("Correctness")
    print("=" * 60)
    ok = True
    for n in [128, 256, 512, 1024]:
        if not test_correctness(bh=4, n=n, d=128):
            ok = False
    print(f"\n{'ALL PASSED' if ok else 'SOME FAILED'}\n")

    print("=" * 60)
    print("Benchmark (bh=32, d=128)")
    print("=" * 60)
    benchmark()

    print("\n" + "=" * 60)
    print("Theoretical DRAM traffic (bh=32, d=128)")
    print("=" * 60)
    dram_analysis()

    print("\nNext: profile with ncu --set full --kernel-name flash_attn_fwd ./bench")

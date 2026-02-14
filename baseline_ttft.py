"""
Baseline TTFT measurement for Llama 3 8B INT4 (AWQ) on RTX 4060 Ti.
Measures prefill latency with standard attention as reference.
"""

import torch
import time
import sys


def check_env():
    print("=" * 60)
    print("Environment Check")
    print("=" * 60)

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("No CUDA device found. Exiting.")
        sys.exit(1)

    print(f"CUDA version: {torch.version.cuda}")

    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"\nGPU: {props.name}")
    print(f"SMs: {props.multi_processor_count}")
    print(f"VRAM: {props.total_memory / 1024**3:.1f} GB")
    print(f"Compute capability: {cap[0]}.{cap[1]}")

    free, total = torch.cuda.mem_get_info()
    print(f"VRAM free: {free / 1024**3:.1f} / {total / 1024**3:.1f} GB\n")

    return props


def load_model():
    print("=" * 60)
    print("Loading Llama 3 8B INT4 (AWQ)")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "TechxGenus/Meta-Llama-3-8B-Instruct-AWQ"
    print(f"Model: {model_id}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda"
    )
    print(f"Loaded in {time.time() - t0:.1f}s")

    free, total = torch.cuda.mem_get_info()
    print(f"VRAM used: {(total - free) / 1024**3:.1f} GB\n")

    return model, tokenizer


def measure_ttft(model, tokenizer, seq_len, num_runs=5, warmup=2):
    """Measure time-to-first-token (prefill latency) via max_new_tokens=1."""
    dummy = "Hello " * (seq_len // 2)
    inputs = tokenizer(
        dummy, return_tensors="pt",
        max_length=seq_len, truncation=True, padding="max_length",
    ).to("cuda")

    actual_len = inputs["input_ids"].shape[1]

    for _ in range(warmup):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1, do_sample=False)
    torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1, do_sample=False)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return {
        "seq_len": actual_len,
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def measure_throughput(model, tokenizer, seq_len=512, gen_tokens=100, num_runs=3):
    """Measure decode throughput in tokens/sec."""
    dummy = "Hello " * (seq_len // 2)
    inputs = tokenizer(
        dummy, return_tensors="pt",
        max_length=seq_len, truncation=True,
    ).to("cuda")

    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10, do_sample=False)
    torch.cuda.synchronize()

    results = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=gen_tokens, do_sample=False)
        end.record()
        torch.cuda.synchronize()
        generated = out.shape[1] - inputs["input_ids"].shape[1]
        results.append(generated / (start.elapsed_time(end) / 1000))

    return sum(results) / len(results)


if __name__ == "__main__":
    props = check_env()
    model, tokenizer = load_model()

    print("=" * 60)
    print("Baseline TTFT (standard attention, no FlashAttention)")
    print("=" * 60)

    results = []
    for sl in [512, 1024, 2048]:
        print(f"\nseq_len={sl}...")
        r = measure_ttft(model, tokenizer, sl)
        results.append(r)
        print(f"  len={r['seq_len']}  avg={r['avg_ms']:.1f}ms  "
              f"min={r['min_ms']:.1f}ms  max={r['max_ms']:.1f}ms")

    print(f"\n{'=' * 60}")
    print("Decode throughput")
    print("=" * 60)
    tps = measure_throughput(model, tokenizer)
    print(f"  {tps:.1f} tok/s (INT4 AWQ)\n")

    print("=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)
    print(f"GPU:       {props.name} ({props.multi_processor_count} SMs)")
    print(f"Model:     Llama 3 8B INT4 (AWQ)")
    print(f"Attention: standard (baseline)\n")
    print(f"{'seq_len':<10} {'avg (ms)':<12} {'min (ms)':<12}")
    print("-" * 34)
    for r in results:
        print(f"{r['seq_len']:<10} {r['avg_ms']:<12.1f} {r['min_ms']:<12.1f}")
    print(f"\nthroughput: {tps:.1f} tok/s")
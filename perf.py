#!/usr/bin/env python3
"""
Compute theoretical peak FLOPS and memory bandwidth for your CUDA GPU.

Requirements:
    pip install pycuda

If pycuda is not available, it will try to use `nvidia-smi` as fallback.
"""

import subprocess

def get_gpu_info_pycuda():
    import pycuda.driver as cuda
    cuda.init()
    dev = cuda.Device(0)
    attrs = dev.get_attributes()

    sm_count = attrs[cuda.device_attribute.MULTIPROCESSOR_COUNT]
    clock_rate = attrs[cuda.device_attribute.CLOCK_RATE] * 1e-3  # kHz → MHz
    bus_width = attrs[cuda.device_attribute.GLOBAL_MEMORY_BUS_WIDTH]  # bits
    mem_clock = attrs[cuda.device_attribute.MEMORY_CLOCK_RATE] * 1e-3  # kHz → MHz
    name = dev.name()

    # Rough heuristic for cores per SM based on architecture
    # (see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)
    cc_major, cc_minor = dev.compute_capability()
    cores_per_sm_table = {
        5: 128, 6: 64, 7: 64, 8: 128, 9: 128
    }
    cores_per_sm = cores_per_sm_table.get(cc_major, 64)

    # FLOPS (FP32)
    flops = 2 * sm_count * cores_per_sm * (clock_rate * 1e6)  # FMA: 2 ops/cycle
    tflops = flops / 1e12

    # Bandwidth (bytes/s)
    bw = (bus_width / 8) * (mem_clock * 1e6) * 2  # DDR: 2 transfers/clock
    gbps = bw / 1e9

    return {
        "name": name,
        "compute_capability": f"{cc_major}.{cc_minor}",
        "SMs": sm_count,
        "cores_per_SM": cores_per_sm,
        "core_clock_MHz": clock_rate,
        "mem_clock_MHz": mem_clock,
        "bus_width_bits": bus_width,
        "FP32_TFLOPS": tflops,
        "mem_bandwidth_GBps": gbps,
    }

def get_gpu_info_nvidia_smi():
    out = subprocess.check_output(["nvidia-smi", "-q"], text=True)
    def find(label):
        import re
        m = re.search(label + r".*?:\s*([\d.]+)", out)
        return float(m.group(1)) if m else None

    clock = find(r"Graphics") or 1500  # MHz
    mem_clock = find(r"Memory") or 7000
    bus_width = find(r"Bus Width") or 256
    cores = find(r"CUDA Cores") or 4096

    flops = 2 * cores * clock * 1e6
    bw = (bus_width / 8) * mem_clock * 1e6 
    return {
        "name": "GPU (nvidia-smi)",
        "FP32_TFLOPS": flops / 1e12,
        "mem_bandwidth_GBps": bw / 1e9,
        "cores": cores,
        "clock_MHz": clock,
        "mem_clock_MHz": mem_clock,
        "bus_width_bits": bus_width,
    }

if __name__ == "__main__":
    try:
        info = get_gpu_info_pycuda()
    except Exception as e:
        print("⚠️ pycuda not available, using nvidia-smi fallback:", e)
        info = get_gpu_info_nvidia_smi()

    print("\n=== GPU Theoretical Performance ===")
    for k, v in info.items():
        if isinstance(v, float):
            print(f"{k:20s}: {v:.2f}")
        else:
            print(f"{k:20s}: {v}")

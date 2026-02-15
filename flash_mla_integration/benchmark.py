import subprocess
import time
import re
import os
import csv
import shutil
from datetime import datetime
from typing import Tuple, Optional

# ================= 配置区域 =================
MODEL_PATH = "deepseek-ai/DeepSeek-V2-Lite-Chat"
GPU_MEM_UTIL = "0.9"
ENV_VAR_NAME = "DISABLE_FLASH_MLA" 
TRITON_CACHE_DIR = os.path.expanduser("~/.triton/cache")
MAX_MODEL_LEN = 9000
GPU_COOLDOWN_SECONDS = 5

# 测试配置: (Input, Output, BatchSizes)
DECODE_FOCUSED_CONFIGS = [
    (1024, 256, [1, 4, 8, 16, 32]),
    (2048, 256, [1, 4, 8, 16, 32]),
    (4096, 256, [1, 4, 8, 16, 32]),
    (8192, 256, [1, 4, 8, 16, 32]),
]

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILE = f"flashmla_bench_{TIMESTAMP}.csv"
LOG_DIR = f"logs_{TIMESTAMP}"

# ================= 工具函数 =================
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

os.makedirs(LOG_DIR, exist_ok=True)

def clear_triton_cache():
    if os.path.exists(TRITON_CACHE_DIR):
        try:
            shutil.rmtree(TRITON_CACHE_DIR)
            os.makedirs(TRITON_CACHE_DIR, exist_ok=True)
        except: pass

def run_vllm_throughput(disable_custom: bool, kv_dtype: str, in_len: int, out_len: int, bs: int, is_warmup: bool = False) -> Optional[str]:
    """
    kv_dtype: "auto" (对应 bf16) 或 "int8"
    disable_custom: True=Base, False=FlashMLA
    """
    env = os.environ.copy()
    env[ENV_VAR_NAME] = "1" if disable_custom else "0"
    
    cmd = [
        "vllm", "bench", "throughput",
        "--model", MODEL_PATH,
        "--dataset-name", "random",
        "--random-input-len", str(in_len),
        "--random-output-len", str(out_len),
        "--random-range-ratio", "0.0",
        "--num-prompts", str(bs),
        "--max-num-seqs", str(bs), 
        "--max-model-len", str(MAX_MODEL_LEN),
        "--trust-remote-code",
        "--dtype", "bfloat16",        # ✨ 统一使用 bfloat16 计算
        "--kv-cache-dtype", kv_dtype, # ✨ 控制 KV 类型 (auto/int8)
        "--gpu-memory-utilization", GPU_MEM_UTIL,
        "--disable-log-stats",
        "--no-enable-chunked-prefill"
    ]
    
    # 如果你的 checkpoint 里没有 scale，这里可能需要加上 --quantization-param-path
    # cmd.extend(["--quantization-param-path", "kv_scales.json"])

    try:
        res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if res.returncode != 0:
            return f"ERROR: {res.stdout[:1000]}"
        return res.stdout
    except Exception as e:
        return f"CRASH: {str(e)}"

def parse_info(output: Optional[str]) -> dict:
    info = {
        "tps": 0.0,
        "mode": "N/A",
        "scales": "" # 默认为空
    }

    if not output or any(x in output for x in ["ERROR", "CRASH", "out of memory"]): 
        info["mode"] = "Err/OOM"
        return info
    
    # 1. 解析 TPS
    m = re.search(r"([\d\.]+)\s+output tokens/s", output)
    if not m: m = re.search(r"Throughput:\s+([\d\.]+)", output)
    if m: info["tps"] = float(m.group(1))

    # 2. 解析 FlashMLA Config
    if "[FlashMLA-Config]" in output:
        # 解析 Splits
        m_split = re.search(r"NumSplits=(\d+)", output)
        if m_split:
            info["mode"] = f"Split-{m_split.group(1)}"
        
        # 解析 Scales (如果有)
        m_scales = re.search(r"Scales=\(k=([\d\.]+),\s*v=([\d\.]+)\)", output)
        if m_scales:
            k, v = m_scales.group(1), m_scales.group(2)
            # 如果是默认值 1.0，可以简化显示，或者始终显示
            info["scales"] = f"Scale:[{k}/{v}]"
    else:
        info["mode"] = "Base"
    
    return info

def main():
    print(f"{BOLD}FlashMLA Benchmark (BF16 Compute, MaxLen={MAX_MODEL_LEN}){RESET}")
    print("=" * 110)
    # 表头去掉了冗余的 INT8 状态列
    print(f"{'In':<5} {'Out':<5} {'BS':<4} | {'Base TPS':<9} | {'BF16 TPS':<9} {'Speedup':<9} | {'INT8 TPS':<9} {'Speedup':<9} | {'Config Info'}")
    print("-" * 110)

    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["In", "Out", "BS", "Base_TPS", "BF16_TPS", "BF16_Speedup", "INT8_TPS", "INT8_Speedup", "Config"])
        
        for in_len, out_len, batch_sizes in DECODE_FOCUSED_CONFIGS:
            for bs in batch_sizes:
                
                # 1. Base (vLLM Official)
                run_vllm_throughput(True, "auto", in_len, out_len, bs, is_warmup=True)
                out_base = run_vllm_throughput(True, "auto", in_len, out_len, bs)
                info_base = parse_info(out_base)
                base_tps = info_base["tps"]
                
                time.sleep(GPU_COOLDOWN_SECONDS)

                # 2. Ours (BF16 KV)
                clear_triton_cache()
                run_vllm_throughput(False, "auto", in_len, out_len, bs, is_warmup=True)
                out_fp16 = run_vllm_throughput(False, "auto", in_len, out_len, bs)
                info_fp16 = parse_info(out_fp16)
                fp16_tps = info_fp16["tps"]
                
                fp16_speedup = fp16_tps / base_tps if base_tps > 0 else 0
                fp16_color = GREEN if fp16_speedup >= 1.05 else RED if fp16_speedup < 0.95 else ""
                
                time.sleep(GPU_COOLDOWN_SECONDS)

                # 3. Ours (INT8 KV)
                clear_triton_cache()
                run_vllm_throughput(False, "int8", in_len, out_len, bs, is_warmup=True)
                out_int8 = run_vllm_throughput(False, "int8", in_len, out_len, bs)
                info_int8 = parse_info(out_int8)
                int8_tps = info_int8["tps"]

                int8_speedup = int8_tps / base_tps if base_tps > 0 else 0
                int8_color = GREEN if int8_speedup >= 1.05 else RED if int8_speedup < 0.95 else ""

                # Meta Info (取 INT8 的 config，因为通常包含 Scales)
                split_info = info_int8["mode"]
                scales_info = info_int8["scales"]
                meta_info = f"{split_info} {scales_info}".strip()

                print(f"{in_len:<5} {out_len:<5} {bs:<4} | "
                      f"{base_tps:<9.1f} | "
                      f"{fp16_tps:<9.1f} {fp16_color}{fp16_speedup:>8.2f}x{RESET} | "
                      f"{int8_tps:<9.1f} {int8_color}{int8_speedup:>8.2f}x{RESET} | "
                      f"{CYAN}{meta_info}{RESET}")
                
                writer.writerow([
                    in_len, out_len, bs, 
                    base_tps, 
                    fp16_tps, f"{fp16_speedup:.2f}x", 
                    int8_tps, f"{int8_speedup:.2f}x", 
                    meta_info
                ])
                f.flush()
                
                time.sleep(GPU_COOLDOWN_SECONDS)

    print(f"\n{BOLD}Results saved to {CSV_FILE}{RESET}")

if __name__ == "__main__":
    main()
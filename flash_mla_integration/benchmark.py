import subprocess
import time
import re
import os
import csv
import shutil
from datetime import datetime
from typing import Tuple, Optional

MODEL_PATH = "deepseek-ai/DeepSeek-V2-Lite-Chat"
GPU_MEM_UTIL = "0.9"
ENV_VAR_NAME = "DISABLE_FLASH_MLA" 
TRITON_CACHE_DIR = os.path.expanduser("~/.triton/cache")
MAX_MODEL_LEN = 9000 # limit GPU memory
GPU_COOLDOWN_SECONDS = 5

DECODE_FOCUSED_CONFIGS = [
    (1024, 256, [1, 4, 8, 16, 32]),
    (2048, 256, [1, 4, 8, 16, 32]),
    (4096, 256, [1, 4, 8, 16, 32]),
    (8192, 256, [1, 4, 8, 16, 32]),
]

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILE = f"decode_bench_{TIMESTAMP}.csv"
LOG_DIR = f"decode_logs_{TIMESTAMP}"

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

def run_vllm_throughput(disable_custom: bool, in_len: int, out_len: int, bs: int, is_warmup: bool = False) -> Optional[str]:
    env = os.environ.copy()
    env[ENV_VAR_NAME] = "1" if disable_custom else "0"
    
    current_out_len = 10 if is_warmup else out_len
    
    cmd = [
        "vllm", "bench", "throughput",
        "--model", MODEL_PATH,
        
        "--dataset-name", "random",
        "--random-input-len", str(in_len),
        "--random-output-len", str(current_out_len),
        "--random-range-ratio", "0.0",
        
        "--num-prompts", str(bs),
        "--max-num-seqs", str(bs), 
        "--max-model-len", str(MAX_MODEL_LEN),
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", GPU_MEM_UTIL,
        "--disable-log-stats",
    ]
    
    try:
        res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if res.returncode != 0:
            return f"ERROR: {res.stdout[:500]}"
        return res.stdout
    except Exception as e:
        return f"CRASH: {str(e)}"

def parse_throughput(output: Optional[str]) -> Tuple[float, str, str]:
    if not output or any(x in output for x in ["ERROR", "CRASH", "out of memory"]): 
        return 0.0, "Err/OOM", "N/A"
    
    mode = "Base"
    m = re.search(r"\[FlashMLA-Config\].*?NumSplits=(\d+)", output)
    if m:
      splits = int(m.group(1))
      mode = f"Split-{splits}"

    raw_matches = re.findall(r"\[FlashMLA\] (.*?): \{(.*?)\}", output)
    
    parsed_configs = {}
    for name, cfg_body in raw_matches:
        bn = re.search(r"BLOCK_N=(\d+)", cfg_body)
        ws = re.search(r"warps=(\d+)", cfg_body)
        stg = re.search(r"stages=(\d+)", cfg_body)
        
        parts = []
        if bn: parts.append(f"BN{bn.group(1)}")
        if ws: parts.append(f"W{ws.group(1)}")
        if stg: parts.append(f"S{stg.group(1)}")
        
        parsed_configs[name] = "/".join(parts)

    if "Fused Path" in parsed_configs:
        best_config = f"F:{parsed_configs['Fused Path']}"
    elif "Split-K Stage 1" in parsed_configs:
        s1 = parsed_configs.get("Split-K Stage 1", "N/A")
        s2 = parsed_configs.get("Split-K Stage 2", "N/A")
        best_config = f"S1:{s1} | S2:{s2}"
    else:
        best_config = "No AutoTune Config"
    
    val = 0.0
    m = re.search(r"([\d\.]+)\s+output tokens/s", output)
    if not m: m = re.search(r"Throughput:\s+([\d\.]+)", output)
    if m: val = float(m.group(1))
    
    return val, mode, best_config

def main():
    print(f"{BOLD}FlashMLA Benchmark (MaxLen={MAX_MODEL_LEN}){RESET}")
    print("=" * 100)
    print(f"{'In':<6} {'Out':<6} {'BS':<5} {'Mode':<10} {'Base TPS':<12} {'Ours TPS':<12} {'Speedup':<15} {'Best Config'}")
    print("-" * 100)

    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["In", "Out", "BS", "Mode", "Base_TPS", "Ours_TPS", "Speedup", "Config"])
        
        for in_len, out_len, batch_sizes in DECODE_FOCUSED_CONFIGS:
            for bs in batch_sizes:
                
                # vllm
                # warmup
                run_vllm_throughput(True, in_len, out_len, bs, is_warmup=True)
                
                # real
                out_base = run_vllm_throughput(True, in_len, out_len, bs)
                base_tps, _, _ = parse_throughput(out_base)

                time.sleep(GPU_COOLDOWN_SECONDS)

                # ours
                # warmup
                clear_triton_cache()
                warmup_out = run_vllm_throughput(False, in_len, out_len, bs, is_warmup=True)
                _, last_mode, best_config = parse_throughput(warmup_out)

                # real
                out_ours = run_vllm_throughput(False, in_len, out_len, bs)
                ours_tps, _, _ = parse_throughput(out_ours)

                speedup = ours_tps / base_tps if base_tps > 0 else 0

                color = GREEN if speedup >= 1.05 else RED if speedup < 0.95 else ""
                speedup_str = f"{color}{speedup:>7.2f}x{RESET}"
                mode_str = f"{CYAN}{last_mode}{RESET}"
                
                print(f"{in_len:<6} {out_len:<6} {bs:<5} {mode_str:<19} {base_tps:<12.1f} {ours_tps:<12.1f} {speedup_str:<24} {best_config}")
                
                writer.writerow([in_len, out_len, bs, last_mode, base_tps, ours_tps, f"{speedup:.2f}x", best_config])
                f.flush()
                
                time.sleep(GPU_COOLDOWN_SECONDS)

    print(f"\n{BOLD}Results saved to {CSV_FILE}{RESET}")

if __name__ == "__main__":
    main()
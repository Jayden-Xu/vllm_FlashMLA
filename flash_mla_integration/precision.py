import subprocess
import os
import json
import csv
import numpy as np
import sys

MODEL_PATH = "deepseek-ai/DeepSeek-V2-Lite-Chat"
ENV_VAR_NAME = "DISABLE_FLASH_MLA"

TEST_CASES = [
    {"name": "Short",  "in_len": 1024,  "bs": 1},
    {"name": "Medium", "in_len": 2048, "bs": 8},
    # {"name": "Long",   "in_len": 4096, "bs": 32}, # 可选，跑太久可以注释掉
]

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

HELPER_SCRIPT = "precision_worker.py"
OUTPUT_CSV = "flashmla_precision_summary.csv"

def create_worker_script():
    content = f"""
import os, sys, json
import torch
import numpy as np
from vllm import LLM, SamplingParams

os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
import logging
logging.getLogger('vllm').setLevel(logging.ERROR)

def run():
    try:
        in_len = int(sys.argv[1])
        bs = int(sys.argv[2])
        kv_dtype = sys.argv[3] # ✨ 新增：接收 kv_dtype 参数
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        prompt = "The quick brown fox jumps over the lazy dog. " * (in_len // 10)
        prompts = [prompt] * bs
        
        # 采样参数: 贪婪解码 + Logprobs
        params = SamplingParams(temperature=0.0, max_tokens=10, ignore_eos=True, logprobs=10)
        
        # ✨ 传入 kv_cache_dtype 并加上关键参数
        llm = LLM(model="{MODEL_PATH}", 
                  max_model_len=9000,              # 对齐 Bash
                  max_num_seqs=bs,                 # 对齐 Bash
                  gpu_memory_utilization=0.9,      # 对齐 Bash (0.85 -> 0.9)
                  
                  # 🚨 关键修复：显式设大，防止 2048*8 超限报错
                  max_num_batched_tokens=65536,    
                  
                  trust_remote_code=True, 
                  dtype="bfloat16", 
                  kv_cache_dtype=kv_dtype,
                  disable_log_stats=True, 
                  enforce_eager=False,
                  enable_chunked_prefill=False,)
                  
        outputs = llm.generate(prompts, params)
        
        results = []
        for o in outputs:
            tokens = []
            if o.outputs[0].logprobs:
                for token_logprobs in o.outputs[0].logprobs:
                    # 提取 Top K logprobs 用于计算 Cosine Similarity
                    probs_dict = {{str(tid): float(obj.logprob) for tid, obj in token_logprobs.items()}}
                    # 提取 Top 1 Token ID
                    top_id = max(token_logprobs.keys(), key=lambda k: token_logprobs[k].logprob)
                    tokens.append({{'token_id': int(top_id), 'logprobs': probs_dict}})
            results.append({{'tokens': tokens}})
            
        print(json.dumps(results))
        
    except Exception as e:
        # 捕获异常并打印 JSON，防止解析失败
        print(json.dumps({{"error": str(e)}}))
        sys.exit(1)

if __name__ == "__main__":
    run()
"""
    with open(HELPER_SCRIPT, "w") as f:
        f.write(content)

def run_case(disable_mla: bool, kv_dtype: str, in_len: int, bs: int):
    env = os.environ.copy()
    env[ENV_VAR_NAME] = "1" if disable_mla else "0"
    
    # args: in_len, bs, kv_dtype
    cmd = ["python", HELPER_SCRIPT, str(in_len), str(bs), kv_dtype]
    
    try:
        res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            print(f"{RED}Error:{RESET} {res.stderr[:200]}") # 打印部分错误日志
            return None
        
        lines = res.stdout.strip().split('\n')
        # 找到最后一行 JSON 输出
        json_line = next((l for l in reversed(lines) if l.strip().startswith('[')), None)
        return json.loads(json_line) if json_line else None
    except Exception as e:
        print(f"Exception: {e}")
        return None

def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

def compare_results(base_res, target_res):
    all_sims = []
    matches = 0
    total = 0
    
    for base, target in zip(base_res, target_res):
        b_toks, t_toks = base['tokens'], target['tokens']
        for i in range(min(len(b_toks), len(t_toks))):
            total += 1
            # 1. Exact Match
            if b_toks[i]['token_id'] == t_toks[i]['token_id']: 
                matches += 1
            
            # 2. Cosine Similarity (Distribution Match)
            b_probs = b_toks[i]['logprobs']
            t_probs = t_toks[i]['logprobs']
            
            # 找共同的 Token ID 计算向量相似度
            common = set(b_probs.keys()) & set(t_probs.keys())
            if len(common) >= 2:
                ids = sorted(common)
                # logprob -> prob
                v1 = np.exp([b_probs[k] for k in ids])
                v2 = np.exp([t_probs[k] for k in ids])
                all_sims.append(cosine_similarity(v1, v2))

    avg_cosine = np.mean(all_sims) if all_sims else 0.0
    match_rate = matches / total if total > 0 else 0.0
    
    return avg_cosine, match_rate

def main():
    create_worker_script()
    
    print(f"{BOLD}FlashMLA Precision Benchmark{RESET}")
    print("=" * 85)
    print(f"{'Case':<10} {'In/BS':<12} | {'FP16 Match':<12} {'FP16 Sim':<10} | {'INT8 Match':<12} {'INT8 Sim':<10}")
    print("-" * 85)

    summary_rows = []
    
    for case in TEST_CASES:
        in_len = case['in_len']
        bs = case['bs']
        desc = f"{in_len}/{bs}"
        
        print(f"{case['name']:<10} {desc:<12} | ", end="", flush=True)
        
        # 1. Base (Official vLLM, BF16)
        base_res = run_case(disable_mla=True, kv_dtype="auto", in_len=in_len, bs=bs)
        if not base_res:
            print(f"{RED}Base Failed{RESET}")
            continue

        # 2. FlashMLA (BF16 KV)
        os.system("rm -rf ~/.triton/cache 2>/dev/null")
        fp16_res = run_case(disable_mla=False, kv_dtype="auto", in_len=in_len, bs=bs)
        
        if fp16_res:
            fp16_sim, fp16_match = compare_results(base_res, fp16_res)
            c_match = GREEN if fp16_match > 0.99 else YELLOW
            c_sim = GREEN if fp16_sim > 0.999 else YELLOW
            print(f"{c_match}{fp16_match:.4f}{RESET}       {c_sim}{fp16_sim:.6f}{RESET}   | ", end="", flush=True)
        else:
            print(f"{RED}FP16 Fail{RESET}    | ", end="", flush=True)

        # 3. FlashMLA (INT8 KV)
        os.system("rm -rf ~/.triton/cache 2>/dev/null")
        int8_res = run_case(disable_mla=False, kv_dtype="int8", in_len=in_len, bs=bs)
        
        if int8_res:
            int8_sim, int8_match = compare_results(base_res, int8_res)
            # INT8 允许略低的精度
            c_match = GREEN if int8_match > 0.90 else RED
            c_sim = GREEN if int8_sim > 0.99 else RED
            print(f"{c_match}{int8_match:.4f}{RESET}       {c_sim}{int8_sim:.6f}{RESET}")
            
            # Save row
            summary_rows.append({
                "Case": case['name'],
                "In_Len": in_len,
                "BS": bs,
                "FP16_Match": f"{fp16_match:.4f}",
                "FP16_Sim": f"{fp16_sim:.6f}",
                "INT8_Match": f"{int8_match:.4f}",
                "INT8_Sim": f"{int8_sim:.6f}",
            })
        else:
            print(f"{RED}INT8 Fail{RESET}")

    # CSV Output
    if summary_rows:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n{GREEN}Summary saved to: {OUTPUT_CSV}{RESET}")
    
    if os.path.exists(HELPER_SCRIPT): os.remove(HELPER_SCRIPT)

if __name__ == "__main__":
    main()
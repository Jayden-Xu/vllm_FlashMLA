import subprocess
import os
import json
import csv
import numpy as np

MODEL_PATH = "deepseek-ai/DeepSeek-V2-Lite-Chat"
ENV_VAR_NAME = "DISABLE_FLASH_MLA"

TEST_CASES = [
    {"name": "Short",  "in_len": 1024,  "bs": 1},
    {"name": "Medium", "in_len": 2048, "bs": 8},
    {"name": "Long",   "in_len": 4096, "bs": 32},
]

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
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
        torch.manual_seed(42)
        np.random.seed(42)
        prompt = "The quick brown fox jumps over the lazy dog. " * (in_len // 10)
        prompts = [prompt] * bs
        params = SamplingParams(temperature=0.0, max_tokens=20, ignore_eos=True, logprobs=10)
        llm = LLM(model="{MODEL_PATH}", max_model_len=max(8192, in_len + 1000), 
                  gpu_memory_utilization=0.85, trust_remote_code=True, 
                  dtype="bfloat16", disable_log_stats=True, enforce_eager=False)
        outputs = llm.generate(prompts, params)
        results = []
        for o in outputs:
            tokens = []
            if o.outputs[0].logprobs:
                for token_logprobs in o.outputs[0].logprobs:
                    probs_dict = {{str(tid): float(obj.logprob) for tid, obj in token_logprobs.items()}}
                    top_id = max(token_logprobs.keys(), key=lambda k: token_logprobs[k].logprob)
                    tokens.append({{'token_id': int(top_id), 'logprobs': probs_dict}})
            results.append({{'tokens': tokens}})
        print(json.dumps(results))
    except Exception as e:
        print(json.dumps({{"error": str(e)}}))
        sys.exit(1)
if __name__ == "__main__":
    run()
"""
    with open(HELPER_SCRIPT, "w") as f:
        f.write(content)

def run_case(disable_mla: bool, in_len: int, bs: int):
    env = os.environ.copy()
    env[ENV_VAR_NAME] = "1" if disable_mla else "0"
    cmd = ["python", HELPER_SCRIPT, str(in_len), str(bs)]
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0: return None
    try:
        lines = res.stdout.strip().split('\n')
        json_line = next((l for l in reversed(lines) if l.strip().startswith('[')), None)
        return json.loads(json_line) if json_line else None
    except: return None

def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

def get_case_summary(base_res, ours_res, case_config):

    all_sims = []
    matches = 0
    total = 0
    
    for base, ours in zip(base_res, ours_res):
        b_toks, o_toks = base['tokens'], ours['tokens']
        for i in range(min(len(b_toks), len(o_toks))):
            total += 1
            if b_toks[i]['token_id'] == o_toks[i]['token_id']: matches += 1
            
            b_probs = b_toks[i]['logprobs']
            o_probs = o_toks[i]['logprobs']
            common = set(b_probs.keys()) & set(o_probs.keys())
            if len(common) >= 2:
                ids = sorted(common)
                v1 = np.exp([b_probs[k] for k in ids])
                v2 = np.exp([o_probs[k] for k in ids])
                all_sims.append(cosine_similarity(v1, v2))

    avg_cosine = np.mean(all_sims) if all_sims else 0.0
    min_cosine = np.min(all_sims) if all_sims else 0.0
    
    return {
        "Case_Name": case_config['name'],
        "Context_Length": case_config['in_len'],
        "Batch_Size": case_config['bs'],
        "Avg_Cosine_Similarity": f"{avg_cosine:.6f}",
        "Min_Cosine_Similarity": f"{min_cosine:.6f}",
        "Token_Match_Rate": f"{matches/total:.4f}"
    }

def main():
    create_worker_script()
    
    summary_rows = []
    
    for case in TEST_CASES:
        print(f"Processing {case['name']}...", end="", flush=True)
        
        base_res = run_case(True, case['in_len'], case['bs'])
        
        os.system("rm -rf ~/.triton/cache 2>/dev/null")
        ours_res = run_case(False, case['in_len'], case['bs'])
        
        if base_res and ours_res:
            row = get_case_summary(base_res, ours_res, case)
            summary_rows.append(row)
            print(f" {GREEN}Done{RESET} (Sim: {row['Avg_Cosine_Similarity']})")
        else:
            print(f" {RED}Failed{RESET}")

    if summary_rows:
        keys = summary_rows[0].keys()
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n{GREEN}Summary saved to: {OUTPUT_CSV}{RESET}")
    
    if os.path.exists(HELPER_SCRIPT): os.remove(HELPER_SCRIPT)

if __name__ == "__main__":
    main()
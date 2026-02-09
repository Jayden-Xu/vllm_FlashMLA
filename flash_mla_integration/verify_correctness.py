import subprocess
import os
import json
import time

MODEL_PATH = "deepseek-ai/DeepSeek-V2-Lite-Chat"

TEST_CASES = [
    {"name": "Fused Kernel (BS=64)", "in_len": 1000, "bs": 64},
    {"name": "Split-K Kernel (BS=1)", "in_len": 8000, "bs": 1},
]

HELPER_SCRIPT = "verify_worker_precision.py"
ENV_VAR_NAME = "DISABLE_FLASH_MLA"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

def create_worker_script():
    content = f"""
import os, sys, json
import torch
from vllm import LLM, SamplingParams

os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
import logging
logging.getLogger('vllm').setLevel(logging.ERROR)

def run():
    try:
        in_len = int(sys.argv[1])
        bs = int(sys.argv[2])

        prompts = [{{'prompt_token_ids': [100] * in_len}}] * bs
        
        params = SamplingParams(
            temperature=0.0, 
            max_tokens=1, 
            ignore_eos=True,
            logprobs=5 
        )

        llm = LLM(
            model="{MODEL_PATH}",
            max_model_len=8192,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            dtype="bfloat16",
            disable_log_stats=True
        )

        outputs = llm.generate(prompts, params)
        
        res_data = []
        for o in outputs:
            token_data = o.outputs[0].logprobs[0]
            top1_id = max(token_data.keys(), key=lambda k: token_data[k].logprob)
            probs_dict = {{tid: obj.logprob for tid, obj in token_data.items()}}
            
            res_data.append({{
                "id": top1_id,
                "probs": probs_dict
            }})
            
        print(json.dumps(res_data))

    except Exception as e:
        print(json.dumps({{"error": str(e)}}))

if __name__ == "__main__":
    run()
"""
    with open(HELPER_SCRIPT, "w") as f:
        f.write(content)

def run_case(disable_mla, in_len, bs):
    env = os.environ.copy()
    env[ENV_VAR_NAME] = "1" if disable_mla else "0"
    cmd = ["python", HELPER_SCRIPT, str(in_len), str(bs)]
    
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if res.returncode != 0: 
        return None, res.stderr[-500:]
    try:
        out = json.loads(res.stdout.strip().split('\\n')[-1])
        if isinstance(out, dict) and 'error' in out: return None, out['error']
        return out, None
    except: return None, res.stdout

def analyze_results(base_res, ours_res, case_name):
    perfect = 0
    acceptable = 0
    fail = 0
    
    print(f"\n{BOLD}Analysis: {case_name}{RESET}")
    print(f"{'Idx':<5} {'Base ID':<8} {'Ours ID':<8} {'Base Prob':<10} {'Ours Prob':<10} {'Diff':<8} {'Verdict'}")
    print("-" * 80)

    for i, (b, o) in enumerate(zip(base_res, ours_res)):
        bid = b['id']
        oid = o['id']
        bid_key = str(bid)
        
        if bid_key not in b['probs']:
            print(f"{i:<5} Error: ID {bid} missing in probs")
            continue

        b_prob = b['probs'][bid_key]
        o_prob_of_bid = o['probs'].get(bid_key, -999.9)
        prob_diff = abs(b_prob - o_prob_of_bid)
        
        if bid == oid:
            perfect += 1
            verdict = f"{GREEN}Exact{RESET}"
        elif prob_diff < 0.05:
            acceptable += 1
            verdict = f"{YELLOW}Precision{RESET}"
        else:
            fail += 1
            verdict = f"{RED}FAIL{RESET}"
            
        if bid != oid or i < 3:
            o_prob_str = f"{o_prob_of_bid:.4f}" if o_prob_of_bid > -100 else "N/A"
            diff_str = f"{prob_diff:.4f}" if o_prob_of_bid > -100 else "Inf"
            print(f"{i:<5} {bid:<8} {oid:<8} {b_prob:<10.4f} {o_prob_str:<10} {diff_str:<8} {verdict}")

    print("-" * 80)
    
    if fail == 0:
        print(f"Result: {GREEN}PASS{RESET} (Exact: {perfect}, Precision Noise: {acceptable})")
    else:
        print(f"Result: {RED}FAIL{RESET} (Logic Errors: {fail})")
    print("=" * 80)

def main():
    print(f"{BOLD}Dual-Mode Precision Verification{RESET}")
    create_worker_script()
    
    for case in TEST_CASES:
        name = case["name"]
        in_len = case["in_len"]
        bs = case["bs"]
        
        print(f"\n{BOLD}>>> Testing Case: {name}{RESET}")
        print(f"Config: Input={in_len}, BatchSize={bs}")
        
        print("Running Baseline...", end="", flush=True)
        base_res, err = run_case(True, in_len, bs)
        if err: 
            print(f"{RED}Error:{RESET} {err}")
            continue
        print(f"{GREEN}Done.{RESET}")
        
        time.sleep(5)

        os.system("rm -rf ~/.triton/cache")
        print("Running Ours...    ", end="", flush=True)
        ours_res, err = run_case(False, in_len, bs)
        if err:
            print(f" {RED}Error:{RESET} {err}")
            continue
        print(f" {GREEN}Done.{RESET}")

        if base_res and ours_res:
            analyze_results(base_res, ours_res, name)
        else:
            print(f"{RED}Skipping analysis due to execution failure.{RESET}")

    if os.path.exists(HELPER_SCRIPT): os.remove(HELPER_SCRIPT)

if __name__ == "__main__":
    main()
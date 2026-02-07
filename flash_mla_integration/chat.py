import os
import torch
from vllm import LLM, SamplingParams

# 1. 强制开启你的算子
os.environ["DISABLE_FLASH_MLA"] = "0"
# 2. 强制显示 INFO 日志 (为了看到 [FlashMLA] 的输出)
os.environ["VLLM_LOGGING_LEVEL"] = "INFO"

def main():
    model_name = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    print(f"\n{'-'*20} 正在加载模型 (FlashMLA Enabled) {'-'*20}")
    
    # 初始化 LLM
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        disable_log_stats=False 
    )

    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=16, # 生成短一点，只要跑通就行
        ignore_eos=True
    )

    print(f"\n{'='*80}")
    
    # --- Case 1: 强制触发 Split-K (Stage 1 & 2) ---
    # 条件：Batch Size 小 (<=4) 且 序列长度长 (>256)
    # 我们构造一个 1000 长度的 Prompt，这样 num_splits 会是 4 (1000/256)
    print(f"\n{'>'*10} 测试路径 1: Split-K Kernel {'>'*10}")
    print("条件: Batch Size=1, Input Len=1000 (触发分块逻辑)")
    
    # 构造一个长 prompt (避免 tokenizer 处理慢，直接用 repeat)
    long_prompt = "What is the capital of China?" * 100
    
    outputs_sk = llm.generate([long_prompt], sampling_params)
    print(f"Split-K 输出片段: {outputs_sk[0].outputs[0].text[:50]}...")
    
    print("-" * 40)
    print("观察上方日志：应该出现 [FlashMLA] Split-K Stage 1: {best_config...}")
    print("-" * 40)


    # --- Case 2: 强制触发 Fused Path ---
    # 条件：Batch Size 大 (>=32)
    print(f"\n{'>'*10} 测试路径 2: Fused Kernel {'>'*10}")
    print("条件: Batch Size=64 (触发 Fused 逻辑)")
    
    short_prompts = ["Calculate 1+1="] * 64
    outputs_f = llm.generate(short_prompts, sampling_params)
    print(f"Fused 输出片段: {outputs_f[0].outputs[0].text.strip()}")

    print("-" * 40)
    print("观察上方日志：应该出现 [FlashMLA] Fused Path: {best_config...}")
    print("-" * 40)

    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
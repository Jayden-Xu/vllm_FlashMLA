import os
import torch
from vllm import LLM, SamplingParams

# 确保开启我们的优化算子
os.environ["DISABLE_FLASH_MLA"] = "0"

def main():
    model_name = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    print(f"\n{'-'*20} 正在加载模型 (FlashMLA Enabled) {'-'*20}")
    
    # 初始化 LLM
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        disable_log_stats=False # 开启日志可以看到 [FlashMLA] 的加载信息
    )

    sampling_params = SamplingParams(
        temperature=0.0, # 用 0.0 观察最确定的逻辑输出
        max_tokens=64,
        stop=["<|EOT|>", "<|end_of_sentence|>"]
    )

    print(f"\n{'='*60}")
    
    # --- 1. 测试 Split-K 路径 (小 Batch Size) ---
    print("\n[测试路径 1]: Split-K (Batch Size = 1)")
    splitk_prompts = ["What is the capital of France?"]
    outputs_sk = llm.generate(splitk_prompts, sampling_params)
    print(f"回答: {outputs_sk[0].outputs[0].text.strip()}")

    # --- 2. 测试 Fused 路径 (大 Batch Size) ---
    # 根据后端逻辑 batch_size >= 32 会触发 Fused Path
    print("\n[测试路径 2]: Fused (Batch Size = 64)")
    fused_prompts = ["Calculate 25 * 4 ="] * 64
    outputs_f = llm.generate(fused_prompts, sampling_params)
    # 我们只看其中一个的结果
    print(f"回答 (抽样): {outputs_f[0].outputs[0].text.strip()}")

    print("\n" + "="*60)
    print("测试完成！如果两个回答都符合逻辑，说明两套算子路径集成成功。")

if __name__ == "__main__":
    main()
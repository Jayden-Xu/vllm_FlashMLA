import os
import sys
import vllm
from vllm import LLM, SamplingParams

os.environ["DISABLE_FLASH_MLA"] = "0"

def main():
    print("="*60)
    
    model_name = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    print(f"\nLoading model: {model_name} ...")
    
    try:
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            disable_log_stats=True
        )
    except Exception as e:
        print(f"\nFailed to load model: {e}")
        return


    prompts = [
        "Please calculate: 25 * 4 + 10 = ?",
    ]

    sampling_params = SamplingParams(
        temperature=0.7, 
        top_p=0.9, 
        max_tokens=128,
        stop=["<|EOT|>", "<|end_of_sentence|>"]
    )

    outputs = llm.generate(prompts, sampling_params)

    print("\n" + "="*60)

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        
        print(f"\n[Query {i+1}]: {prompt}")
        print(f"[Answer {i+1}]: {generated_text.strip()}")
        print("-" * 40)

if __name__ == "__main__":
    main()
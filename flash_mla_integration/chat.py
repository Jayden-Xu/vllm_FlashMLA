import os
import sys
import io
from vllm import LLM, SamplingParams

os.environ["DISABLE_FLASH_MLA"] = "0"
os.environ["VLLM_LOGGING_LEVEL"] = "INFO"

class CaptureAndPrint:
    def __init__(self):
        self.terminal = sys.stdout
        self.buffer = io.StringIO()

    def write(self, message):
        self.terminal.write(message)
        self.buffer.write(message)

    def flush(self):
        self.terminal.flush()
        self.buffer.flush()

    def get_value(self):
        return self.buffer.getvalue()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.terminal

def main():
    model_name = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    print(f"\n{'-'*20} Loading Model: {model_name} (FlashMLA Enabled) {'-'*20}")
    
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        disable_log_stats=False 
    )

    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=16, 
        ignore_eos=True
    )

    print(f"\n{'='*80}")

    print(f"\n{'>'*10}Testing Path 1: Split-K Kernel {'>'*10}")
    print("Batch Size=1, Input Len=1000")
    
    long_prompt = "What is the capital of the United States? Just give me the answer." * 100
    
    with CaptureAndPrint() as cap_sk:
        outputs_sk = llm.generate([long_prompt], sampling_params)
    
    print(f"Model Output: {outputs_sk[0].outputs[0].text[:50]}...")
    
    print("-" * 40)

    print(f"\n{'>'*10}Testing Path 2: Fused Kernel {'>'*10}")
    print("Batch Size=64")
    
    short_prompts = ["Calculate 1+1="] * 64
    
    with CaptureAndPrint() as cap_f:
        outputs_f = llm.generate(short_prompts, sampling_params)
        
    print(f"Model Output: {outputs_f[0].outputs[0].text.strip()}")
    print("-" * 40)

if __name__ == "__main__":
    main()
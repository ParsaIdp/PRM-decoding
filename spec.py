import time
import wandb
from datasets import load_dataset
from vllm import LLM, SamplingParams
import torch

# Initialize WandB project
wandb.init(project="llm_spec_decoding", name="benchmark_gsm8k_math500")

# Model configurations
model_id = "meta-llama/Llama-3.3-70B-Instruct"
spec_model = "meta-llama/Llama-3.2-1B"

# Initialize the LLM with speculative decoding
llm = LLM(
    model=model_id,
    speculative_model=spec_model,
    tensor_parallel_size=1,
    num_speculative_tokens=5,
    device="cuda" if torch.cuda.is_available() else "cpu",
    disable_async_output_proc=True
)

# Sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Load datasets
gsm8k_dataset = load_dataset("openai/gsm8k", split="test")
math500_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

# Function to benchmark token generation
def benchmark_dataset(dataset, dataset_name):
    for idx, example in enumerate(dataset):
        question = example["question"] if dataset_name == "gsm8k" else example["problem"]
        prompts = [question]

        start_time = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.perf_counter()

        # Calculate tokens generated and tokens/sec
        total_tokens = sum(len(output.tokens) for output in outputs[0].outputs)
        elapsed_time = end_time - start_time
        tokens_per_second = total_tokens / elapsed_time

        # Log results to WandB
        wandb.log({
            "dataset": dataset_name,
            "example_idx": idx,
            "question": question,
            "total_tokens": total_tokens,
            "elapsed_time": elapsed_time,
            "tokens_per_second": tokens_per_second
        })

        print(f"[{dataset_name}] Example {idx}: {tokens_per_second:.2f} tokens/sec")

# Run benchmark on both datasets
print("Benchmarking GSM8K...")
benchmark_dataset(gsm8k_dataset, "gsm8k")

print("Benchmarking MATH500...")
benchmark_dataset(math500_dataset, "math500")

# Finish WandB run
wandb.finish()

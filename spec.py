import time
from datasets import load_dataset
from vllm import LLM, SamplingParams
from modal import Stub, Image, Secret
from huggingface_hub import login
import os
import torch

# Define Modal app and image with secrets
app = Stub("llm_spec_decoding_Parsa")
image = (
    Image.debian_slim()
    .apt_install("build-essential") 
    .pip_install(
        "torch"
    )
    .pip_install(
        "xformers==0.0.20",
        find_links="https://raw.githubusercontent.com/facebookresearch/xformers/main/wheels/cu118/torch2.0/index.html"
    )
    .pip_install("huggingface-hub", "datasets", "vllm", "triton")
).env({"HUGGINGFACE_TOKEN": "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"})

# Authenticate Hugging Face
def authenticate_hf():
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if hf_token is None:
        raise ValueError("Hugging Face token not found in environment variables.")
    login(token=hf_token)
    print("Authenticated with Hugging Face.")


    
def initialize_llm():
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if hf_token is None:
        raise ValueError("Hugging Face token not found.")

    return LLM(
        model="meta-llama/Llama-3.2-11B-Vision",
        speculative_model="meta-llama/Llama-3.2-1B",
        tensor_parallel_size=1,
        num_speculative_tokens=5,
        device="cuda",
        trust_remote_code=True,
        max_model_len=2048,
        block_size=16,
        worker_cls="vllm.worker.worker.Worker",
        disable_async_output_proc=True,
        enforce_eager=True,  
        hf_overrides={"use_auth_token": hf_token}
    )

# Sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


@app.function(
    image=image,
    gpu="A100",
    memory="32Gi"
)
def load_gsm8k():
    return load_gsm8k_core()  # Call the core function

@app.function(
    image=image,
    gpu="A100",
    memory="32Gi"
)
def load_math500():
    return load_math500_core()  # Call the core function

def load_gsm8k_core():
    authenticate_hf()
    return load_dataset("openai/gsm8k", "main", split="test", use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))

def load_math500_core():
    authenticate_hf()
    return load_dataset("HuggingFaceH4/MATH-500", split="test", use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))




@app.function(
    image=image,
    gpu="A100",
    memory="32Gi"
)
def benchmark_dataset_function(dataset_name: str):
    # Perform the benchmarking logic
    authenticate_hf()
    llm = initialize_llm()

    # Load dataset
    if dataset_name == "gsm8k":
        dataset = load_gsm8k()
    elif dataset_name == "math500":
        dataset = load_math500()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Perform benchmarking
    run_benchmark(dataset, dataset_name, llm)


def run_benchmark(dataset, dataset_name, llm):
    # Logic to benchmark dataset
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

        # Log results to Modal's stdout
        print({
            "dataset": dataset_name,
            "example_idx": idx,
            "question": question,
            "total_tokens": total_tokens,
            "elapsed_time": elapsed_time,
            "tokens_per_second": tokens_per_second
        })

        print(f"[{dataset_name}] Example {idx}: {tokens_per_second:.2f} tokens/sec")


@app.local_entrypoint()
def main():
    print("Authenticating...")
    authenticate_hf()

    print("Benchmarking GSM8K...")
    dataset = load_gsm8k_core() 
    llm = initialize_llm()  
    run_benchmark(dataset, "gsm8k", llm)

    print("Benchmarking MATH500...")
    dataset = load_math500_core()  
    run_benchmark(dataset, "math500", llm)

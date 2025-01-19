from prm_search import LlamaPRMSearchForCausalLM

# Initialize the model
llm = LlamaPRMSearchForCausalLM(
    {
        "model_path": "meta-llama/Llama-3.2-1B-Instruct",
        "gpu_memory_utilization": 0.8,
        "prm_path": "???",
    }
)
print(llm)

# Generate text with beam search
output = llm.generate_with_beam_search(
    "What are the applications of AI?", num_beams=3, max_length=50
)
print("Beam Search Output:", output)

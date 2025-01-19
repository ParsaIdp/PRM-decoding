import torch
from torch import nn
from transformers import LlamaForCausalLM, AutoTokenizer

# from vllm.vllm.attention import attention
from torch.nn import MultiheadAttention

from sal.models.reward_models import load_prm
from sal.search import beam_search

# from vllm.vllm.attention import PagedAttentionWithRoPE
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)


class LlamaPRMSearchForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Load Llama and tokenizer
        print(config)
        print(config["model_path"])
        self.model = LlamaForCausalLM.from_pretrained(config["model_path"])
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_path"])

        # Initialize PRM
        self.prm = load_prm(config)

        # Replace attention mechanism with PagedAttentionWithRoPE
        """self.paged_attention = PagedAttentionWithRoPE(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
        )"""
        self.attention = MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            batch_first=True,
        )

        # Replace feed-forward layers with parallelized versions
        self.ffn1 = ColumnParallelLinear(config.hidden_size, config.intermediate_size)
        self.ffn2 = RowParallelLinear(config.intermediate_size, config.hidden_size)

    def forward(self, input_ids, positions, kv_caches, attn_metadata):
        # Embed inputs
        embeddings = self.model.model.embed_tokens(input_ids)
        embeddings += self.model.model.embed_positions(positions)

        # Attention operation
        attn_output = self.paged_attention(embeddings, kv_caches, attn_metadata)

        # PRM scoring
        prm_scores = self.prm(attn_output)

        # Feed-forward layers
        ffn_output = self.ffn2(self.ffn1(attn_output))

        # Combine outputs with PRM scores
        return ffn_output + prm_scores

    def generate_with_beam_search(self, input_text, num_beams=5, max_length=50):
        # Tokenize input text
        input_ids = self.tokenizer(input_text, return_tensors="pt")["input_ids"]

        # Use SAL's beam_search
        beam_search_results = beam_search(
            model=self.model,
            input_ids=input_ids,
            num_beams=num_beams,
            max_length=max_length,
        )

        # Decode and return generated text
        return [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in beam_search_results
        ]

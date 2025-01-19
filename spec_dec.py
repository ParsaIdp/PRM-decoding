#!/usr/bin/env python3
# speculative_decode_llama.py

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

#######################################################
# 1) Load Models & Tokenizers
#######################################################
def load_models_and_tokenizers(draft_model_name, target_model_name, device_map="auto"):
    """
    Loads a small 'draft' LLaMA model and a large 'target' LLaMA model,
    along with their tokenizers. Returns the (draft_model, target_model, 
    draft_tokenizer, target_tokenizer).
    
    device_map='auto' will try to place layers across available GPU(s).
    For very large models (70B), consider using accelerate or a manual
    device mapping.
    """
    tokenizer_draft = LlamaTokenizer.from_pretrained(draft_model_name)
    tokenizer_target = LlamaTokenizer.from_pretrained(target_model_name)
    
    draft_model = LlamaForCausalLM.from_pretrained(draft_model_name, device_map=device_map)
    target_model = LlamaForCausalLM.from_pretrained(target_model_name, device_map=device_map)
    
    draft_model.eval()
    target_model.eval()
    
    return draft_model, target_model, tokenizer_draft, tokenizer_target


#######################################################
# 2) Top-k filtering utility for sampling
#######################################################
def top_k_filter(logits, top_k=50):
    """
    Keeps only the top_k values in 'logits' across the last dimension,
    and sets the rest to -inf. Used for sampling.
    """
    if top_k <= 0:
        return logits
    values, indices = torch.topk(logits, k=top_k, dim=-1)
    out = torch.full_like(logits, float('-inf'))
    out.scatter_(1, indices, values)
    return out


#######################################################
# 3) Sampling tokens from the draft model (with cache)
#######################################################
def draft_sample(
    model,
    prefix_ids,
    num_tokens,
    past_key_values=None,
    top_k=50,
    temperature=1.0
):
    """
    Sample `num_tokens` from `model` given a known prefix (prefix_ids).
    We use and update `past_key_values` to avoid recomputing from scratch.
    
    Returns:
      (sampled_ids, updated_past_key_values)
    """
    device = next(model.parameters()).device
    sampled_ids = []

    # If we have no cache yet, run the prefix through the model
    if past_key_values is None:
        if len(prefix_ids) == 0:
            raise ValueError("prefix_ids cannot be empty if past_key_values is None.")
        input_ids = torch.tensor([prefix_ids], device=device)
        with torch.no_grad():
            out = model(input_ids, use_cache=True)
        past_key_values = out.past_key_values

        
        logits = out.logits[:, -1, :]
    
        filtered_logits = top_k_filter(logits, top_k=top_k)

        filtered_logits = filtered_logits / temperature
        probs = torch.softmax(filtered_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        sampled_ids.append(next_token_id)
        input_ids = torch.tensor([[next_token_id]], device=device)
        with torch.no_grad():
            out = model(input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values

        remaining = num_tokens - 1
    else:
        remaining = num_tokens

    for _ in range(remaining):
        # Step forward with the previously generated token
        # The last token is sampled_ids[-1], but we already advanced one step above.
        logits = out.logits[:, -1, :]
        filtered_logits = top_k_filter(logits, top_k=top_k)
        filtered_logits = filtered_logits / temperature
        probs = torch.softmax(filtered_logits, dim=-1)
        next_token_id = torch.multinomial(probs, 1).item()
        sampled_ids.append(next_token_id)

        # Update model state
        input_ids = torch.tensor([[next_token_id]], device=device)
        with torch.no_grad():
            out = model(input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values

    return sampled_ids, past_key_values


#######################################################
# 4) Partial acceptance function
#######################################################
def partial_accept(
    prefix_ids,
    candidate_ids,
    draft_model,
    target_model,
    threshold=0.0,
    past_key_values_draft=None,
    past_key_values_target=None
):
    """
    Attempt partial acceptance of the tokens in `candidate_ids`.
    
    We step token by token through BOTH the draft_model and target_model,
    updating their past_key_values. We track the running difference
    (log p_L - log p_S). If it stays above `threshold`, we accept the token.
    Once it dips below threshold, we stop accepting further tokens.
    
    Returns:
      accepted_ids, 
      num_accepted (int),
      updated_past_draft,
      updated_past_target
    """
    device = next(draft_model.parameters()).device

    # If we have no caches yet, build them by running prefix
    if past_key_values_draft is None and len(prefix_ids) > 0:
        input_ids = torch.tensor([prefix_ids], device=device)
        with torch.no_grad():
            out_d = draft_model(input_ids, use_cache=True)
        past_key_values_draft = out_d.past_key_values

    if past_key_values_target is None and len(prefix_ids) > 0:
        input_ids = torch.tensor([prefix_ids], device=device)
        with torch.no_grad():
            out_t = target_model(input_ids, use_cache=True)
        past_key_values_target = out_t.past_key_values

    accepted_ids = []
    logp_s = 0.0
    logp_l = 0.0
    num_accepted = 0

    # We'll generate token by token
    for i, token_id in enumerate(candidate_ids):
        # Step draft model forward by one token
        input_ids = torch.tensor([[token_id]], device=device)
        with torch.no_grad():
            out_d = draft_model(
                input_ids=input_ids,
                past_key_values=past_key_values_draft,
                use_cache=True
            )
        logits_d = out_d.logits[:, -1, :]
        past_key_values_draft = out_d.past_key_values
        token_logprob_s = torch.log_softmax(logits_d, dim=-1)[0, token_id].item()
        logp_s += token_logprob_s

        # Step target model forward by one token
        with torch.no_grad():
            out_t = target_model(
                input_ids=input_ids,
                past_key_values=past_key_values_target,
                use_cache=True
            )
        logits_t = out_t.logits[:, -1, :]
        past_key_values_target = out_t.past_key_values
        token_logprob_l = torch.log_softmax(logits_t, dim=-1)[0, token_id].item()
        logp_l += token_logprob_l

        # Check acceptance ratio
        delta = logp_l - logp_s
        if delta >= threshold:
            # Accept this token
            accepted_ids.append(token_id)
            num_accepted += 1
        else:
            # We do NOT accept this token (nor any subsequent ones).
            # Break and let the main loop handle the remainder.
            break

    return accepted_ids, num_accepted, past_key_values_draft, past_key_values_target


#######################################################
# 5) Fallback sampling from target model (one token)
#######################################################
def sample_next_token(
    prefix_ids,
    model,
    past_key_values=None,
    top_k=50,
    temperature=1.0
):
    """
    Sample exactly 1 token from `model`, given the current prefix (prefix_ids).
    If `past_key_values` is None, we'll build it from scratch using prefix_ids.
    Returns:
      next_token_id, updated_past_key_values
    """
    device = next(model.parameters()).device

    if past_key_values is None:
        # Build from scratch
        input_ids = torch.tensor([prefix_ids], device=device)
        with torch.no_grad():
            out = model(input_ids, use_cache=True)
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :]
    else:
        # If we already have the cache, the last token is prefix_ids[-1].
        input_ids = torch.tensor([[prefix_ids[-1]]], device=device)
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
        logits = out.logits[:, -1, :]

    filtered_logits = top_k_filter(logits, top_k=top_k)
    filtered_logits = filtered_logits / temperature
    probs = torch.softmax(filtered_logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1).item()
    updated_past_key_values = out.past_key_values

    return next_id, updated_past_key_values


#######################################################
# 6) Main Speculative Decoding function
#######################################################
def speculative_decode(
    prompt,
    draft_model,
    target_model,
    tokenizer_draft,
    tokenizer_target,
    max_new_tokens=100,
    alpha=4,
    threshold=0.0,
    top_k=50,
    temperature=1.0
):
    """
    A partial-acceptance speculative decoding approach:
      1) 'Draft' alpha tokens from the smaller model.
      2) Attempt partial acceptance with the large model (token by token).
      3) For rejected tokens, sample them from the large model directly.
      4) Repeat until max_new_tokens is reached.
    """
    # Tokenize prompt
    prefix_ids = tokenizer_draft.encode(prompt, add_special_tokens=False)
    device = next(draft_model.parameters()).device

    # Build initial caches
    past_draft = None
    past_target = None
    if len(prefix_ids) > 0:
        input_ids = torch.tensor([prefix_ids], device=device)
        with torch.no_grad():
            out_d = draft_model(input_ids, use_cache=True)
            past_draft = out_d.past_key_values
            out_t = target_model(input_ids, use_cache=True)
            past_target = out_t.past_key_values

    generated_ids = prefix_ids[:]

    for _ in range(max_new_tokens):
       
        draft_chunk, past_draft = draft_sample(
            model=draft_model,
            prefix_ids=generated_ids,
            num_tokens=alpha,
            past_key_values=past_draft,
            top_k=top_k,
            temperature=temperature
        )

        # 2) Partial acceptance
        accepted_ids, num_accepted, past_draft, past_target = partial_accept(
            prefix_ids=generated_ids,
            candidate_ids=draft_chunk,
            draft_model=draft_model,
            target_model=target_model,
            threshold=threshold,
            past_key_values_draft=past_draft,
            past_key_values_target=past_target
        )

        # Extend generated_ids by what was accepted
        generated_ids.extend(accepted_ids)

        # If not all accepted, fallback to big model for the remainder
        remainder = alpha - num_accepted
        if remainder > 0:
            for _ in range(remainder):
                next_id, past_target = sample_next_token(
                    prefix_ids=generated_ids,
                    model=target_model,
                    past_key_values=past_target,
                    top_k=top_k,
                    temperature=temperature
                )
                generated_ids.append(next_id)

                # Also keep the draft model's cache in sync for subsequent steps
                # Easiest approach: re-invoke the draft model on next_id so 
                # it doesn't drift too far behind.
                input_ids = torch.tensor([[next_id]], device=device)
                with torch.no_grad():
                    out_d = draft_model(input_ids, past_key_values=past_draft, use_cache=True)
                past_draft = out_d.past_key_values

    # Decode final
    return tokenizer_target.decode(generated_ids, skip_special_tokens=True)


#######################################################
# 7) Main
#######################################################
def main():
    DRAFT_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"  
    TARGET_MODEL_NAME = "meta-llama/Llama-3.1-70B"

    draft_model, target_model, tokenizer_draft, tokenizer_target = load_models_and_tokenizers(
        draft_model_name=DRAFT_MODEL_NAME,
        target_model_name=TARGET_MODEL_NAME,
        device_map="auto"
    )

    prompt = "Explain the theory of relativity in simple terms."
    max_new_tokens = 100
    alpha = 4               
    threshold = 0.0         
    top_k = 50
    temperature = 0.8

    result = speculative_decode(
        prompt=prompt,
        draft_model=draft_model,
        target_model=target_model,
        tokenizer_draft=tokenizer_draft,
        tokenizer_target=tokenizer_target,
        max_new_tokens=max_new_tokens,
        alpha=alpha,
        threshold=threshold,
        top_k=top_k,
        temperature=temperature
    )

    print("=== Speculative Decoding Result ===")
    print(result)


if __name__ == "__main__":
    main()

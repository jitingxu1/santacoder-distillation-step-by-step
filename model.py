import torch


FIM_PREFIX = "<fim-prefix>"
FIM_MIDDLE = "<fim-middle>"
FIM_SUFFIX = "<fim-suffix>"
FIM_PAD = "<fim-pad>"
EOD = "<|endoftext|>"

# inspired by santacoder demo: https://huggingface.co/spaces/bigcode/santacoder-demo
def fim_generation(model, tokenizer,prompt, max_new_tokens, temperature):
    prefix = prompt.split("<FILL-HERE>")[0]
    suffix = prompt.split("<FILL-HERE>")[1]
    middle = infill(model, tokenizer, (prefix, suffix), max_new_tokens, temperature)
    # return post_processing_fim(prefix, middle, suffix)
    return middle

def extract_fim_part(s: str):
    # Find the index of 
    start = s.find(FIM_MIDDLE) + len(FIM_MIDDLE)
    stop = s.find(EOD, start) or len(s)
    return s[start:stop]

def infill(model, tokenizer, prefix_suffix_tuples, max_new_tokens, temperature):
    if type(prefix_suffix_tuples) == tuple:
        prefix_suffix_tuples = [prefix_suffix_tuples]
        
    prompts = [f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}" for prefix, suffix in prefix_suffix_tuples]
    # `return_token_type_ids=False` is essential, or we get nonsense output.
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id
        )
    # WARNING: cannot use skip_special_tokens, because it blows away the FIM special tokens.
    return [        
        extract_fim_part(tokenizer.decode(tensor, skip_special_tokens=False)) for tensor in outputs
    ]


def code_generation(model, tokenizer, prompt, max_new_tokens=512, temperature=0.1, seed=46): 
    if "<FILL-HERE>" in prompt:
        return fim_generation(model, tokenizer, prompt, max_new_tokens, temperature=temperature)
    else:
        raise ValueError("No <FILL-HERE> in input")
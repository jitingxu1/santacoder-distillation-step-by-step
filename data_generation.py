import argparse
import random
import numpy as np
import torch

from dotenv import load_dotenv
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import openai

from model import code_generation
from rationale import openai_retinale

load_dotenv()
FIM_PREFIX = "<fim-prefix>"
FIM_MIDDLE = "<fim-middle>"
FIM_SUFFIX = "<fim-suffix>"
FIM_PAD = "<fim-pad>"
EOD = "<|endoftext|>"


def split_content(content: str, segment_length: int = 1024):
    """
    Split a string into segments of a fixed length with optional overlap.

    Args:
        input_ids (str): The input string to be split.
        segment_length (int): The desired fixed length for each segment. Default is 1024.

    Returns:
        list of str: A list of segments, each containing a portion of the input string with the specified fixed length.

    Example:
        >>> input_string = "This is a sample input string for testing."
        >>> segments = split_content(input_string, segment_length=10)
        >>> print(segments)
        ['This is a ', 'sample inp', 'ut string', ' for test', 'ing.']
    """
    segments = []
    for i in range(0, len(content), segment_length):
        segment = content[i:i + segment_length]
        if i > 0 and len(segment) < segment_length:
            borrow = segment_length - len(segment)
            segment = segments[-1][-borrow:] + segment
        if len(segment) == segment_length:
            segments.append(segment)

    return segments


def permute(
    model, tokenizer, 
    sample,
    np_rng,
):
    """
    Take in a sample (list of tokens) and perform a FIM transformation on it with a probability of fim_rate, using two FIM modes:
    PSM and SPM (with a probability of fim_spm_rate).
    """

    boundaries = list(np_rng.randint(low=0, high=len(sample) + 1, size=2))
    boundaries.sort()

    prefix = sample[: boundaries[0]]
    middle = sample[boundaries[0] : boundaries[1]]
    suffix = sample[boundaries[1] :]
    model_input = FIM_PREFIX+prefix+FIM_SUFFIX+suffix+FIM_MIDDLE+middle
    prompt = prefix + "<FILL-HERE>" + suffix
    model_output = code_generation(model, tokenizer, prompt, 512, temperature=0.0001, seed=42)

    return prompt, model_input, middle, model_output

def create_dataset(model, tokenizer, dataset, np_rng, args):
    exploded_rows = []
    for example in dataset:
 
        content = example['content']
        splitted_content = split_content(content, args.seq_length)
        exploded_rows.extend(splitted_content)

    prompts = []
    model_inputs = []
    middles = []
    model_outputs = []
    openai_rationales = []
    if args.nosample > 0:
      exploded_rows = exploded_rows[:args.nosample]

    cnt = 0
    for content in tqdm(exploded_rows):
        prompt, model_input, middle, model_output = permute(model, tokenizer, content, np_rng)
        openai_rationales.append(openai_retinale(content))
        prompts.append(prompt)
        model_inputs.append(model_input)
        middles.append(middle)
        model_outputs.append(model_output)

    # Create a new dataset from the exploded rows
    exploded_dataset = Dataset.from_dict(
        {
            "santacoder_prompts": prompts,
            "fim_inputs": model_inputs,
            "label_middles": middles,
            "santacoder_outputs": model_outputs,
            "openai_rationales": openai_rationales,

        }
    )

    return exploded_dataset

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--seq_length', type=int, required=True)
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--data_subset_path', type=str, required=False)
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--nosample', type=int, default=0)
    parser.add_argument('--run', type=int, default=46)
    parser.add_argument('--push_to_hub', action='store_true')
    parser.add_argument('--save_local', action='store_true')
    args = parser.parse_args()
  

    tokenizer_fim = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)

    tokenizer_fim.add_special_tokens({
        "additional_special_tokens": [EOD, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD],
        "pad_token": EOD,
    })

    if torch.cuda.is_available():
      # Move model to GPU
      model = model.to("cuda")
  

    dataset = load_dataset(
        args.dataset,
        data_dir=args.data_subset_path,
        split=args.split,
        # use_auth_token=use_auth_token,
        # streaming=False
    )
    if args.subsample < 1.0:
        dataset = dataset.train_test_split(test_size=1.0-args.subsample, seed=args.run)['train']

    np_rng = np.random.RandomState(seed=args.run)
    dataset = create_dataset(model, tokenizer_fim, dataset, np_rng, args)

    if args.save_local:
      print("save to local disk")
      dataset.save_to_disk(f"data_{args.nosample}.hf")
    if args.push_to_hub:
      print("pushing data to hub")
      dataset.push_to_hub(f"jitx/distillation_code_{args.nosample}")

    # python3 data_generation.py --model bigcode/santacoder --dataset bigcode/the-stack --seq_length 1024 --split train --data_subset_path data/moonscript --subsample 0.0005 --run 46

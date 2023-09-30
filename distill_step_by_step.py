# Copyright 2023 The Distilling-step-by-step authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import shutil
import logging

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration

from model_utils import TaskPrefixTrainer, TaskPrefixDataCollator
from model import FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX
from metrics import compute_metrics_text


os.environ["WANDB_PROJECT"] = "Tabby"
os.environ["WANDB_WATCH"]="all"



def get_config_dir(args):
    return f'{args.dataset}/{args.from_pretrained.split("/")[1]}'


def train_and_evaluate(args, run, tokenizer, tokenized_datasets, compute_metrics):
    # set_seed(run)

    model = T5ForConditionalGeneration.from_pretrained(args.from_pretrained)

    model.resize_token_embeddings(len(tokenizer))

    if args.parallelize:
        model.parallelize()
    
    config_dir = get_config_dir(args)
    output_dir = f'ckpts/{config_dir}/{run}'  # for model ckpts
    logging_dir = f'logs/{config_dir}/{run}'  # for training logs

    if args.no_log:
        logging_strategy = 'no'
        logging_dir = None
    else:
        logging_strategy = 'steps'

    # clear output dir if already exists
    if os.path.exists(output_dir):
        logging.info('Found existing ckpt directory. Deleted the old directory for the latest run.')
        shutil.rmtree(output_dir)

    training_args = Seq2SeqTrainingArguments(
        output_dir,
        remove_unused_columns = False,
        evaluation_strategy = 'steps',
        eval_steps=args.eval_steps,
        save_strategy='no',
        save_steps=args.eval_steps,
        logging_dir=logging_dir,
        logging_strategy=logging_strategy,
        logging_steps=args.eval_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_accumulation_steps=args.grad_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        seed=run,
        local_rank=args.local_rank,
        bf16=args.bf16,
        generation_max_length=args.gen_max_len,
        prediction_loss_only=False,
        report_to="wandb" if args.wandb_run_name else None,
        run_name=args.wandb_run_name,
    )

    if args.model_type == 'task_prefix':
        data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)
    elif args.model_type == 'standard':
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    else:
        raise ValueError


    trainer_kwargs = {
        'alpha': args.alpha,
        'output_rationale': args.output_rationale,
        'model': model,
        'args': training_args,
        'train_dataset': tokenized_datasets["train"],
        'eval_dataset': {'test': tokenized_datasets["test"],},
        'data_collator': data_collator,
        'tokenizer': tokenizer,
        'compute_metrics': compute_metrics,
    }
    

    if args.model_type == 'task_prefix':
        trainer = TaskPrefixTrainer(**trainer_kwargs)
    elif args.model_type == 'standard':
        trainer_kwargs.pop('alpha')
        trainer_kwargs.pop('output_rationale')
        trainer = Seq2SeqTrainer(**trainer_kwargs)
    else:
        raise ValueError
    

    trainer.train()


def run(args):
    #### Prepare datasets
    datasets = load_dataset(args.dataset, split="train")
    datasets = datasets.rename_column('santacoder_outputs', "label")
    datasets = datasets.rename_column('openai_rationales', "rationale")
    datasets = datasets.rename_column('fim_inputs', "input")
    datasets = datasets.remove_columns(['santacoder_prompts', 'label_middles'])


    if args.subsample < 1.0:
        datasets['train'] = datasets['train'].train_test_split(test_size=1.0-args.subsample, seed=args.run)['train']

    #### Prepare datasets Prepare data for training
    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)
    addtional_tokens = tokenizer.special_tokens_map['additional_special_tokens']
    addtional_tokens.extend([FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX])
    tokenizer.add_special_tokens({
        "additional_special_tokens": addtional_tokens,
    })

    def tokenize_function(examples):
        model_inputs = tokenizer(
          [
            text.replace("  ", "\t") # space issue in T5Tokenizer
            for text in examples['input']
          ],
          max_length=args.max_input_length,
          truncation=True,
          padding=True,
        )
        prompt = "Explain the below moonscript code within 50 words:"
        expl_model_inputs = tokenizer(
          [
            prompt + "\n" + text.replace("  ", "\t")
            for text in examples['input']
          ],
          max_length=args.max_input_length,
          truncation=True,
          padding=True,
        )
        model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
        model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']

        with tokenizer.as_target_tokenizer():
            label_output_encodings = tokenizer(examples['label'], max_length=512, truncation=True, padding=True,)
            rationale_output_encodings = tokenizer(examples['rationale'], max_length=256, truncation=True, padding=True,)

        model_inputs['labels'] = label_output_encodings['input_ids']
        model_inputs['aux_labels'] = rationale_output_encodings['input_ids']

        return model_inputs

    tokenized_datasets = datasets.map(
        tokenize_function,
        remove_columns=['input', 'label', 'rationale'],
        batched=True
    )

    compute_metrics = compute_metrics_text(tokenizer)

    tokenized_datasets= tokenized_datasets.train_test_split(0.2)

    train_and_evaluate(args, args.run, tokenizer, tokenized_datasets, compute_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--warmup_steps', type=int, default=8)
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lr_scheduler_type', type=str, default="cosine")
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--from_pretrained', type=str, default='google/t5-v1_1-base')
    parser.add_argument('--label_type', type=str, default='gt')
    parser.add_argument('--llm', type=str, default='palm')
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--grad_steps', type=int, default=2)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gen_max_len', type=int, default=512)
    parser.add_argument('--parallelize', action='store_true')
    parser.add_argument('--model_type', type=str, default='task_prefix')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--wandb_run_name', type=str, default='santacoder-distillation')
    parser.add_argument('--wandb_watch', type=str, default='gradients')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--output_rationale', action='store_true')

    args = parser.parse_args()

    if len(args.wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = args.wandb_watch

    run(args)
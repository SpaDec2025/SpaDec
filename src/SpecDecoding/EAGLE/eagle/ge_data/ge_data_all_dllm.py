import argparse

import copy

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--index', type=int, default=1)
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
parser.add_argument('--outdir', type=str, default='outdir0')
args = parser.parse_args()
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]
from modeling_llama_dllm import LlamaDLLMForCausalLM
from accelerate import Accelerator
from transformers import LlamaTokenizer, AutoTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from datasets import load_dataset
import json
from fastchat.model.model_adapter import get_conversation_template

accelerator = Accelerator()
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def generate_prompt(instruction, input_text=None):
    if input_text:
        return PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input_text)
    else:
        return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)


bigname="models/Vicuna"


def build_dataset_rank(
        tokenizer, split="train",
        select=None,
):
    ds = load_dataset('json', data_files="/datasets/vicuna-alpaca_data_result.json")
    ds = ds['train']
    ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(args.start, args.end))
    original_columns1 = ds1.column_names

    def preprocess_function(examples):
        new_examples = {
            "conversation":[],
            "input_ids": [],
            "loss_mask": []
        }
        # print(examples)
        for i in range(len(examples.data['response'])):
            prompt = generate_prompt(examples.data.get("instruction", "")[i], examples.data.get("input", "")[i])
            response = examples.data.get("response")[i]
            conversation = prompt + response
            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=2048,
                truncation=True,
            ).input_ids[0]
            loss_mask=torch.ones_like(input_ids)
            prompt_len = len(tokenizer(prompt).input_ids)
            loss_mask[:prompt_len] = 0
            conversation_len = len(tokenizer(conversation).input_ids)
            loss_mask[conversation_len:] = 0

            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )

    ds1.set_format(type="torch")
    return ds1

bigmodel = LlamaDLLMForCausalLM.from_pretrained(
        bigname,
        torch_dtype=torch.float16,
        device_map={"": accelerator.process_index},
        attn_implementation="eager",
    )
bigtokenizer = LlamaTokenizer.from_pretrained(bigname)
bigtokenizer.padding_side = 'left'
bigtokenizer.pad_token = bigtokenizer.eos_token

bigmodel.eval()

ds = build_dataset_rank(bigtokenizer)
print(ds)


@torch.no_grad()
def ge(data):
    input_ids=data["input_ids"].cuda()
    outs_big = bigmodel(input_ids, output_hidden_states=True)
    hidden_state_big = outs_big.hidden_states[-1]
    max_prob_tokens_big = torch.argmax(outs_big.logits, dim=-1)
    probs = torch.softmax(outs_big.logits, dim=-1)
    maxp=probs[0].max(dim=1).values
    td={"input_ids":input_ids.cpu()[0],"hidden_state":hidden_state_big.cpu()[0],"loss_mask":data["loss_mask"].cpu()[0]}
    return td

outdir = f'{args.outdir}/{args.index}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

def writedata(name,data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')


for id,data in enumerate(ds):
    if id%100==0:
        print(id,end="\t")
    if id % 1000 == 0:
        print("")
    outdata = ge(data)
    writedata(outdir,outdata)



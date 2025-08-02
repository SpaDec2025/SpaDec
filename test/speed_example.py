import os
import argparse
import numpy as np
import json
from tqdm import tqdm
import torch
from eagle.model.ea_model import EaModel
from eagle.model.utils import *
import time


def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Create directory: {directory}")


def write_to_json(data, filename):
    try:
        ensure_directory_exists(filename)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Success to write file：{filename}")
    except Exception as e:
        print(f"Fail to write file：{str(e)}")
        raise


def generate_prompt(example):
    # Customized data preprocessing process
    return example


@torch.no_grad()
def gene(model, tokenizer, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.base_model.device)
    input_len = input_ids.size(-1)
    start_time = time.perf_counter()
    output = model.eagenerate(
        torch.as_tensor(input_ids).cuda(),
        temperature=0.0,
        log=True
    )
    end_time = time.perf_counter()
    output_ids = output[0]
    result_map = {
        'start_time': start_time,
        'end_time': end_time,
        'length': output_ids.size(-1) - input_len
    }
    return result_map


def main():
    parser = argparse.ArgumentParser
    parser.add_argument('--base_model_path', type=str, required=True)
    parser.add_argument('--ea_model_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()

    print(f"base_model_path = {args.base_model_path}")
    print(f"ea_model_path = {args.ea_model_path}")
    print(f"Result will save to {args.save_path}")

    total_data = []
    prompts = []
    try:
        with open(args.input_data_path, "r", encoding='utf-8') as f:
            total_data = json.load(f)
    except Exception as e:
        print(f"error: {str(e)}")
        return

    data = total_data
    for i in range(len(data)):
        example = data[i]
        prompt = generate_prompt(example)
        prompts.append(prompt)

    model = EaModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        Type="Mixtral",
        total_token=60,
        depth=5,
        top_k=10,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    model.eval()
    tokenizer = model.get_tokenizer()
    logits_processor = None # temperature = 0
    result_maps = []
    for i in tqdm(range(len(prompts))):
        print(f"No:{i} prompt start inference:")
        result_map = gene(model, tokenizer, prompts[i])
        result_map['id'] = i
        result_maps.append(result_map)
    write_to_json(result_maps, args.save_path)


if __name__ == "__main__":
    main()
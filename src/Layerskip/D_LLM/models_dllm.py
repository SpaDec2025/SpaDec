import os
import sys
import json
import torch

from llama import Tokenizer
from llama.model_train import ModelArgs, Transformer
from vicuna.model_train import Transformer_vicuna, ModelArgs_Vicuna
from collections import OrderedDict
from safetensors import safe_open

def LLaMA2_7B_Dynamic(args, **kwargs):
    llama_model_path = args.llama_model_path
    llama_param_path = args.llama_param_path

    checkpoint = torch.load(os.path.join(llama_model_path, "consolidated.00.pth"), map_location="cpu")

    with open(llama_param_path, "r") as f:
        params = json.load(f)
    
    model_args: ModelArgs = ModelArgs(
        max_seq_len=args.max_seq_len,
        max_batch_size=32,
        lora_rank=args.lora_rank,
        dynamic_active_target=args.dynamic_active_target,
        dynamic_start_layer=args.dynamic_start_layer,
        dynamic_router_hdim=args.dynamic_router_hdim,
        dynamic_reserve_initials=args.dynamic_reserve_initials,
        **params
    )
    tokenizer = Tokenizer(model_path=args.tokenizer_path)

    model_args.vocab_size = tokenizer.n_words
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    model_llama_dynamic = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model_llama_dynamic.load_state_dict(checkpoint, strict=False)

    for name, param in model_llama_dynamic.named_parameters():
        if "lora" in name or "router" in name:
            param.requires_grad = True
            param.data = param.data.float()
        else:
            param.requires_grad = False

    return model_llama_dynamic


def Vicuna_7B_Dynamic(args, **kwargs):
    llama_param_path = args.llama_param_path

    original_checkpoint = OrderedDict()
    model_path = args.llama_model_path
    index_file = os.path.join(model_path, "pytorch_model.bin.index.json")
    if os.path.exists(index_file):
        with open(index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)
        for weight_file in set(index["weight_map"].values()):
            file_path = os.path.join(model_path, weight_file)
            state_dict = torch.load(file_path, map_location="cpu")
            for key, value in state_dict.items():
                original_checkpoint[key] = value

    checkpoint = OrderedDict()
    # adapt key name
    for key, value in original_checkpoint.items():
        new_key = key.replace('model.', '', 1)
        checkpoint[new_key] = value

    with open(llama_param_path, "r") as f:
        params = json.load(f)

    model_args: ModelArgs_Vicuna = ModelArgs_Vicuna(
        max_seq_len=args.max_seq_len,
        max_batch_size=32,
        lora_rank=args.lora_rank,
        dynamic_active_target=args.dynamic_active_target,
        dynamic_start_layer=args.dynamic_start_layer,
        dynamic_router_hdim=args.dynamic_router_hdim,
        dynamic_reserve_initials=args.dynamic_reserve_initials,
        **params
    )
    tokenizer = Tokenizer(model_path=args.tokenizer_path)

    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    model_llama_dynamic = Transformer_vicuna(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    missing, unexpected = model_llama_dynamic.load_state_dict(checkpoint, strict=False)

    print("---- missing params (LoRA/router expected) ----")
    for k in missing:
        print(k)

    print("\n---- unexpected params ----")
    for k in unexpected:
        print(k)

    for name, param in model_llama_dynamic.named_parameters():
        if "lora" in name or "router" in name:
            param.requires_grad = True
            param.data = param.data.float()
        else:
            param.requires_grad = False

    return model_llama_dynamic
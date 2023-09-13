#! /usr/bin/python3

import torch
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers import GPT2Tokenizer
from torch.nn.functional import pad
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import ctypes
_cudart = ctypes.CDLL('libcudart.so')

def cu_prof_start():
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception('cudaProfilerStart() returned %d' % ret)

def cu_prof_stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception('cudaProfilerStop() returned %d' % ret)

class Evaluator:
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        latency = 0
        l = len(self.dataset)
        #l = 1
        for i in range(0, l):
            batch = self.dataset[i]
            input_ids = batch['input_ids'].cuda().unsqueeze(0)
            label = input_ids[:, -1]
            pad_len = 512 - input_ids.shape[1]
            input_ids = pad(input_ids, (0, pad_len), value=1)
            torch.cuda.synchronize()
            start.record()
            outputs = model(input_ids, use_cache=True)
            end.record()
            torch.cuda.synchronize()
            latency += start.elapsed_time(end)
            last_token_logits = outputs.logits[:, -2-pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()

        acc = hit / total
        lantecy = latency / l
        return acc, lantecy

# OPT SERIES
from datasets import load_dataset
model_path = 'facebook/opt-6.7b'
print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
print("Loading dataset...")
dataset = load_dataset('lambada', split='validation[:1000]')
evaluator = Evaluator(dataset, tokenizer)

print("Loading model...")
model_fp16 = OPTForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map='auto', use_cache=True)
print("Evaluating...")
cu_prof_start()
acc_fp16, lantecy_fp16 = evaluator.evaluate(model_fp16)
cu_prof_stop()
print(f'FP16 accuracy: {acc_fp16}, per-sample lantecy: {lantecy_fp16:.3f}ms')

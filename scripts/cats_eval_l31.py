import argparse
import json
import math
import os
import sys

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, MistralConfig, MistralForCausalLM

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from experiments.models.sparse_silu.ugly_utils import (  # noqa: E402
    SparseLlamaConfig,
    SparseLlamaForCausalLM,
    SparseMistralConfig,
    SparseMistralforCausalLM,
    activate_stats,
    deactivate_stats,
    disable_sparse_silu,
    enable_sparse_silu,
    get_sparse_config,
    print_dead_neuron_stats,
    set_sparse_threshold,
)
from utils.constants import MISTRAL  # noqa: E402
from utils.utils import get_model_type_from_name  # noqa: E402


def parse_dtype(dtype: str):
    key = dtype.lower()
    if key in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if key in ["fp16", "float16", "half"]:
        return torch.float16
    if key in ["fp32", "float32"]:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def first_device(model):
    return next(model.parameters()).device


def load_cats_sparse_model(model_path: str, dtype: str, attn_implementation: str):
    model_type = get_model_type_from_name(model_path)
    BaseConfig = MistralConfig if model_type == MISTRAL else LlamaConfig
    SparseConfig = SparseMistralConfig if model_type == MISTRAL else SparseLlamaConfig
    SparseCausalLM = SparseMistralforCausalLM if model_type == MISTRAL else SparseLlamaForCausalLM

    config = BaseConfig.from_pretrained(model_path)
    config = get_sparse_config(
        config,
        model_type=model_type,
        use_sparse_model=True,
        use_sparse_predictor=False,
        use_sparse_regularization=False,
        use_graceful_regularization=False,
    )
    config.use_cache = True

    SparseConfig.register_for_auto_class()
    SparseCausalLM.register_for_auto_class("AutoModelForCausalLM")

    model = SparseCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=parse_dtype(dtype),
        attn_implementation=attn_implementation,
        device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return model, tokenizer


def collect_hist_for_threshold(
    model,
    tokenizer,
    split: str,
    max_samples: int,
    max_length: int,
    batch_size: int,
):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = []
    for row in dataset:
        text = row["text"]
        if text and text.strip():
            texts.append(text)
            if max_samples > 0 and len(texts) >= max_samples:
                break

    device = first_device(model)
    activate_stats(model, is_collect_histogram=True)
    enable_sparse_silu(model)
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Collecting CATS hist"):
            chunk = texts[i : i + batch_size]
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            model(input_ids=input_ids, attention_mask=attn)


def build_eval_token_ids(tokenizer, split: str, max_samples: int):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    ids = []
    used = 0
    eos = tokenizer.eos_token_id
    for row in dataset:
        text = row["text"]
        if not text or not text.strip():
            continue
        tok = tokenizer.encode(text, add_special_tokens=False)
        if len(tok) == 0:
            continue
        ids.extend(tok)
        if eos is not None:
            ids.append(eos)
        used += 1
        if max_samples > 0 and used >= max_samples:
            break
    return torch.tensor(ids, dtype=torch.long)


def eval_ppl_from_ids(model, token_ids, context_size: int, window_size: int):
    model.eval()
    stride = window_size
    max_length = context_size + window_size
    seq_len = token_ids.numel()
    seq_len = seq_len - (seq_len % stride)
    if seq_len <= 0:
        raise ValueError("No tokens available for PPL evaluation.")

    device = first_device(model)
    nlls = []
    with torch.no_grad():
        for begin in tqdm(range(0, seq_len, stride), desc="PPL"):
            end = min(begin + max_length, seq_len)
            input_ids = token_ids[begin:end].unsqueeze(0).to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-stride] = -100
            out = model(input_ids=input_ids, labels=target_ids)
            nlls.append(out.loss.float())
            if end >= seq_len:
                break
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    return ppl


def run_lm_eval(model, tokenizer, tasks, limit, batch_size, max_length, dtype):
    from lm_eval.evaluator import evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import get_task_dict

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        device="cuda",
        dtype=dtype,
    )
    task_dict = get_task_dict(tasks)
    result = evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        log_samples=False,
    )
    return result.get("results", {})


def pick_metric(task_result: dict, preferred):
    for k in preferred:
        if k in task_result:
            return float(task_result[k])
    return math.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--calib_split", type=str, default="train")
    parser.add_argument("--calib_samples", type=int, default=256)
    parser.add_argument("--calib_max_length", type=int, default=1024)
    parser.add_argument("--calib_batch_size", type=int, default=8)
    parser.add_argument("--ppl_split", type=str, default="test")
    parser.add_argument("--ppl_samples", type=int, default=-1)
    parser.add_argument("--context_size", type=int, default=2048)
    parser.add_argument("--window_size", type=int, default=512)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["arc_challenge", "mmlu", "openbookqa", "winogrande"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_max_length", type=int, default=2048)
    args = parser.parse_args()

    model, tokenizer = load_cats_sparse_model(
        args.model_path, args.dtype, args.attn_implementation
    )

    # 1) Collect histogram stats and set threshold for targeted sparsity.
    collect_hist_for_threshold(
        model,
        tokenizer,
        split=args.calib_split,
        max_samples=args.calib_samples,
        max_length=args.calib_max_length,
        batch_size=args.calib_batch_size,
    )
    set_sparse_threshold(model, args.sparsity, use_relu=False)
    deactivate_stats(model)

    # 2) PPL (dense path: disable CATS sparse masking)
    token_ids = build_eval_token_ids(tokenizer, args.ppl_split, args.ppl_samples)
    disable_sparse_silu(model)
    dense_ppl = eval_ppl_from_ids(model, token_ids, args.context_size, args.window_size)

    # 3) PPL + measured sparsity (sparse path)
    enable_sparse_silu(model)
    activate_stats(model, is_collect_histogram=False)
    sparse_ppl = eval_ppl_from_ids(model, token_ids, args.context_size, args.window_size)
    measured_total_sparsity, layer_sparsities = print_dead_neuron_stats(model)
    deactivate_stats(model)

    # 4) Zero-shot benchmark
    eval_results = run_lm_eval(
        model,
        tokenizer,
        args.tasks,
        args.limit,
        args.eval_batch_size,
        args.eval_max_length,
        args.dtype,
    )

    summary = {
        "model": args.model_path,
        "dtype": args.dtype,
        "p_target": args.sparsity,
        "cats_parser_default_targeted_sparsity": 0.9,
        "dense_ppl_wikitext2": dense_ppl,
        "sparse_ppl_wikitext2": sparse_ppl,
        "measured_total_sparsity_percent": measured_total_sparsity,
        "per_layer_sparsity_percent": layer_sparsities,
        "is_n_m_structured_sparsity": False,
        "has_quantized_model_experiment_in_repo": False,
        "benchmark": {
            "arc_challenge": pick_metric(eval_results.get("arc_challenge", {}), ["acc_norm,none", "acc,none"]),
            "mmlu": pick_metric(eval_results.get("mmlu", {}), ["acc,none"]),
            "openbookqa": pick_metric(eval_results.get("openbookqa", {}), ["acc_norm,none", "acc,none"]),
            "winogrande": pick_metric(eval_results.get("winogrande", {}), ["acc,none"]),
        },
        "raw_eval_results": eval_results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

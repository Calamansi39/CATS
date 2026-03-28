import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaConfig, MistralConfig

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
except Exception as exc:
    raise ImportError("matplotlib is required for heatmap output.") from exc

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

PROJ_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
HEATMAP_PROJS = {"q_proj", "k_proj", "v_proj"}


def parse_dtype(dtype: str):
    key = dtype.lower()
    if key in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if key in ["fp16", "float16", "half"]:
        return torch.float16
    if key in ["fp32", "float32"]:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def parse_layers(text: str):
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def first_device(model):
    return next(model.parameters()).device


def load_cats_sparse_model(model_path: str, dtype: str, attn_implementation: str):
    model_type = get_model_type_from_name(model_path)
    base_config = MistralConfig if model_type == MISTRAL else LlamaConfig
    sparse_config = SparseMistralConfig if model_type == MISTRAL else SparseLlamaConfig
    sparse_causal_lm = SparseMistralforCausalLM if model_type == MISTRAL else SparseLlamaForCausalLM

    config = base_config.from_pretrained(model_path)
    config = get_sparse_config(
        config,
        model_type=model_type,
        use_sparse_model=True,
        use_sparse_predictor=False,
        use_sparse_regularization=False,
        use_graceful_regularization=False,
    )
    config.use_cache = True

    sparse_config.register_for_auto_class()
    sparse_causal_lm.register_for_auto_class("AutoModelForCausalLM")

    model = sparse_causal_lm.from_pretrained(
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


def collect_hist_for_threshold(model, tokenizer, split: str, max_samples: int, max_length: int, batch_size: int):
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


def attach_input_hooks(
    model,
    target_layers_0idx,
    heatmap_seq_len,
    collect_stats=True,
    collect_capture=True,
    heatmap_d_start=0,
    heatmap_d_end=-1,
):
    n_layers = len(model.model.layers)
    stats = {k: [{"zero": 0, "total": 0} for _ in range(n_layers)] for k in PROJ_ORDER}
    captured = {}
    handles = []

    for i, layer in enumerate(model.model.layers):
        module_map = {
            "q_proj": layer.self_attn.q_proj,
            "k_proj": layer.self_attn.k_proj,
            "v_proj": layer.self_attn.v_proj,
            "o_proj": layer.self_attn.o_proj,
            "up_proj": layer.mlp.up_proj,
            "gate_proj": layer.mlp.gate_proj,
            "down_proj": layer.mlp.down_proj,
        }

        for proj_name, module in module_map.items():
            def _make_hook(layer_idx, p_name):
                def _hook(_module, inputs):
                    if not inputs:
                        return
                    x = inputs[0]
                    if not torch.is_tensor(x):
                        return
                    x_det = x.detach()
                    if collect_stats:
                        stats[p_name][layer_idx]["zero"] += int((x_det == 0).sum().item())
                        stats[p_name][layer_idx]["total"] += int(x_det.numel())

                    key = (layer_idx, p_name)
                    if (
                        collect_capture
                        and
                        p_name in HEATMAP_PROJS
                        and layer_idx in target_layers_0idx
                        and key not in captured
                        and x_det.dim() == 3
                        and x_det.size(0) >= 1
                    ):
                        x0 = x_det[0]
                        if heatmap_seq_len > 0:
                            x0 = x0[:heatmap_seq_len]
                        d0 = max(0, int(heatmap_d_start))
                        d1 = int(heatmap_d_end)
                        if d1 <= 0 or d1 > x0.size(1):
                            d1 = x0.size(1)
                        if d0 >= d1:
                            d0 = 0
                            d1 = x0.size(1)
                        x0 = x0[:, d0:d1]
                        captured[key] = x0.float().cpu()

                return _hook

            handles.append(module.register_forward_pre_hook(_make_hook(i, proj_name)))
    return handles, stats, captured


def summarize_stats(stats):
    per_layer = {}
    overall = {}
    for proj_name, entries in stats.items():
        vals = []
        z_all = 0
        t_all = 0
        for e in entries:
            z = e["zero"]
            t = e["total"]
            z_all += z
            t_all += t
            vals.append(float(z / t) if t > 0 else 0.0)
        per_layer[proj_name] = vals
        overall[proj_name] = float(z_all / t_all) if t_all > 0 else 0.0
    return per_layer, overall


def save_heatmaps(captured, output_dir, log_min_exp, square_s=0, square_d=64):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for (layer_idx, proj_name), x in sorted(captured.items(), key=lambda kv: kv[0]):
        x_abs = x.abs()
        maxv = float(x_abs.max().item())
        x_norm = x_abs / maxv if maxv > 0 else x_abs
        zero_mask = (x == 0).float()

        base = f"layer{layer_idx + 1:02d}_{proj_name}"
        heatmap_path = output_dir / f"{base}_norm01_heatmap.png"
        zero_path = output_dir / f"{base}_zero_mask.png"

        exp = int(log_min_exp) if log_min_exp is not None else 5
        if exp < 1:
            exp = 1
        vmin = 10.0 ** (-exp)
        x_plot = torch.clamp(x_norm, min=vmin, max=1.0)
        x_plot_sd = x_plot.numpy()
        zero_sd = zero_mask.numpy()
        # Plot as x=S, y=D by transposing from [S, D] -> [D, S].
        x_plot_ds = x_plot_sd.T
        zero_ds = zero_sd.T
        s_len, d_len = int(x.shape[0]), int(x.shape[1])
        fig_h = 6.0
        fig_w = max(8.0, min(40.0, fig_h * (s_len / max(d_len, 1))))

        plt.figure(figsize=(fig_w, fig_h))
        plt.imshow(
            x_plot_ds,
            aspect="equal",
            cmap="magma",
            norm=LogNorm(vmin=vmin, vmax=1.0),
        )
        cbar = plt.colorbar()
        ticks = [10.0 ** (-k) for k in range(exp, 0, -1)] + [1.0]
        cbar.set_ticks(ticks)
        plt.xlabel("S")
        plt.ylabel("D")
        plt.title(f"{base} normalized |x| in [0,1], log scale [1e-{exp}, 1] (x=S, y=D, B=1)")
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=200)
        plt.close()

        plt.figure(figsize=(fig_w, fig_h))
        plt.imshow(zero_ds, aspect="equal", cmap="gray_r", vmin=0, vmax=1)
        plt.colorbar()
        plt.xlabel("S")
        plt.ylabel("D")
        plt.title(f"{base} zero mask (X==0) (x=S, y=D)")
        plt.tight_layout()
        plt.savefig(zero_path, dpi=200)
        plt.close()

        item = {
            "layer_1idx": layer_idx + 1,
            "projection": proj_name,
            "shape_sd": [int(x.shape[0]), int(x.shape[1])],
            "norm01_heatmap": str(heatmap_path),
            "zero_mask_heatmap": str(zero_path),
        }

        sq_s = int(square_s) if square_s is not None else 0
        sq_d = int(square_d) if square_d is not None else 64
        if sq_s > 0 and sq_d > 0 and x_norm.size(0) >= sq_s and x_norm.size(1) >= sq_d:
            x_head = x_plot[:sq_s, :sq_d]
            x_tail = x_plot[-sq_s:, :sq_d]
            z_head = zero_mask[:sq_s, :sq_d]
            z_tail = zero_mask[-sq_s:, :sq_d]

            head_path = output_dir / f"{base}_head_s{sq_s}_d{sq_d}_norm01_heatmap.png"
            tail_path = output_dir / f"{base}_tail_s{sq_s}_d{sq_d}_norm01_heatmap.png"
            head_zero_path = output_dir / f"{base}_head_s{sq_s}_d{sq_d}_zero_mask.png"
            tail_zero_path = output_dir / f"{base}_tail_s{sq_s}_d{sq_d}_zero_mask.png"

            plt.figure(figsize=(6, 6))
            plt.imshow(
                x_head.numpy().T,
                aspect="equal",
                cmap="magma",
                norm=LogNorm(vmin=vmin, vmax=1.0),
            )
            cbar = plt.colorbar()
            cbar.set_ticks(ticks)
            plt.xlabel("S")
            plt.ylabel("D")
            plt.title(f"{base} head S[0:{sq_s}] D[0:{sq_d}] (x=S, y=D)")
            plt.tight_layout()
            plt.savefig(head_path, dpi=220)
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.imshow(
                x_tail.numpy().T,
                aspect="equal",
                cmap="magma",
                norm=LogNorm(vmin=vmin, vmax=1.0),
            )
            cbar = plt.colorbar()
            cbar.set_ticks(ticks)
            plt.xlabel("S")
            plt.ylabel("D")
            plt.title(f"{base} tail S[-{sq_s}:] D[0:{sq_d}] (x=S, y=D)")
            plt.tight_layout()
            plt.savefig(tail_path, dpi=220)
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.imshow(z_head.numpy().T, aspect="equal", cmap="gray_r", vmin=0, vmax=1)
            plt.colorbar()
            plt.xlabel("S")
            plt.ylabel("D")
            plt.title(f"{base} head S[0:{sq_s}] zero mask (x=S, y=D)")
            plt.tight_layout()
            plt.savefig(head_zero_path, dpi=220)
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.imshow(z_tail.numpy().T, aspect="equal", cmap="gray_r", vmin=0, vmax=1)
            plt.colorbar()
            plt.xlabel("S")
            plt.ylabel("D")
            plt.title(f"{base} tail S[-{sq_s}:] zero mask (x=S, y=D)")
            plt.tight_layout()
            plt.savefig(tail_zero_path, dpi=220)
            plt.close()

            item.update(
                {
                    "square_shape_sd": [sq_s, sq_d],
                    "square_head_norm01_heatmap": str(head_path),
                    "square_tail_norm01_heatmap": str(tail_path),
                    "square_head_zero_mask": str(head_zero_path),
                    "square_tail_zero_mask": str(tail_zero_path),
                }
            )

        written.append(item)

    return written


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
    return torch.exp(torch.stack(nlls).mean()).item()


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
    result = evaluate(lm=lm, task_dict=task_dict, limit=limit, log_samples=False)
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
    parser.add_argument("--tasks", nargs="+", default=["arc_challenge", "mmlu", "openbookqa", "winogrande"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_max_length", type=int, default=2048)
    parser.add_argument("--heatmap_layers", type=str, default="8,16,24")
    parser.add_argument(
        "--heatmap_layer_index_base",
        type=int,
        default=1,
        choices=[0, 1],
        help="Interpret heatmap layer ids as 0-based or 1-based.",
    )
    parser.add_argument("--heatmap_seq_len", type=int, default=512)
    parser.add_argument("--heatmap_d_start", type=int, default=0, help="Start channel index (inclusive).")
    parser.add_argument("--heatmap_d_end", type=int, default=-1, help="End channel index (exclusive); <=0 means full D.")
    parser.add_argument("--heatmap_dir", type=str, default=None)
    parser.add_argument(
        "--heatmap_stage",
        type=str,
        default="dense",
        choices=["dense", "sparse"],
        help="Which stage to capture heatmaps from. dense=before sparsification, sparse=after sparsification.",
    )
    parser.add_argument(
        "--heatmap_log_min_exp",
        type=int,
        default=5,
        help="Log-scale lower bound exponent E for heatmap color: [1e-E, 1].",
    )
    parser.add_argument(
        "--heatmap_square_s",
        type=int,
        default=0,
        help="If >0, also export square heatmaps for head/tail S windows with this size.",
    )
    parser.add_argument(
        "--heatmap_square_d",
        type=int,
        default=64,
        help="D width for square head/tail heatmaps.",
    )
    args = parser.parse_args()

    model, tokenizer = load_cats_sparse_model(args.model_path, args.dtype, args.attn_implementation)
    n_layers = len(model.model.layers)

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

    user_layers = parse_layers(args.heatmap_layers)
    if args.heatmap_layer_index_base == 0:
        target_layers_0idx = [x for x in user_layers if 0 <= x < n_layers]
    else:
        target_layers_0idx = [x - 1 for x in user_layers if 1 <= x <= n_layers]
    captured = {}
    token_ids = build_eval_token_ids(tokenizer, args.ppl_split, args.ppl_samples)

    disable_sparse_silu(model)
    dense_handles = []
    if args.heatmap_stage == "dense":
        dense_handles, _, captured = attach_input_hooks(
            model,
            set(target_layers_0idx),
            args.heatmap_seq_len,
            collect_stats=False,
            collect_capture=True,
            heatmap_d_start=args.heatmap_d_start,
            heatmap_d_end=args.heatmap_d_end,
        )
    dense_ppl = eval_ppl_from_ids(model, token_ids, args.context_size, args.window_size)
    for h in dense_handles:
        h.remove()

    enable_sparse_silu(model)
    activate_stats(model, is_collect_histogram=False)
    if args.heatmap_stage == "sparse":
        sparse_handles, hook_stats, captured = attach_input_hooks(
            model,
            set(target_layers_0idx),
            args.heatmap_seq_len,
            collect_stats=True,
            collect_capture=True,
            heatmap_d_start=args.heatmap_d_start,
            heatmap_d_end=args.heatmap_d_end,
        )
    else:
        sparse_handles, hook_stats, _ = attach_input_hooks(
            model,
            set(),
            args.heatmap_seq_len,
            collect_stats=True,
            collect_capture=False,
            heatmap_d_start=args.heatmap_d_start,
            heatmap_d_end=args.heatmap_d_end,
        )
    sparse_ppl = eval_ppl_from_ids(model, token_ids, args.context_size, args.window_size)
    for h in sparse_handles:
        h.remove()

    measured_total_sparsity, layer_sparsities = print_dead_neuron_stats(model)
    deactivate_stats(model)

    per_layer_ratio, overall_ratio = summarize_stats(hook_stats)
    heatmap_dir = args.heatmap_dir or os.path.join(os.path.dirname(args.model_path), "cats_heatmaps")
    heatmaps = save_heatmaps(
        captured,
        heatmap_dir,
        args.heatmap_log_min_exp,
        square_s=args.heatmap_square_s,
        square_d=args.heatmap_square_d,
    )

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
        "dense_ppl_wikitext2": dense_ppl,
        "sparse_ppl_wikitext2": sparse_ppl,
        "measured_total_sparsity_percent": measured_total_sparsity,
        "per_layer_sparse_mlp_sparsity_percent": layer_sparsities,
        "x_zero_ratio_before_xw": {
            "per_layer": per_layer_ratio,
            "overall": overall_ratio,
            "note": "Ratios are measured on linear inputs X before XW over sparse PPL pass.",
        },
        "heatmap_stage": args.heatmap_stage,
        "heatmap_layer_index_base": int(args.heatmap_layer_index_base),
        "heatmap_log_scale": {
            "vmax": 1.0,
            "vmin": 10.0 ** (-int(args.heatmap_log_min_exp)),
        },
        "heatmap_square_window": {
            "enabled": bool(int(args.heatmap_square_s) > 0),
            "s": int(args.heatmap_square_s),
            "d": int(args.heatmap_square_d),
        },
        "heatmap_targets_user_layers_1idx": user_layers,
        "heatmap_d_range": [int(args.heatmap_d_start), int(args.heatmap_d_end)],
        "heatmaps": heatmaps,
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
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

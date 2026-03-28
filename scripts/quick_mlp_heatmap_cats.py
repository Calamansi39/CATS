import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
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
    enable_sparse_silu,
    get_sparse_config,
    set_sparse_threshold,
)
from utils.constants import MISTRAL  # noqa: E402
from utils.utils import get_model_type_from_name  # noqa: E402


def parse_dtype(dtype: str):
    k = dtype.lower()
    if k in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if k in ["fp16", "float16", "half"]:
        return torch.float16
    if k in ["fp32", "float32"]:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def parse_layers(text: str):
    out = []
    for p in text.split(","):
        p = p.strip()
        if p:
            out.append(int(p))
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
        t = row["text"]
        if t and t.strip():
            texts.append(t)
            if max_samples > 0 and len(texts) >= max_samples:
                break

    device = first_device(model)
    activate_stats(model, is_collect_histogram=True)
    enable_sparse_silu(model)
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
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


def get_non_empty_text(split: str, sample_index: int, concat_texts: int):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    valid = []
    for row in ds:
        t = row["text"]
        if t and t.strip():
            valid.append(t)
    idx = max(0, min(sample_index, len(valid) - 1))
    n = max(1, int(concat_texts))
    text = "\n\n".join(valid[idx : min(len(valid), idx + n)])
    return text, idx, len(valid), min(len(valid), idx + n) - idx


def attach_mlp_hooks(model, target_layers_0idx, seq_len, d_start, d_end):
    captured = {}
    handles = []

    for i, layer in enumerate(model.model.layers):
        if i not in target_layers_0idx:
            continue

        def _slice_tensor(x):
            if not torch.is_tensor(x):
                return None
            if x.dim() != 3 or x.size(0) < 1:
                return None
            x0 = x.detach()[0]
            if seq_len > 0:
                x0 = x0[:seq_len]
            d0 = max(0, int(d_start))
            d1 = int(d_end)
            if d1 <= 0 or d1 > x0.size(1):
                d1 = x0.size(1)
            if d0 >= d1:
                d0 = 0
                d1 = x0.size(1)
            return x0[:, d0:d1].float().cpu()

        def _sparse_act_hook(layer_idx):
            def _hook(_module, _inputs, output):
                key = (layer_idx, "mlp_sparse_act")
                if key in captured:
                    return
                x = output[0] if isinstance(output, tuple) else output
                x = _slice_tensor(x)
                if x is not None:
                    captured[key] = x

            return _hook

        def _down_pre_hook(layer_idx):
            def _hook(_module, inputs):
                key = (layer_idx, "mlp_down_input")
                if key in captured or not inputs:
                    return
                x = _slice_tensor(inputs[0])
                if x is not None:
                    captured[key] = x

            return _hook

        handles.append(layer.mlp.sparse_act_fn.register_forward_hook(_sparse_act_hook(i)))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(_down_pre_hook(i)))

    return handles, captured


def save_heatmaps(captured, output_dir, log_min_exp, square_s, square_d):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    exp = max(1, int(log_min_exp))
    vmin = 10.0 ** (-exp)
    ticks = [10.0 ** (-k) for k in range(exp, 0, -1)] + [1.0]

    for (layer_idx, name), x in sorted(captured.items(), key=lambda kv: kv[0]):
        x_abs = x.abs()
        maxv = float(x_abs.max().item())
        x_norm = x_abs / maxv if maxv > 0 else x_abs
        x_plot = torch.clamp(x_norm, min=vmin, max=1.0)
        zero_mask = (x == 0).float()

        x_ds = x_plot.numpy().T  # y=D, x=S
        z_ds = zero_mask.numpy().T
        s_len, d_len = int(x.shape[0]), int(x.shape[1])
        fig_h = 6.0
        fig_w = max(8.0, min(40.0, fig_h * (s_len / max(d_len, 1))))

        base = f"layer{layer_idx:02d}_{name}"
        norm_path = output_dir / f"{base}_norm01_xS_yD.png"
        zero_path = output_dir / f"{base}_zero_mask_xS_yD.png"

        plt.figure(figsize=(fig_w, fig_h))
        plt.imshow(x_ds, aspect="equal", cmap="magma", norm=LogNorm(vmin=vmin, vmax=1.0))
        cbar = plt.colorbar()
        cbar.set_ticks(ticks)
        plt.xlabel("S")
        plt.ylabel("D")
        plt.title(f"{base} normalized |x| (x=S, y=D)")
        plt.tight_layout()
        plt.savefig(norm_path, dpi=220)
        plt.close()

        plt.figure(figsize=(fig_w, fig_h))
        plt.imshow(z_ds, aspect="equal", cmap="gray_r", vmin=0, vmax=1)
        plt.colorbar()
        plt.xlabel("S")
        plt.ylabel("D")
        plt.title(f"{base} zero mask (X==0) (x=S, y=D)")
        plt.tight_layout()
        plt.savefig(zero_path, dpi=220)
        plt.close()

        item = {
            "layer_0idx": layer_idx,
            "name": name,
            "shape_sd": [s_len, d_len],
            "zero_ratio": float((x == 0).float().mean().item()),
            "norm01_heatmap": str(norm_path),
            "zero_mask_heatmap": str(zero_path),
        }

        if square_s > 0 and square_d > 0 and s_len >= square_s and d_len >= square_d:
            x_head = x_plot[:square_s, :square_d].numpy().T
            x_tail = x_plot[-square_s:, :square_d].numpy().T
            z_head = zero_mask[:square_s, :square_d].numpy().T
            z_tail = zero_mask[-square_s:, :square_d].numpy().T

            head_path = output_dir / f"{base}_head_s{square_s}_d{square_d}_norm01_xS_yD.png"
            tail_path = output_dir / f"{base}_tail_s{square_s}_d{square_d}_norm01_xS_yD.png"
            head_zero_path = output_dir / f"{base}_head_s{square_s}_d{square_d}_zero_xS_yD.png"
            tail_zero_path = output_dir / f"{base}_tail_s{square_s}_d{square_d}_zero_xS_yD.png"

            plt.figure(figsize=(6, 6))
            plt.imshow(x_head, aspect="equal", cmap="magma", norm=LogNorm(vmin=vmin, vmax=1.0))
            cbar = plt.colorbar()
            cbar.set_ticks(ticks)
            plt.xlabel("S")
            plt.ylabel("D")
            plt.title(f"{base} head S[0:{square_s}]")
            plt.tight_layout()
            plt.savefig(head_path, dpi=240)
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.imshow(x_tail, aspect="equal", cmap="magma", norm=LogNorm(vmin=vmin, vmax=1.0))
            cbar = plt.colorbar()
            cbar.set_ticks(ticks)
            plt.xlabel("S")
            plt.ylabel("D")
            plt.title(f"{base} tail S[-{square_s}:]")
            plt.tight_layout()
            plt.savefig(tail_path, dpi=240)
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.imshow(z_head, aspect="equal", cmap="gray_r", vmin=0, vmax=1)
            plt.colorbar()
            plt.xlabel("S")
            plt.ylabel("D")
            plt.title(f"{base} head zero mask")
            plt.tight_layout()
            plt.savefig(head_zero_path, dpi=240)
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.imshow(z_tail, aspect="equal", cmap="gray_r", vmin=0, vmax=1)
            plt.colorbar()
            plt.xlabel("S")
            plt.ylabel("D")
            plt.title(f"{base} tail zero mask")
            plt.tight_layout()
            plt.savefig(tail_zero_path, dpi=240)
            plt.close()

            item.update(
                {
                    "square_shape_sd": [int(square_s), int(square_d)],
                    "square_head_norm01_heatmap": str(head_path),
                    "square_tail_norm01_heatmap": str(tail_path),
                    "square_head_zero_mask": str(head_zero_path),
                    "square_tail_zero_mask": str(tail_zero_path),
                }
            )

        written.append(item)

    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--calib_split", type=str, default="train")
    parser.add_argument("--calib_samples", type=int, default=16)
    parser.add_argument("--calib_max_length", type=int, default=1024)
    parser.add_argument("--calib_batch_size", type=int, default=8)
    parser.add_argument("--sample_split", type=str, default="test")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--sample_concat_texts", type=int, default=1)
    parser.add_argument("--sample_max_length", type=int, default=512)
    parser.add_argument("--heatmap_layers", type=str, default="0,8,16,24")
    parser.add_argument("--heatmap_layer_index_base", type=int, default=0, choices=[0, 1])
    parser.add_argument("--heatmap_seq_len", type=int, default=0)
    parser.add_argument("--heatmap_d_start", type=int, default=0)
    parser.add_argument("--heatmap_d_end", type=int, default=64)
    parser.add_argument("--heatmap_log_min_exp", type=int, default=5)
    parser.add_argument("--heatmap_square_s", type=int, default=64)
    parser.add_argument("--heatmap_square_d", type=int, default=64)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    model, tokenizer = load_cats_sparse_model(args.model_path, args.dtype, args.attn_implementation)

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
    enable_sparse_silu(model)

    n_layers = len(model.model.layers)
    user_layers = parse_layers(args.heatmap_layers)
    if args.heatmap_layer_index_base == 0:
        target_layers = [x for x in user_layers if 0 <= x < n_layers]
    else:
        target_layers = [x - 1 for x in user_layers if 1 <= x <= n_layers]

    hooks, captured = attach_mlp_hooks(
        model,
        set(target_layers),
        seq_len=args.heatmap_seq_len,
        d_start=args.heatmap_d_start,
        d_end=args.heatmap_d_end,
    )

    text, used_idx, n_valid, used_n = get_non_empty_text(
        args.sample_split, args.sample_index, args.sample_concat_texts
    )
    device = first_device(model)
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=args.sample_max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    attn = attn.to(device) if attn is not None else None

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attn, use_cache=False)

    for h in hooks:
        h.remove()

    heatmaps = save_heatmaps(
        captured,
        args.output_dir,
        log_min_exp=args.heatmap_log_min_exp,
        square_s=args.heatmap_square_s,
        square_d=args.heatmap_square_d,
    )

    out = {
        "model_path": args.model_path,
        "dtype": args.dtype,
        "target_sparsity_p": args.sparsity,
        "sample": {
            "split": args.sample_split,
            "requested_index_non_empty": args.sample_index,
            "used_index_non_empty": used_idx,
            "concat_texts": used_n,
            "num_non_empty_in_split": n_valid,
            "tokenized_length": int(input_ids.size(1)),
        },
        "heatmap_targets_user": user_layers,
        "heatmap_targets_resolved_0idx": target_layers,
        "captured_names": ["mlp_sparse_act", "mlp_down_input"],
        "heatmaps": heatmaps,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

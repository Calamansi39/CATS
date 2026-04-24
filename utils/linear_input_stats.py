import json
import os
import re

import torch


_ACTIVE_LOGGER = None
_PROJ_ORDER = ["q", "k", "v", "o", "up", "gate", "down"]


def _parse_tag(tag):
    match = re.fullmatch(r"layer_(\d+)\.(q|k|v|o|up|gate|down)", tag)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


class LinearInputStatsLogger:
    def __init__(self, output_json, num_samples=None, seq_len=None):
        self.output_json = output_json
        self.num_samples = num_samples
        self.seq_len = seq_len
        self._stats = {}

    @torch.no_grad()
    def observe(self, tag, x):
        parsed = _parse_tag(tag)
        if parsed is None or not torch.is_tensor(x) or x.numel() == 0:
            return

        layer_idx, proj = parsed
        entry = self._stats.setdefault(
            (layer_idx, proj),
            {
                "zero_count": 0,
                "total_count": 0,
                "two_four_count": 0,
                "two_four_exact_count": 0,
                "group_count": 0,
                "calls": 0,
            },
        )

        x = x.detach()
        entry["calls"] += 1
        entry["zero_count"] += int((x == 0).sum().item())
        entry["total_count"] += int(x.numel())

        if x.shape[-1] >= 4:
            width = (x.shape[-1] // 4) * 4
            if width > 0:
                groups = x[..., :width].reshape(-1, 4)
                zero_groups = (groups == 0).sum(dim=-1)
                entry["two_four_count"] += int((zero_groups >= 2).sum().item())
                entry["two_four_exact_count"] += int((zero_groups == 2).sum().item())
                entry["group_count"] += int(zero_groups.numel())

    def dump(self):
        max_layer = max((layer_idx for layer_idx, _ in self._stats.keys()), default=-1)
        layers = []
        for layer_idx in range(max_layer + 1):
            layer_entry = {"layer": layer_idx}
            for proj in _PROJ_ORDER:
                stats = self._stats.get((layer_idx, proj))
                if stats is None:
                    layer_entry[proj] = {
                        "zero_ratio": 0.0,
                        "two_four_ratio": 0.0,
                        "two_four_exact_ratio": 0.0,
                        "numel": 0,
                        "group_count": 0,
                        "calls": 0,
                    }
                    continue
                total = stats["total_count"]
                groups = stats["group_count"]
                layer_entry[proj] = {
                    "zero_ratio": float(stats["zero_count"] / total) if total else 0.0,
                    "two_four_ratio": float(stats["two_four_count"] / groups) if groups else 0.0,
                    "two_four_exact_ratio": float(stats["two_four_exact_count"] / groups) if groups else 0.0,
                    "numel": int(total),
                    "group_count": int(groups),
                    "calls": int(stats["calls"]),
                }
            layers.append(layer_entry)

        summary = {
            "num_samples": self.num_samples,
            "seq_len": self.seq_len,
            "definition": {
                "zero_ratio": "fraction of exact zeros in the input tensor before each linear layer",
                "two_four_ratio": "fraction of contiguous groups of 4 along the last dimension with at least 2 zeros",
                "two_four_exact_ratio": "fraction of contiguous groups of 4 along the last dimension with exactly 2 zeros",
            },
            "layers": layers,
        }
        os.makedirs(os.path.dirname(self.output_json), exist_ok=True)
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return self.output_json


def set_linear_input_stats_logger(logger):
    global _ACTIVE_LOGGER
    _ACTIVE_LOGGER = logger


def clear_linear_input_stats_logger():
    global _ACTIVE_LOGGER
    _ACTIVE_LOGGER = None


@torch.no_grad()
def record_linear_input_stats(tag, x):
    if _ACTIVE_LOGGER is not None:
        _ACTIVE_LOGGER.observe(tag, x)


def dump_linear_input_stats_logger():
    if _ACTIVE_LOGGER is None:
        return None
    return _ACTIVE_LOGGER.dump()

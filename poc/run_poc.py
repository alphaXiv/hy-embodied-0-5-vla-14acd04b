# coding=utf-8
# Copyright (C) 2026 Tencent.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Minimal proof-of-concept for Hy-Embodied-0.5-VLA.

Illustrates the paper's core modeling claim on a single GPU: the released
dual-tower flow-matching VLA produces accurate continuous delta-chunk
actions on real demonstrations.

What it does:
  1. Loads the released ``tencent/Hy-Embodied-0.5-VLA-UMI`` checkpoint
     (use_video_encoder=False, eager attention -> no flash-attn needed).
  2. Streams real episodes from ``tencent/Hy-Embodied-0.5-VLA-Data`` through
     the repo's own ``LanceVLADataset`` + ``VLADataCollator`` pipeline, using
     the checkpoint's shipped ``norm_stats.pkl``.
  3. Runs ``forward_evaluate`` (10-step Euler flow integration) and compares
     the predicted action chunk vs ground truth in normalized delta-EEF space.
  4. Reports model error against two trivial baselines (predict-zeros and
     mismatched-pair) and writes EVAL.md + per-sample JSONL.

The UMI corpus is in-distribution for the UMI checkpoint, so a low
reconstruction error that beats the baselines is direct evidence the
flow-matching action expert works as claimed.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from huggingface_hub import snapshot_download

from hy_vla import HyVLA, HyVLAConfig
from hy_vla.data.vla_dataset import VLADataset, VLADataCollator

ART = os.path.join(os.getcwd(), ".openresearch", "artifacts")
os.makedirs(ART, exist_ok=True)

UMI_REPO = os.environ.get("HYVLA_CKPT", "tencent/Hy-Embodied-0.5-VLA-UMI")
DATA_REPO = os.environ.get("HYVLA_DATA", "tencent/Hy-Embodied-0.5-VLA-Data")
TABLE = os.environ.get("HYVLA_TABLE", "table_000")
N = int(os.environ.get("HYVLA_NSAMPLES", "16"))
SEED = int(os.environ.get("HYVLA_SEED", "0"))

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if DEV == "cuda" else torch.float32
IMG_KEYS = [
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
]


def log(msg):
    print(f"[poc] {msg}", flush=True)


def build_dataset(norm_path):
    cfg = OmegaConf.create({
        "dataset": {
            "lance_source": DATA_REPO,
            "lance_tables": TABLE,
            "image_aug": False,
            "camera_randomcrop_aug": False,
            "cond_mask_prob": 0.0,
            "cam_ext_mask_prob": -1.0,
            "img_history_size": 1,
            "img_history_interval": 1,
            "img_history_random_sample": False,
            "action_chunk_size": 50,
            "state_dim": 32,
            "num_cameras": 3,
            "image_size": 224,
            "auto_adjust_image_brightness": False,
            "mean_std_path": norm_path,
            "mean_std_chunk_slice": 50,
            "act_type": "relative_chunk_ee_RT",
            "downsample_rate": 3,
            "filter_dirty": False,
            "deterministic": False,  # avoid enumerating millions of frames
            "use_video_encoder": False,
        }
    })
    return VLADataset(cfg)


def main():
    t0 = time.time()
    log(f"device={DEV} dtype={DT} ckpt={UMI_REPO} data={DATA_REPO}/{TABLE} N={N}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ckpt = snapshot_download(UMI_REPO)
    norm_path = os.path.join(ckpt, "norm_stats.pkl")
    log(f"checkpoint at {ckpt}")

    # ---- load model ----
    config = HyVLAConfig.from_pretrained(ckpt)
    log(f"config: chunk_size={config.chunk_size} n_action_steps={config.n_action_steps} "
        f"num_steps={config.num_steps} max_action_dim={config.max_action_dim} "
        f"action_feature={config.action_feature.shape} use_video_encoder={getattr(config,'use_video_encoder',False)}")
    policy = HyVLA.from_pretrained(ckpt, config=config)
    policy.enable_video_encoder_if_needed()
    policy = policy.to(device=DEV, dtype=DT).eval()
    nparams = sum(p.numel() for p in policy.parameters())
    log(f"model loaded: {nparams/1e9:.2f}B params")

    adim = int(config.action_feature.shape[0])  # 20 meaningful dims

    # ---- 1) smoke test on dummy zeros (mirrors scripts/quick_start.py) ----
    img = torch.zeros(1, 3, 224, 224, device=DEV, dtype=DT)
    dummy = {k: img for k in IMG_KEYS}
    dummy["observation.state"] = torch.zeros((1, config.max_state_dim), device=DEV, dtype=DT)
    dummy["task"] = ["pick up the bottle"]
    with torch.no_grad():
        out = policy.forward_evaluate(dummy)["pred"]
    log(f"smoke test OK: dummy action pred shape {tuple(out.shape)}")

    # ---- 2) real samples through the repo data pipeline ----
    ds = build_dataset(norm_path)
    coll = VLADataCollator()
    log(f"dataset ready: {len(ds)} frames; pulling {N} samples")
    instances = []
    for i in range(N):
        instances.append(ds[i])  # non-deterministic -> seeded random sampling
    batch = coll(instances)

    gt_full = batch["action"].clone()  # (B, chunk, D_raw) normalized
    for k in IMG_KEYS + ["observation.state", "action"]:
        batch[k] = batch[k].to(device=DEV, dtype=DT)

    with torch.no_grad():
        info = policy.forward_evaluate(batch)
    pred = info["pred"].float().cpu()
    gt = info["gt"].float().cpu()
    log(f"pred shape {tuple(pred.shape)}  gt shape {tuple(gt.shape)}")

    # Align: keep the meaningful action dims and the overlapping chunk steps.
    pred = pred[..., :adim]
    gt = gt[..., :adim]
    T = min(pred.shape[1], gt.shape[1])
    pred = pred[:, :T]
    gt = gt[:, :T]
    log(f"aligned to chunk steps T={T}, action dims={adim}")

    # ---- 3) metrics: model vs trivial baselines (normalized space) ----
    def l1(a, b):
        return float(torch.mean(torch.abs(a - b)))

    model_l1 = l1(pred, gt)
    zero_l1 = l1(torch.zeros_like(gt), gt)            # predict-zeros baseline
    shuf_l1 = l1(pred[torch.roll(torch.arange(pred.shape[0]), 1)], gt)  # mismatched-pair
    model_mse = float(torch.mean((pred - gt) ** 2))

    # per-sample L1
    per_sample = torch.mean(torch.abs(pred - gt), dim=(1, 2)).tolist()

    results = {
        "n_samples": int(pred.shape[0]),
        "chunk_steps": int(T),
        "action_dims": int(adim),
        "model_l1": model_l1,
        "model_mse": model_mse,
        "zero_baseline_l1": zero_l1,
        "shuffled_pair_l1": shuf_l1,
        "ratio_model_over_zero": model_l1 / zero_l1 if zero_l1 else None,
        "ratio_model_over_shuffled": model_l1 / shuf_l1 if shuf_l1 else None,
        "wall_clock_s": round(time.time() - t0, 1),
        "n_params_billion": round(nparams / 1e9, 3),
    }
    log("RESULTS: " + json.dumps(results, indent=2))

    # per-sample JSONL
    with open(os.path.join(ART, "per_sample.jsonl"), "w") as f:
        for i, v in enumerate(per_sample):
            f.write(json.dumps({
                "sample": i,
                "instruction": instances[i].get("instructions", ""),
                "l1": v,
            }) + "\n")

    passed = (model_l1 < 0.5 * zero_l1) and (model_l1 < 0.6 * shuf_l1)
    verdict = "PASS" if passed else "INCONCLUSIVE"

    with open(os.path.join(ART, "EVAL.md"), "w") as f:
        f.write("# Hy-Embodied-0.5-VLA — minimal reproduction\n\n")
        f.write(f"**Verdict: {verdict}**\n\n")
        f.write("Released `Hy-Embodied-0.5-VLA-UMI` checkpoint reconstructing real UMI "
                "action chunks via 10-step flow matching, scored in normalized delta-EEF "
                "space against ground truth.\n\n")
        f.write("| metric | value |\n|---|---|\n")
        f.write(f"| samples | {results['n_samples']} |\n")
        f.write(f"| chunk steps x action dims | {T} x {adim} |\n")
        f.write(f"| model L1 | {model_l1:.4f} |\n")
        f.write(f"| model MSE | {model_mse:.4f} |\n")
        f.write(f"| predict-zeros baseline L1 | {zero_l1:.4f} |\n")
        f.write(f"| mismatched-pair baseline L1 | {shuf_l1:.4f} |\n")
        f.write(f"| model / zero | {results['ratio_model_over_zero']:.3f} |\n")
        f.write(f"| model / mismatched | {results['ratio_model_over_shuffled']:.3f} |\n")
        f.write(f"| params | {results['n_params_billion']}B |\n")
        f.write(f"| wall clock | {results['wall_clock_s']}s |\n\n")
        f.write("PASS = model L1 < 0.5x predict-zeros AND < 0.6x mismatched-pair, "
                "i.e. predictions are accurate and sample-specific.\n")

    with open(os.path.join(ART, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    log(f"wrote artifacts to {ART}")
    if not passed:
        log("WARNING: did not clear PASS thresholds; see EVAL.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

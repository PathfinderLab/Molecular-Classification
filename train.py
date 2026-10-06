#!/usr/bin/env python3
"""Train MIL models on the fixed patient-disjoint TCGA train/validation split."""

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch

from mil_common import (DATA, HERE, MODELS, SPLIT_ID, balanced_patient_accuracy,
                        check_feature_files, feature, feature_path, forward,
                        make_model, sha, validate_split)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--tcga-features", default=os.environ.get("TCGA_FEATURE_DIR"),
                        help="Directory containing <TCGA slide ID>.h5 UNI feature files")
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--min-select-epoch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--smoke", action="store_true", help="One forward/backward update, no output")
    args = parser.parse_args()
    if not args.tcga_features:
        parser.error("Set --tcga-features or TCGA_FEATURE_DIR")
    assert args.min_select_epoch <= args.min_epochs <= args.max_epochs
    assert torch.cuda.is_available(), "CUDA GPU required for this training recipe"
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")

    train, validation, _, _ = validate_split()
    check_feature_files(train + validation, args.tcga_features)
    by_class = {cls: defaultdict(list) for cls in range(4)}
    for row in train:
        by_class[int(row["label"])][row["patient_id"]].append(row)
    assert all(by_class.values())

    model = make_model(args.model, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    run = HERE / "runs" / f"{args.model}_seed{args.seed}"
    config = {
        "split_id": SPLIT_ID, "model": args.model, "seed": args.seed,
        "train_csv_sha256": sha(DATA / "train.csv"),
        "validation_csv_sha256": sha(DATA / "validation.csv"),
        "train_slides": len(train), "validation_slides": len(validation),
        "optimizer": "AdamW", "peak_lr": 1e-4, "weight_decay": 1e-3,
        "effective_batch": 16,
        "sampling": "class-balanced patient sampling; one slide from selected patient per draw",
        "lr_schedule": "cosine, 100-epoch horizon, floor 1e-6",
        "checkpoint_selection": "highest validation patient balanced accuracy; earliest tie",
        "min_select_epoch": args.min_select_epoch, "min_epochs": args.min_epochs,
        "max_epochs": args.max_epochs, "patience": args.patience,
        "acmil": {"n_token": 5, "n_masked_patch": 10, "mask_drop": 0.6}
                 if args.model == "acmil" else None,
    }
    if not args.smoke:
        run.mkdir(parents=True, exist_ok=True)
        if (run / "best.pt").exists():
            raise FileExistsError(f"Refusing to overwrite existing run: {run}")
        (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    steps = 16 * math.ceil(len({row["patient_id"] for row in train}) / 16)
    start = time.monotonic()
    best, best_epoch = -1, 0
    for epoch in range(1, 2 if args.smoke else args.max_epochs + 1):
        model.train()
        rng = random.Random(args.seed + epoch * 100)
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for step in range(1 if args.smoke else steps):
            fraction = ((epoch - 1) + step / steps) / 100
            lr = 1e-6 + 0.5 * (1e-4 - 1e-6) * (1 + math.cos(math.pi * fraction))
            for group in optimizer.param_groups:
                group["lr"] = lr
            cls = rng.randrange(4)
            patient = rng.choice(sorted(by_class[cls]))
            row = rng.choice(by_class[cls][patient])
            x = feature(feature_path(row, args.tcga_features)).unsqueeze(0).to(device)
            y = torch.tensor([cls], device=device)
            _, loss = forward(model, args.model, x, y)
            (loss if args.smoke else loss / 16).backward()
            total += float(loss.detach())
            if args.smoke or (step + 1) % 16 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if args.smoke:
            print(json.dumps({"smoke": "ok", "model": args.model,
                              "patches": x.shape[1], "loss": total}), flush=True)
            return
        validation_result = balanced_patient_accuracy(
            model, args.model, validation, device, args.tcga_features)
        record = {
            "epoch": epoch, "training_loss": total / steps,
            "validation_patient_balanced_accuracy": validation_result["balanced_accuracy"],
            "validation_recall": validation_result["per_class_recall"],
            "elapsed_seconds": round(time.monotonic() - start, 1),
        }
        with (run / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if epoch >= args.min_select_epoch and validation_result["balanced_accuracy"] > best + 1e-12:
            best = validation_result["balanced_accuracy"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "config": config, "best_epoch": epoch,
                        "validation_patient_balanced_accuracy": best}, run / "best.pt")
        (run / "status.json").write_text(json.dumps({
            "phase": "training", "last_epoch": epoch, "best_epoch": best_epoch,
            "best_validation_balanced_accuracy": best}, indent=2) + "\n")
        if epoch >= args.min_epochs and best_epoch and epoch - best_epoch >= args.patience:
            break
    (run / "status.json").write_text(json.dumps({
        "phase": "complete", "last_epoch": epoch, "best_epoch": best_epoch,
        "best_validation_balanced_accuracy": best, "test_evaluated": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

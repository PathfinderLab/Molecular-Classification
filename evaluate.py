#!/usr/bin/env python3
"""Evaluate a completed run on the fixed TCGA and external DGIST tests."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from mil_common import (DATA, HERE, CLASSES, MODELS, SPLIT_ID, check_feature_files,
                        feature, feature_path, forward, make_model, sha, validate_split)


def score(model, name, rows, run, tcga_features, dgist_features):
    output = []
    with torch.inference_mode():
        for row in rows:
            path = feature_path(row, tcga_features, dgist_features)
            x = feature(path).unsqueeze(0).cuda()
            logits, _ = forward(model, name, x)
            probabilities = logits.softmax(-1)[0].cpu().numpy()
            label = int(row["label"])
            prediction = int(probabilities.argmax())
            output.append({**row, "predicted_id": prediction,
                           "correct": prediction == label,
                           **{f"p_{CLASSES[i]}": float(probabilities[i]) for i in range(4)}})
    cohort = "TCGA119" if rows[0]["cohort"] == "TCGA" else "DGIST38_external"
    with (run / f"{cohort}_predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    y = np.array([int(row["label"]) for row in output])
    p = np.array([[row[f"p_{label}"] for label in CLASSES] for row in output])
    predicted = p.argmax(1)
    confusion = [[int(((y == i) & (predicted == j)).sum()) for j in range(4)] for i in range(4)]
    recall = {CLASSES[i]: confusion[i][i] / sum(confusion[i]) for i in range(4)}
    auc = {CLASSES[i]: float(roc_auc_score(y == i, p[:, i])) for i in range(4)}
    result = {"n": len(output), "patients": len({row["patient_id"] for row in output}),
              "correct": int((y == predicted).sum()), "accuracy": float((y == predicted).mean()),
              "balanced_accuracy": sum(recall.values()) / 4,
              "macro_ovr_auroc": sum(auc.values()) / 4, "per_class_ovr_auroc": auc,
              "per_class_recall": recall, "confusion_matrix": confusion}
    (run / f"{cohort}_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cohort": cohort, **result}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--tcga-features", default=os.environ.get("TCGA_FEATURE_DIR"))
    parser.add_argument("--dgist-features", default=os.environ.get("DGIST_FEATURE_DIR"))
    args = parser.parse_args()
    if not args.tcga_features or not args.dgist_features:
        parser.error("Set --tcga-features and --dgist-features (or their *_FEATURE_DIR variables)")
    assert torch.cuda.is_available(), "CUDA GPU required for this model implementation"
    torch.set_num_threads(2)
    _, _, tcga, dgist = validate_split()
    check_feature_files(tcga + dgist, args.tcga_features, args.dgist_features)
    run = HERE / "runs" / f"{args.model}_seed{args.seed}"
    status = json.loads((run / "status.json").read_text())
    assert status["phase"] == "complete", status
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    assert config["split_id"] == SPLIT_ID and config["model"] == args.model and config["seed"] == args.seed
    assert config["train_csv_sha256"] == sha(DATA / "train.csv")
    assert config["validation_csv_sha256"] == sha(DATA / "validation.csv")
    assert checkpoint["best_epoch"] == status["best_epoch"]
    model = make_model(args.model, torch.device("cuda:0"))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    results = {"TCGA119": score(model, args.model, tcga, run, args.tcga_features, args.dgist_features),
               "DGIST38_external": score(model, args.model, dgist, run, args.tcga_features, args.dgist_features)}
    (run / "test_results.json").write_text(json.dumps(results, indent=2) + "\n")
    status["test_evaluated"] = True
    (run / "status.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()

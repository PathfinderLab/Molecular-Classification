"""Shared utilities for the fixed, patient-disjoint molecular-subtype split."""

import csv
import functools
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE / "models"))
from architecture.transmil_mba import TransMIL_MBA  # noqa: E402
from architecture.transformer import ACMIL_GA  # noqa: E402

CLASSES = ["EBV", "GS", "MSI", "CIN"]
MODELS = ["transmil_mba", "acmil"]
SPLIT_ID = "CLEAN_70_10_20_TCGA_DGIST_20260928_v1"
FIELDS = ["cohort", "slide", "patient_id", "label"]


def read(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == FIELDS, (path, reader.fieldnames)
        return list(reader)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_split():
    groups = [read(DATA / name) for name in (
        "train.csv", "validation.csv", "test_TCGA119.csv", "test_DGIST38_external.csv"
    )]
    assert [len(group) for group in groups] == [454, 64, 119, 38]
    assert all(row["cohort"] == "TCGA" for group in groups[:3] for row in group)
    assert all(row["cohort"] == "DGIST" for row in groups[3])
    patients = []
    slides = []
    for group in groups:
        assert all(row["label"] in {"0", "1", "2", "3"} for row in group)
        assert all(row["slide"] and Path(row["slide"]).name == row["slide"]
                   and "/" not in row["slide"] and "\\" not in row["slide"] for row in group)
        keys = {(row["cohort"], row["slide"]) for row in group}
        assert len(keys) == len(group), "Duplicate slide within split"
        slides.append(keys)
        patients.append({(row["cohort"], row["patient_id"]) for row in group})
    for i in range(4):
        for j in range(i + 1, 4):
            assert not patients[i] & patients[j], "Patient overlap between splits"
            assert not slides[i] & slides[j], "Slide overlap between splits"
    return groups


def feature_path(row, tcga_features, dgist_features=None):
    folder = tcga_features if row["cohort"] == "TCGA" else dgist_features
    if folder is None:
        raise ValueError(f"Feature directory required for {row['cohort']}")
    return Path(folder) / f"{row['slide']}.h5"


def check_feature_files(rows, tcga_features, dgist_features=None):
    missing = [str(path) for row in rows
               if not (path := feature_path(row, tcga_features, dgist_features)).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing feature files; first: {missing[0]}")


@functools.lru_cache(maxsize=1024)
def feature(path):
    with h5py.File(path) as handle:
        assert handle.attrs.get("complete", False), path
        array = handle["features"][:]
    assert array.ndim == 2 and array.shape[1] == 1024 and np.isfinite(array).all(), (path, array.shape)
    return torch.from_numpy(array).float()


def make_model(name, device):
    assert name in MODELS
    if name == "transmil_mba":
        cfg = SimpleNamespace(D_feat=1024, D_inner=512, n_class=4, n_token=2)
        model = TransMIL_MBA(cfg, n_token=2)
    else:
        cfg = SimpleNamespace(D_feat=1024, D_inner=512, n_class=4, n_token=5)
        model = ACMIL_GA(cfg, n_token=5, n_masked_patch=10, mask_drop=0.6)
    return model.to(device)


def forward(model, name, x, y=None):
    branch, logits, attention = model(x)
    n = 2 if name == "transmil_mba" else 5
    assert branch.shape == (n, 4) and logits.shape == (1, 4)
    if y is None:
        return logits, None
    ce = torch.nn.functional.cross_entropy
    weights = attention.softmax(-1)
    diversity = sum(torch.cosine_similarity(weights[:, i], weights[:, j], dim=-1).mean()
                    for i in range(n) for j in range(i + 1, n)) / (n * (n - 1) / 2)
    loss = ce(logits, y) + ce(branch, y.repeat(n)) + diversity
    assert torch.isfinite(loss)
    return logits, loss


def balanced_patient_accuracy(model, name, rows, device, tcga_features):
    model.eval()
    grouped = defaultdict(list)
    with torch.inference_mode():
        for row in rows:
            path = feature_path(row, tcga_features)
            x = feature(path).unsqueeze(0).to(device)
            logits, _ = forward(model, name, x)
            grouped[row["patient_id"]].append((int(row["label"]), logits.softmax(-1)[0].cpu().numpy()))
    truth, predictions = [], []
    for group in grouped.values():
        assert len({value[0] for value in group}) == 1
        truth.append(group[0][0])
        predictions.append(int(np.mean([value[1] for value in group], axis=0).argmax()))
    recalls = []
    for cls in range(4):
        indices = [i for i, label in enumerate(truth) if label == cls]
        assert indices, ("Missing validation class", cls)
        recalls.append(sum(predictions[i] == cls for i in indices) / len(indices))
    return {"patients": len(truth), "balanced_accuracy": sum(recalls) / 4,
            "per_class_recall": dict(zip(CLASSES, recalls))}

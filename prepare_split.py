#!/usr/bin/env python3
"""Verify the published patient-disjoint split and validation assignment."""
import hashlib
import json
from collections import Counter, defaultdict

from mil_common import validate_split

SEED = "TCGA70_10_20_20260928_v1"
PATIENT_QUOTA = {0: 2, 1: 2, 2: 5, 3: 20}


def main():
    train, validation, tcga_test, dgist_test = validate_split()
    pool = train + validation
    patients = defaultdict(list)
    for row in pool:
        assert row["cohort"] == "TCGA" and row["label"] in {"0", "1", "2", "3"}
        patients[row["patient_id"]].append(row)
    by_class = defaultdict(list)
    for patient, rows in patients.items():
        assert len({row["label"] for row in rows}) == 1, patient
        by_class[int(rows[0]["label"])].append(patient)
    assert len(patients) == 235
    selected = None
    for salt in range(100000):
        chosen = set()
        for cls, quota in PATIENT_QUOTA.items():
            ranked = sorted(by_class[cls], key=lambda patient:
                            hashlib.sha256(f"{SEED}|{salt}|{cls}|{patient}".encode()).hexdigest())
            chosen.update(ranked[:quota])
        if sum(len(patients[patient]) for patient in chosen) == 64:
            selected = salt, chosen
            break
    assert selected is not None
    salt, selected_patients = selected
    assert {row["patient_id"] for row in validation} == selected_patients
    assert all(row["patient_id"] not in selected_patients for row in train)
    assert len({row["patient_id"] for row in train}) == 206
    assert len({row["patient_id"] for row in validation}) == 29
    assert {(row["cohort"], row["patient_id"]) for row in pool}.isdisjoint(
        {(row["cohort"], row["patient_id"]) for row in tcga_test + dgist_test})
    summary = {"selection_seed": SEED, "selected_salt": salt,
               "label_mapping": {"0": "EBV", "1": "GS", "2": "MSI", "3": "CIN"},
               "patient_quota": PATIENT_QUOTA,
               "splits": {name: {"slides": len(rows),
                                 "patients": len({row["patient_id"] for row in rows}),
                                 "class_counts": dict(Counter(row["label"] for row in rows))}
                          for name, rows in zip(("train", "validation", "test_TCGA119", "test_DGIST38_external"),
                                                (train, validation, tcga_test, dgist_test))}}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

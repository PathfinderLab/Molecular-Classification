#!/usr/bin/env python3
"""Encode pre-extracted 224-pixel RGB slide patches with TRIDENT's UNI v1 encoder."""
import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from mil_common import DATA, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("tcga", "dgist"), required=True)
    parser.add_argument("--patch-h5-dir", type=Path, required=True,
                        help="Folder of <slide>.h5 files with an images dataset [N,224,224,3]")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--uni-weights", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required")
    if not args.uni_weights.is_file():
        parser.error(f"Missing UNI weights: {args.uni_weights}")
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    try:
        from trident.patch_encoder_models import encoder_factory
    except ImportError as exc:
        parser.error(f"Install TRIDENT first: {exc}")
    torch.set_num_threads(2)
    files = ("train.csv", "validation.csv", "test_TCGA119.csv") if args.cohort == "tcga" else ("test_DGIST38_external.csv",)
    rows = [row for name in files for row in read(DATA / name)]
    slides = sorted({row["slide"] for row in rows})
    model = encoder_factory("uni_v1", weights_path=str(args.uni_weights)).eval().to("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, slide in enumerate(slides, 1):
        source_path = args.patch_h5_dir / f"{slide}.h5"
        output_path = args.output_dir / f"{slide}.h5"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        with h5py.File(source_path) as source:
            images = source["images"]
            count = len(images)
            if count == 0 or images.shape[1:] != (224, 224, 3) or images.dtype != np.uint8:
                raise ValueError(f"Expected nonempty uint8 RGB 224x224 patches: {source_path}")
            if output_path.exists():
                with h5py.File(output_path) as previous:
                    if previous.attrs.get("complete", False) and previous["features"].shape == (count, 1024):
                        print(f"[{index}/{len(slides)}] skip {slide}", flush=True)
                        continue
                raise FileExistsError(f"Incomplete/conflicting output requires manual review: {output_path}")
            temporary = output_path.with_suffix(".h5.part")
            if temporary.exists():
                raise FileExistsError(f"Incomplete prior extraction: {temporary}")
            try:
                with h5py.File(temporary, "w") as output:
                    features = output.create_dataset("features", (count, 1024), dtype="float32")
                    if "coords" in source:
                        source.copy("coords", output)
                    for start in range(0, count, args.batch_size):
                        batch = torch.stack([model.eval_transforms(Image.fromarray(image))
                                             for image in images[start:start + args.batch_size]]).to("cuda:0")
                        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                            encoded = model(batch).float().cpu().numpy()
                        if encoded.shape != (len(batch), 1024) or not np.isfinite(encoded).all():
                            raise ValueError(f"Invalid UNI output for {slide}, patches {start}:{start + len(batch)}")
                        features[start:start + len(batch)] = encoded
                    output.attrs["complete"] = True
                    output.attrs["encoder"] = "TRIDENT uni_v1"
                    output.attrs["precision"] = "mixed_float16"
                    output.attrs["patch_filter"] = "all source patches; no tumour segmentation"
                os.replace(temporary, output_path)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        print(f"[{index}/{len(slides)}] encoded {slide}: {count} patches", flush=True)


if __name__ == "__main__":
    main()

# TCGA/DGIST molecular-subtype MIL reproduction package

This repository contains the fixed, patient-disjoint CSV split and self-contained training/evaluation code for **TransMIL-MBA** and **ACMIL-GA** on four gastric molecular subtypes. It is a clean-split experiment package; it does not contain the historical checkpoints or scores from earlier, overlapping splits.

## Data and split

| CSV | Cohort | Slides | Patients | Purpose |
| --- | --- | ---: | ---: | --- |
| `data/train.csv` | TCGA | 454 | 206 | Model fitting |
| `data/validation.csv` | TCGA | 64 | 29 | Checkpoint selection |
| `data/test_TCGA119.csv` | TCGA | 119 | 109 | Fixed internal evaluation list |
| `data/test_DGIST38_external.csv` | DGIST | 38 | 32 | External test |

The TCGA slide proportions are 71.3%/10.0%/18.7% (approximately 70/10/20). All four CSVs are patient-disjoint for training runs made with this package. Labels are `0=EBV`, `1=GS`, `2=MSI`, `3=CIN`. The split and validation assignment can be checked without feature files:

```bash
python3 prepare_split.py
```

## Required features

Both models consume **UNI v1** slide patch features, one HDF5 file per slide. Each feature directory must contain `<slide>.h5` for every corresponding CSV row. Each file must have a `features` dataset shaped `[patches, 1024]` and a true `complete` attribute. TCGA and DGIST may reside in separate directories. Their paths are supplied at runtime and never stored in the CSVs.

The source patch files used for this experiment contain RGB `uint8` images of shape `[patches,224,224,3]`. The newly extracted slides used [TRIDENT](https://github.com/mahmoodlab/TRIDENT) at nominal 20× with 224-pixel patches, zero overlap, Otsu tissue segmentation, and at least 50% tissue coverage. Some pre-existing TCGA patch files were reused; their tissue selection is not guaranteed identical to the newly extracted files. All patches in each selected tumour-sample slide were encoded, with no tumour-only patch filter. The UNI encoder was run with TRIDENT's `uni_v1` evaluation transform and mixed float16 inference. To generate features from already extracted patch HDF5 files, install TRIDENT and provide an authorized UNI v1 checkpoint:

```bash
python3 extract_uni_features.py --cohort tcga --patch-h5-dir /path/to/tcga_patch_h5 --output-dir /path/to/tcga_uni_h5 --uni-weights /path/to/uni_v1_weights.bin
python3 extract_uni_features.py --cohort dgist --patch-h5-dir /path/to/dgist_patch_h5 --output-dir /path/to/dgist_uni_h5 --uni-weights /path/to/uni_v1_weights.bin
```

Raw slides, patches, UNI weights, and UNI features are not bundled. The patch HDF5 files or raw slides must be obtained separately. This package cannot reproduce features from raw slides alone.

The results are reproducible only when the same slide versions, patch coordinates/selection, UNI weights, and feature-extraction settings are used. The patient split prevents MIL-level overlap; any overlap in external UNI pretraining must be assessed separately.

## Environment and run

Install a CUDA-compatible PyTorch build, then the remaining packages in `requirements.txt`. The original runtime used Python with PyTorch 2.11, NumPy 1.26, h5py 3.16, einops 0.8, and scikit-learn 1.7. A CUDA GPU is required by this model implementation.

```bash
python3 -m pip install -r requirements.txt
python3 prepare_split.py
python3 train.py --model transmil_mba --tcga-features /path/to/tcga_uni_h5
python3 evaluate.py --model transmil_mba --tcga-features /path/to/tcga_uni_h5 --dgist-features /path/to/dgist_uni_h5
python3 train.py --model acmil --tcga-features /path/to/tcga_uni_h5
python3 evaluate.py --model acmil --tcga-features /path/to/tcga_uni_h5 --dgist-features /path/to/dgist_uni_h5
```

Alternatively set `TCGA_FEATURE_DIR`, `DGIST_FEATURE_DIR`, and optionally `PYTHON`, then run `bash run_models.sh`. For a one-step hardware and data check, run `python3 train.py --model transmil_mba --tcga-features /path/to/tcga_uni_h5 --smoke` (and likewise for `acmil`). Smoke mode writes no checkpoint.

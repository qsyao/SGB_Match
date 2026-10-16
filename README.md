# SGB-Match

Code and selected fine-stage checkpoints for **Structure-Guided Self-Supervised Matching for One-Shot Medical Landmark Detection** (accepted at IEEE BIBM 2026).

SGB-Match uses structure-guided contrastive learning and a global-to-local matching pipeline to transfer landmarks from one annotated image. The maintained implementation here comes from `code/cc2d-final` in the research workspace.

## Contents

- `code/cc2d-final/`: model, dataset, configuration, evaluation, and coarse/fine SSL scripts.
- `checkpoints/`: one selected **fine-stage** checkpoint for each dataset. These are real files, tracked through Git LFS when published.
- `metadata/manifest.json` and `metadata/metrics.csv`: checkpoint provenance and previously recorded measurements.
- `CITATION.bib`: provisional citation. Add pages, DOI, and final proceedings metadata after publication.

No images, medical datasets, training logs, intermediate checkpoints, or prediction files are included.

## Checkpoints

| Dataset | Best epoch | Fine-stage checkpoint | Recorded MRE (mm) |
| --- | ---: | --- | ---: |
| Cephalometric (head) | 179 | `checkpoints/head.pth` | 1.772831 |
| Hand | 9 | `checkpoints/hand.pth` | 1.177376 |
| Chest | 14 | `checkpoints/chest.pth` | 5.458034 |
| Leg | 89 | `checkpoints/leg.pth` | 2.792324 |

The values are from the recorded evaluation protocol, **not independent held-out estimates**. Checkpoints and local settings were selected using Test results. A later reevaluation of the leg checkpoint gave MRE 2.792614 mm with unchanged SDR. See `metadata/metrics.csv` for both values.

## Environment and data

The source was developed with Python 3.9 and the pinned dependencies in `code/cc2d-final/requirements.txt`. A CUDA GPU is required by the current scripts. From the repository root:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r code/cc2d-final/requirements.txt
```

Provide the four original datasets under `dataset/Cephalometric`, `dataset/hand`, `dataset/chest`, and `dataset/leg` in the layout expected by the dataset loaders. The configs under `code/cc2d-final/config/` use paths relative to `code/cc2d-final`. Dataset licenses and access requirements are governed by their respective providers.

The fine-stage evaluator also needs coarse-stage pseudo-label JSON files under `final_runs/<tag>/coarse_stage/pseudo_labels/`. Those files are **not** in this release. They can be generated with the coarse training/evaluation pipeline; the four included fine-stage checkpoints alone cannot reproduce the reported end-to-end inference. The original coarse checkpoints are not included because this release keeps only the four requested best fine-stage models.

## Training and evaluation

Run commands from `code/cc2d-final` so the relative `../../dataset` and `../../final_runs` paths resolve correctly:

```bash
cd code/cc2d-final
python -m scripts_SSL.train_template --dataset head --tag head_example --coarse_stage 1
python -m scripts_SSL.test_coarse --dataset head --tag head_example --epoch best
python -m scripts_SSL.train_template --dataset head --tag head_example --coarse_stage 0
python -m scripts_SSL.test_fine --dataset head --tag head_example --epoch best --skip_infer_train 1
```

Replace `head` with `hand`, `chest`, or `leg`. For the released weights, put a selected checkpoint at `final_runs/<tag>/fine_stage/model_epoch_best.pth`, provide the matching coarse pseudo-labels under that tag, and use `--epoch best`. The model code constructs a ConvNeXt backbone with `pretrained=True`; the first run may need access to upstream pretrained weights even when loading a released checkpoint.

## Citation

```bibtex
@inproceedings{yao2026sgbmatch,
  title     = {Structure-Guided Self-Supervised Matching for One-Shot Medical Landmark Detection},
  author    = {Yao, Qingsong and Huang, Zhen and Wang, Ao and Wang, Rongsheng and Chen, Wenting and Wang, Jianji and Zhou, S. Kevin},
  booktitle = {Proceedings of the IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year      = {2026},
  note      = {Accepted; publication details to be updated when available}
}
```

The camera-ready source records affiliations for Qingsong Yao (1,4), Zhen Huang (2), Ao Wang (1), Rongsheng Wang (2), Wenting Chen (3), Jianji Wang (7), and S. Kevin Zhou (2,5,6). Authors and order in this citation follow that source and the supplied author list.

## Publication notes

Large `.pth` files require Git LFS. Install Git LFS and run `git lfs install` before staging or pushing checkpoints. This directory does not include a license; reuse permissions remain reserved until the authors select one.

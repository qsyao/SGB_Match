# SGB-Match

Official implementation of **Structure-Guided Self-Supervised Matching for One-Shot Medical Landmark Detection** (IEEE BIBM 2026).

Medical landmark detection identifies anatomically meaningful points in an image, such as points on the skull, hand, lung boundaries, or lower limbs. SGB-Match detects these landmarks using **one annotated template image per dataset**, together with unlabeled training images.

The method learns dense image correspondences through self-supervised contrastive learning. A **structure-guided bias**, constructed from spatial distance and edge cues, helps the learned features distinguish anatomical structures with similar appearances. At inference, a **global matcher** transfers landmarks from the template to the target image, and a **local matcher** refines each prediction within a cropped region.

## Installation

Use Python 3.9 and a CUDA-enabled GPU. From the repository root:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r code/cc2d-final/requirements.txt
```

The model uses ConvNeXtV2-Tiny and SAM-Med2D image features. Download the pretrained SAM-Med2D model from the [official SAM-Med2D repository](https://github.com/OpenGVLab/SAM-Med2D#-results) and place it at:

```text
code/cc2d-final/sam-med2d_b.pth
```

The ConvNeXtV2 pretrained weights are downloaded through `timm` on first use. The current model constructor loads these backbone weights before loading a task checkpoint, so prepare them for both training and evaluation.

## Data Preparation

The four X-ray datasets are available through the original Google Drive links below, including our proposed **BMPLE** dataset for lower-limb landmark detection. Download the files and extract them into the `dataset/` directory at the repository root.

- [head dataset (Cephalometric)](https://drive.google.com/file/d/1PQhNrwZpa9y6AhWjX_qcS_zrptcTCSvq/view?usp=sharing)
- [hand dataset](https://drive.google.com/file/d/16MD617NySbf7EXlkA07RlMnEUAUYgHCt/view?usp=sharing)
- [chest dataset](https://drive.google.com/file/d/1Gkfl5wnaU2J-TMAkOrKM604cmxw9PPiz/view?usp=sharing)
- [leg dataset (BMPLE)](https://drive.google.com/file/d/1MWCWA64xJ9Rt4MpcqinQMkJ2Fs7ekGRr/view?usp=sharing)

### Dataset overview

| Dataset | `--dataset` | Images | Landmarks per image | Training / test | Data directory |
| --- | --- | ---: | ---: | --- | --- |
| Cephalometric | `head` | 400 | 19 | 150 / 250 | `dataset/Cephalometric/` |
| Hand X-ray | `hand` | 909 | 37 | 550 / 300; 59 unused | `dataset/hand/` |
| Chest X-ray | `chest` | 279 | 6 | 195 / 50; 34 unused | `dataset/chest/` |
| BMPLE | `leg` | 190 | 20 | 120 / 70 | `dataset/leg/` |

**Cephalometric (head).** Lateral skull radiographs from the ISBI cephalometric landmark benchmark. Each image has 19 landmarks annotated by two experts; the loader averages their coordinates. Images have a resolution of 1935 × 2400 pixels and a pixel spacing of 0.1 mm. Training uses image IDs 001–150, and testing uses 151–400.

**Hand.** Hand radiographs with 37 landmarks. The loader sorts the eligible image filenames, uses the first 550 for training and the last 300 for testing, and leaves the intervening 59 unused by the training/evaluation commands below. Images are resized to 1500 × 1500 before network preprocessing. Physical error is estimated by assuming that the distance between the two wrist landmarks is 50 mm.

**Chest.** A curated subset of 279 chest radiographs with six landmarks on the lung boundaries. After sorting filenames and excluding generated edge images, the loader uses the first 195 images for training and the last 50 for testing; the intervening 34 are unused. Labels contain normalized coordinates. Evaluation uses the equivalent pixel distance at 512 × 512 resolution: the loader normalizes images to 1024 × 1024 and applies a scale factor of 0.5. This is a fixed-resolution pixel metric, rather than measured physical spacing.

**BMPLE (leg).** Lower-limb radiographs for biomechanical landmark detection, with 20 landmarks per image. Training uses image IDs 1–120, and `test.json` specifies the 70 test images. Each image is accompanied by a metadata JSON containing landmark coordinates, original image dimensions, and pixel spacing. The loader maps images to 2400 × 5800 before network preprocessing and adjusts the spacing accordingly.

### Directory layout and annotations

Arrange the extracted files as follows. Keep the original filenames because the loaders use them to match images and annotations and determine the splits.

```text
dataset/
├── Cephalometric/
│   ├── RawImage/
│   │   ├── TrainingData/       # 001.bmp–150.bmp
│   │   └── Test1Data/          # 151.bmp–400.bmp for the combined test set
│   ├── 400_junior/            # <image_id>.txt
│   └── 400_senior/            # <image_id>.txt
├── hand/
│   ├── jpg/                   # <image_id>.jpg
│   └── all.csv
├── chest/
│   ├── pngs/                  # <image_id>.png
│   └── labels/                # <image_id>.txt
└── leg/
    ├── train/                 # <image_id>.png and <image_id>_meta.json
    ├── test/                  # <image_id>.png and <image_id>_meta.json
    └── test.json
```

For Cephalometric, each annotation file starts with 19 landmark lines whose first field is `x,y`. The default `Test` mode reads all 250 test images from `Test1Data/`; if an archive separates images 301–400 into `Test2Data/`, copy those images into `Test1Data/` as well.

For Hand, `all.csv` has no header: the first column is the numeric image ID, followed by 37 `(x, y)` coordinate pairs in the original image coordinates. For Chest, the first line of each label file is the landmark count, followed by six lines of normalized `x y` coordinates. For BMPLE, retain the supplied metadata: the loader selects 20 landmarks from its `gt` list and uses `original_shape` and `spacing` to transform coordinates and compute errors.

### Configurations and one-shot templates

Dataset paths and landmark counts are defined in `code/cc2d-final/config/<dataset>.yaml`. Training, crop, and template settings are defined in `config/SSL_<dataset>.yaml` in the same directory.

| Dataset | Training configuration | Template image | `test.id_oneshot` | Network input (H × W) |
| --- | --- | --- | --- | --- |
| Head | [SSL_head.yaml](code/cc2d-final/config/SSL_head.yaml) | `125.bmp` | `[125]` | 384 × 384 |
| Hand | [SSL_hand.yaml](code/cc2d-final/config/SSL_hand.yaml) | `3153.jpg` | `[10]` | 384 × 384 |
| Chest | [SSL_chest.yaml](code/cc2d-final/config/SSL_chest.yaml) | `CHNCXR_0173_0.png` | `[165]` | 384 × 384 |
| Leg | [SSL_leg.yaml](code/cc2d-final/config/SSL_leg.yaml) | `118.png` | `[118]` | 768 × 384 |

For Head and Leg, `id_oneshot` is the image ID. For Hand and Chest, it is the **zero-based index in the sorted eligible image list**, rather than the filename. Keep the supplied template settings when evaluating the provided checkpoints.

## Training

All commands below run from the source directory so that dataset and output paths resolve correctly:

```bash
cd code/cc2d-final
```

The example uses `head` and stores outputs under `final_runs/head_example/`. Replace the dataset with `hand`, `chest`, or `leg` and use a separate tag for each experiment.

### 1. Train the global matcher

```bash
python -m scripts_SSL.train_template --dataset head --tag head_example --coarse_stage 1
```

This stage learns correspondences on full images. Checkpoints are saved under `final_runs/head_example/coarse_stage/`.

### 2. Generate coarse landmark predictions

```bash
python -m scripts_SSL.test_coarse --dataset head --tag head_example --epoch best
```

This evaluates the global matcher and writes coarse predictions for the test and training images to `final_runs/head_example/coarse_stage/pseudo_labels/`. These predictions determine the crop centers used by the local matcher.

### 3. Train the local matcher

```bash
python -m scripts_SSL.train_template --dataset head --tag head_example --coarse_stage 0
```

This stage learns correspondences within landmark-centered crops. Use the **same tag** as the global stage so the loader can find its coarse predictions. Checkpoints are saved under `final_runs/head_example/fine_stage/`.

## Evaluation and Pretrained Models

Evaluate the trained local matcher with:

```bash
python -m scripts_SSL.test_fine --dataset head --tag head_example --epoch best --skip_infer_train 1
```

The evaluator reports mean radial error (MRE; lower is better) and successful detection rate (SDR; higher is better). SDR is the percentage of predicted landmarks within each configured error threshold. Head, Hand, and Leg use millimeter-based errors; Chest uses the fixed-resolution pixel scale described above.

### Provided fine-stage checkpoints

| Dataset | Checkpoint | MRE |
| --- | --- | ---: |
| Head | `checkpoints/head.pth` | 1.772831 mm |
| Hand | `checkpoints/hand.pth` | 1.177376 mm |
| Chest | `checkpoints/chest.pth` | 5.458034 px at 512 × 512 |
| Leg | `checkpoints/leg.pth` | 2.792324 mm |

To evaluate a provided checkpoint, first complete the global training and coarse prediction steps for the same dataset and tag. Then, from `code/cc2d-final/`, copy the fine-stage checkpoint into the run directory and evaluate it:

```bash
mkdir -p ../../final_runs/head_example/fine_stage
cp ../../checkpoints/head.pth ../../final_runs/head_example/fine_stage/model_epoch_best.pth
python -m scripts_SSL.test_fine --dataset head --tag head_example --epoch best --skip_infer_train 1
```

Fine-stage inference also requires the annotated template and coarse predictions. Coarse models and predictions are not bundled; results depend on the coarse model used.

Checkpoints and settings were selected using Test-set MRE. See [model metadata](metadata/manifest.json) and [evaluation records](metadata/metrics.csv) for configuration and measurement details.

## Citation

If you use SGB-Match in your research, please cite:

```bibtex
@inproceedings{yao2026sgbmatch,
  title     = {Structure-Guided Self-Supervised Matching for One-Shot Medical Landmark Detection},
  author    = {Yao, Qingsong and Huang, Zhen and Wang, Ao and Wang, Rongsheng and Chen, Wenting and Wang, Jianji and Zhou, S. Kevin},
  booktitle = {Proceedings of the IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year      = {2026},
  note      = {Accepted; publication details to be updated when available}
}
```

For the original one-shot landmark work and BMPLE dataset, please also cite:

```bibtex
@inproceedings{yao2021one,
  title     = {One-Shot Medical Landmark Detection},
  author    = {Yao, Qingsong and Quan, Quan and Xiao, Li and Zhou, S. Kevin},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2021},
  year      = {2021},
  doi       = {10.1007/978-3-030-87196-3_17}
}
```

## Acknowledgements

We thank the authors of [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D) and [timm](https://github.com/huggingface/pytorch-image-models) for their open-source implementations, and the dataset contributors for making the landmark benchmarks available.

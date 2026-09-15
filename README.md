# BUSS-Net

BUSS-Net is a two-dimensional polyp segmentation model built on a fixed nnU-Net `PlainConvUNet`. It adds GeoFSS bridges, a Boundary-conditioned Local-Global Bottleneck (BLG), and Uncertainty-Guided Gather-Distribute (UG-GD) modules.

This repository is a minimal source release. It contains the model and training code, the formal C00 configuration, and dataset preparation utilities. Datasets, test scripts, experiment outputs, checkpoints, logs, manuscript files, and third-party source trees are intentionally excluded.

## Repository layout

```text
configs/comparisons/C00_buss-net.yaml  Formal BUSS-Net configuration
scripts/download_data.py               Download a user-specified archive
scripts/prepare_dataset.py              Resize and binarize image/mask pairs
scripts/create_eval_split.py            Generate the deterministic evaluation manifest
scripts/train.py                        Training entry point
src/buss_net/                           Model, losses, metrics, data pipeline, and trainer
```

## Environment

Python 3.10 or 3.11 and an NVIDIA CUDA environment are recommended. Install the PyTorch build matching the local CUDA driver first, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Choose the appropriate command at https://pytorch.org/get-started/locally/
pip install torch torchvision
pip install -r requirements.txt
```

`mamba-ssm` requires a compatible CUDA/PyTorch toolchain. BUSS-Net uses `nnunetv2` and `dynamic-network-architectures` as Python dependencies; their source trees and pretrained weights are not included.

## Public datasets and the adopted split

The project follows the widely used PraNet polyp-segmentation split. The original PraNet repository provides the ready-made public-data packages:

- [Training package: TrainDataset.zip](https://drive.google.com/file/d/1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb/view?usp=sharing)
- [Evaluation package: TestDataset.zip](https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view?usp=sharing)
- [PraNet dataset instructions and provenance](https://github.com/DengPingFan/PraNet#31-trainingtesting)
- [Official Kvasir-SEG page and download](https://datasets.simula.no/kvasir-seg/)

Please review each dataset's terms of use and cite its original publication. No dataset file is tracked by this repository.

| Use | Dataset | Images |
| --- | --- | ---: |
| Train | Kvasir-SEG | 900 |
| Train | CVC-ClinicDB | 550 |
| Evaluation | Kvasir-SEG | 100 |
| Evaluation | CVC-ClinicDB | 62 |
| Evaluation | CVC-ColonDB | 380 |
| Evaluation | CVC-300 | 60 |
| Evaluation | ETIS-LaribPolypDB | 196 |
| **Total** | **Train / evaluation** | **1450 / 798** |

The 1,450 training pairs are used for parameter learning. The remaining 798 pairs form five named evaluation subsets. The current training program evaluates this complete 798-image pool after every epoch and uses its mean Dice score to update `best.pt`. Therefore, this protocol has no separate validation set or locked final test set; results selected this way should be described as evaluation-pool model-selection results. For a confirmatory study, add an independent validation split and keep the final test set locked.

## Dataset layout

After downloading and extracting the archives, arrange the raw files as follows. Each image and mask must share the same filename stem.

```text
data/raw/
├── TrainDataset/
│   ├── image/
│   └── mask/
└── TestDataset/
    ├── CVC-300/{image,mask}/
    ├── CVC-ClinicDB/{image,mask}/
    ├── CVC-ColonDB/{image,mask}/
    ├── ETIS-LaribPolypDB/{image,mask}/
    └── Kvasir/{image,mask}/
```

Prepare the data at the formal input size and create a manifest that records every evaluation sample:

```bash
python scripts/prepare_dataset.py --raw-dir data/raw --out-dir data/processed --size 416
python scripts/create_eval_split.py --test-root data/processed/test \
  --out data/processed/splits/eval_split.json --split-seed 2026
```

Both `data/` and the generated manifest are ignored by Git.

## Training

```bash
python scripts/train.py --config configs/comparisons/C00_buss-net.yaml --device cuda
```

The formal configuration trains for 100 epochs at 416 x 416 resolution, uses random initialization, and writes checkpoints and logs below `experiments/comparisons/C00_buss-net/`. The complete `experiments/` directory is ignored by Git.


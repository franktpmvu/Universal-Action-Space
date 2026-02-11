# Universal Action Space

**Official PyTorch implementation of "A Universal Action Space for General Behavior Analysis"**

This repository provides a framework for efficient animal behavior analysis. By leveraging a **Universal Action Space (UAS)** constructed from large-scale human action datasets (Kinetics), this codebase allows you to recognize diverse animal behaviors with high accuracy using **frozen backbones** and **lightweight classifiers**.

![teaser](figures/teaser.png)

## Highlights

*   **🚀 Highly Efficient:** Train downstream tasks in **minutes to hours** (vs. days) using Linear Probing.
*   **❄️ Frozen Backbone:** No need to fine-tune heavy video transformers. We project animal behaviors into a pre-trained, universal human motion space.
*   **🏆 State-of-the-Art:** Outperforms traditional fine-tuning and LoRA on **MammalNet** and **ChimpBehave** benchmarks.
*   **🧩 Modular Design:** Built on top of [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer), making it easy to integrate with existing video analysis pipelines.

## Model Zoo & Results

We provide pre-trained configs and baselines for multiple animal behavior datasets.

**Note:** All results reported below are based on training with a single **NVIDIA RTX 3090 (24GB)** GPU.

### 🦁 MammalNet

| Dataset | Backbone | Pretrain | Top-1 Acc | Mean Class Acc | Training Time | # Params | Config | Chkpt |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MammalNet** | Swin-B | [K400](https://github.com/franktpmvu/Universal-Action-Space/releases/download/Pre-train/swin_base_patch244_window877_kinetics400_22k.pth) | 56.6% | 43.2% | ~8 hrs | 12.3K | [config](configs/recognition/swin/swin_base_patch244_window877_mammalnet_k400.py) | [github](https://github.com/franktpmvu/Universal-Action-Space/releases/download/Checkpoints/mammalnet_k400_UAS_epoch_30.pth) |

### 🦍 ChimpBehave

| Dataset | Backbone | Pretrain | Top-1 Acc | Mean Class Acc | Training Time | # Params | Config | Chkpt |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ChimpBehave** | Swin-B | [K400](https://github.com/franktpmvu/Universal-Action-Space/releases/download/Pre-train/swin_base_patch244_window877_kinetics400_22k.pth) | 93.7% | 65.8% | **~4 hrs** | 7.2K | [config](configs/recognition/swin/swin_base_patch244_window877_chimpBehave_k400.py) | [github](https://github.com/franktpmvu/Universal-Action-Space/releases/download/Checkpoints/chimpbehave_k400_UAS_epoch_30.pth) |
| **ChimpBehave** | Swin-B | [K600](https://github.com/franktpmvu/Universal-Action-Space/releases/download/Pre-train/swin_base_patch244_window877_kinetics600_22k.pth) | 93.5% | 72.3% | **~4 hrs** | 7.2K | [config](configs/recognition/swin/swin_base_patch244_window877_chimpBehave_k600.py) | [github](https://github.com/franktpmvu/Universal-Action-Space/releases/download/Checkpoints/chimpbehave_k600_UAS_epoch_30.pth) |
| **ChimpBehave** | Swin-B | [K700](https://github.com/franktpmvu/Universal-Action-Space/releases/download/Pre-train/swin_base_patch244_window877_kinetics700_22k.pth) | 94.2% | 56.4% | **~4 hrs** | 7.2K | [config](configs/recognition/swin/swin_base_patch244_window877_chimpBehave_k700.py) | [github](https://github.com/franktpmvu/Universal-Action-Space/releases/download/Checkpoints/chimpbehave_k700_UAS_epoch_30.pth) |

### ⚡ Efficiency Comparison (Kinetics-700 diff)

Our method achieves competitive accuracy with significantly lower resource usage compared to full fine-tuning.

| Method | Top-1 Acc | Training Time | # Trainable Params | Config |
| :--- | :---: | :---: | :---: | :---: |
| **UAS (Ours)** | 87.9% | **54 hrs** | **0.1 M** | [config](configs/recognition/swin/swin_base_patch244_window877_kinetics700_diff_k600.py) |
| LoRA | 88.6% | 86 hrs | 1.6 M | [config](configs/recognition/swin/swin_base_patch244_window877_kinetics700_diff_k600_lora.py) |
| Full Fine-tuning | 88.8% | 105 hrs | 87.7 M | - |

## Usage

### 1. Installation
Please refer to [docs/install.md](docs/install.md) for environment setup.

### 2. Data Preparation
See [docs/data_preparation.md](docs/data_preparation.md) for full instructions on downloading and formatting the datasets.

We provide the specific split lists used in our paper for reproducibility:
*   **MammalNet:** We use the official train/val split configuration provided by the dataset authors.
*   **ChimpBehave:** [train list](https://github.com/franktpmvu/Universal-Action-Space/releases/download/Pre-release/chimpbehave_train.txt) | [val list](https://github.com/franktpmvu/Universal-Action-Space/releases/download/Pre-release/chimpbehave_val.txt)
*   **Kinetics-700 diff:** [train list](https://github.com/franktpmvu/Universal-Action-Space/releases/download/Pre-release/k700_diff_train.txt) | [val list](https://github.com/franktpmvu/Universal-Action-Space/releases/download/Pre-release/k700_diff_val.txt)

### 3. Training

#### Standard Training (Linear Probing)
To train a classifier on the **ChimpBehave** dataset using a frozen Kinetic-400 backbone:

```bash
python tools/train.py configs/recognition/swin/swin_base_patch244_window877_chimpBehave_k400.py \
    --work-dir ./work_dirs/chimpBehave_k400 \
    --cfg-options freeze_backbone=True load_from=checkpoints/swin_base_patch244_window877_kinetics400_22k.pth
```

**Common Arguments:**
*   `--work-dir`: Specifies the directory where logs and checkpoints will be saved.
*   `--cfg-options`: Override specific config parameters from the command line:
    *   `freeze_backbone=True`: Freezes the transformer backbone (required for UAS mode).
    *   `load_from=/path/to/checkpoint.pth`: Loads a pre-trained backbone weight.

#### LoRA Training (Optional)
You can also enable Low-Rank Adaptation (LoRA) on top of the backbone without needing a separate config file. Just append the relevant options:

```bash
python tools/train.py configs/recognition/swin/swin_base_patch244_window877_chimpBehave_k400.py \
    --work-dir ./work_dirs/chimpBehave_k400_lora \
    --cfg-options freeze_backbone=True \
    model.backbone.use_lora=True \
    model.backbone.lora_rank=32 \
    model.backbone.lora_alpha=32
```

**LoRA Configuration:**
*   `model.backbone.use_lora`: Set to `True` to insert LoRA layers.
*   `model.backbone.lora_rank`: Dimension of the low-rank adaptation matrices.
*   `model.backbone.lora_alpha`: Scaling factor for LoRA weights.

*Note: Dedicated `_lora` config files are not strictly necessary; you can inject these settings into any standard config via `--cfg-options`.*

### 4. Validation
Evaluate your trained model:

```bash
python tools/test.py configs/recognition/swin/swin_base_patch244_window877_chimpBehave_k400.py \
    work_dirs/chimpBehave_k400/epoch_30.pth \
    --eval top_k_accuracy mean_class_accuracy
```

<!--
## Citation

If you use this code or models in your research, please cite:

```bibtex
@article{UAS2025,
  author={Chang, Hung-Shuo and Yang, Yue-Cheng and Chen, Yu-Hsi and Wang, Chien-Yao and Liao, Hong-Yuan Mark},
  title={A Universal Action Space for General Behavior Analysis},
  journal={2025 IEEE Conference on Artificial Intelligence (CAI)},
  year={2025}
}
```
--!>

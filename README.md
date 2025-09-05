# [🎥 ICCV2025] ESSENTIAL: Episodic and Semantic Memory Integration for Video Class-Incremental Learning

[![Conference](https://img.shields.io/badge/ICCV-2025-blue)](https://iccv2025.thecvf.com/)  
[![arXiv](https://img.shields.io/badge/arXiv-2508.10896-b31b1b.svg)](https://arxiv.org/abs/2508.10896)  
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://jong980812.github.io/ESSENTIAL/)  
[![Lab](https://img.shields.io/badge/Vision%20and%20Learning-Lab-orange)](https://vll.khuh.ac.kr/)  

---

<p align="center">
  <img src="./docs/static/image/overview.png" alt="ESSENTIAL Overview" width="750"/>
</p>

---
## 👨‍💻 Authors

| [Jongseo Lee<sup>1*</sup>](https://jong980812.github.io/) | [Kyungho Bae<sup>2*</sup>](https://github.com/Backdrop9019) | [Kyle Min<sup>3</sup>](https://sites.google.com/view/kylemin) | [Gyeong-Moon Park<sup>4†</sup>](https://gyeongmoon.github.io/) | [Jinwoo Choi<sup>1†</sup>](https://sites.google.com/site/jchoivision/) |
|---|---|---|---|---|

<sup>*</sup> Equal contribution, <sup>†</sup> Corresponding author
## 📜 License
This work is licensed under a [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## 📌 Highlight
- **Accepted at ICCV 2025 (Highlight Presentation)**
- Proposes **ESSENTIAL**, a framework inspired by human memory that integrates **episodic and semantic memory** for **video class-incremental learning (VCIL)**.
- Achieves a **favorable trade-off** between memory efficiency and recognition performance compared to prior VCIL methods.
- Code release includes **training, evaluation, and visualization tools**.

## 📑 Contents
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Training](#-training)
- [Evaluation](#-evaluation)

## 🚀 Installation

We recommend using **conda** to create a clean environment.  
The code has been tested with **Python 3.8**, **PyTorch 2.0.1**, and **CUDA 11.7**.

### Step 1. Create conda environment
```bash
conda create -n ESSENTIAL python=3.8 -y
conda activate ESSENTIAL
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```
## 📂 Dataset

We provide the annotation files for ESSENTIAL on [Hugging Face Hub](https://huggingface.co/datasets/KHUjongseo/ESSENTIAL/tree/main).

### Step 1. Download annotations
Please download the annotation files from the link above.

### Step 2. Organize dataset
Place the downloaded files under the `./data/` directory as follows:
```
ESSENTIAL/
│── data/
│   ├── TCD/
│   │   ├── ...
│   │   └── 
│   ├── vCLIMB/
│   │   ├── ...
│   │   └── 
│   └── …
```
### Step 3. Prepare raw videos
- The benchmark datasets (e.g., Kinetics-400) should be downloaded separately.  
## 🎯 Training
...

## 📊 Evaluation
...
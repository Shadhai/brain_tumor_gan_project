# A Controlled Augmentation Framework Reveals the Critical Role of GAN Quality
### A Case Study on Minority Brain Tumor Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)

## 📄 Abstract
Class imbalance in medical imaging datasets causes deep learning classifiers to miss rare diseases.  
Generative adversarial networks (GANs) can synthesise additional minority‑class samples, but the optimal ratio of synthetic‑to‑real data – and its impact on diagnostic reliability – has never been systematically studied.

We designed a controlled five‑set experiment (D1‑D5) on brain MRI data, artificially limiting the pituitary class to 200 images.  
A class‑specific DCGAN generated additional training images at four ratios (0, 200, 600, 1000), while a traditional augmentation baseline matched the count of the best GAN set.  
All classifiers (EfficientNet‑B0) were evaluated over five seeds with paired t‑tests.

**Key finding:** Pituitary recall rose monotonically from **0.880 → 0.984** as synthetic data increased, and traditional augmentation reached **0.994**.  
However, **FID > 230**, a real‑vs‑fake classifier scored **99.8 % accuracy**, and t‑SNE embeddings showed completely separate clusters – proving the GAN never learned the true pituitary distribution.  
This negative result becomes the main contribution: *augmentation‑ratio conclusions are meaningless without rigorous GAN quality verification.*

We provide the full code, trained models (upon acceptance), and a multi‑modal quality‑assurance protocol that future researchers can reuse.

👉 **Read the full technical report:** [`Technical_Report_GAN_Augmentation_2026.pdf`](Technical_Report_GAN_Augmentation_2026.pdf)

---

## 🧠 Architecture Overview
The figure below summarises the D1‑D5 augmentation protocol and the multi‑modal evaluation pipeline.

![Architecture Diagram](architecture_diagram.png) *(you can replace this with a real diagram)*

1. **Class‑specific DCGAN** trained on all 1,406 pituitary images.  
2. **Controlled minority scenario:** 200 real pituitary images kept.  
3. **Five training sets** constructed:  
   - D1 – 200 real (baseline)  
   - D2 – 200 real + 200 GAN  
   - D3 – 200 real + 600 GAN  
   - D4 – 200 real + 1000 GAN  
   - D5 – 200 real + 600 traditional augmentations (rotation, flip, brightness)  
4. **Evaluation:** FID, real‑vs‑fake classifier, t‑SNE, Grad‑CAM, confusion matrices.

---

## 📊 Key Results

| Dataset | Val. Accuracy | Val. Pit. Recall | Test Accuracy | Test Pit. Recall |
|---------|--------------|------------------|---------------|------------------|
| D1 (Baseline) | 0.9714 ± 0.0091 | 0.8800 ± 0.0400 | 0.9194 | 0.94 |
| D3 (600 GAN)  | 0.9716 ± 0.0068 | 0.9650 ± 0.0116 | 0.9137 | 0.90 |
| D4 (1000 GAN) | 0.9802 ± 0.0036 | 0.9842 ± 0.0067 | 0.9250 | 0.92 |
| D5 (Trad. aug)| 0.9777 ± 0.0020 | 0.9943 ± 0.0036 | 0.9237 | 0.94 |

**Quality metrics:**
- FID: D2–D4 all > 230, D5 = 113
- Real‑vs‑fake CNN test accuracy: **99.8 %**
- t‑SNE: Real and GAN images form completely disjoint clusters

Plots available in `outputs/plots/`.

---

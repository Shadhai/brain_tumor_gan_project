# 🔬 A Controlled Augmentation Framework Reveals the Critical Role of GAN Quality

## A Reproducibility-Focused Study on Minority Brain Tumor Detection in MRI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)
![Research](https://img.shields.io/badge/Research-Medical%20AI-8A2BE2?style=for-the-badge)
![Reproducibility](https://img.shields.io/badge/Reproducibility-Full-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

### 🧠 Brain MRI • 🎯 Class Imbalance • ⚡ GAN Augmentation • 📊 Explainable AI • 🔬 Reproducible Research

</div>

---

# 🌟 Overview

Deep learning systems in medical imaging frequently fail on **rare disease classes** because training datasets are severely imbalanced. While **Generative Adversarial Networks (GANs)** are widely proposed as a solution, most studies evaluate only *classification improvement* and ignore whether the generated medical images are actually realistic.

This repository presents a **controlled augmentation framework** that investigates:

* How increasing GAN-generated data affects minority-class detection
* Whether performance gains truly correspond to medically realistic synthesis
* Why augmentation studies require rigorous quality verification

---

# 📄 Abstract

We designed a controlled five-dataset experiment (**D1–D5**) using a public brain MRI dataset.
The pituitary tumor class was intentionally reduced to only **200 real images** to simulate a clinically rare minority condition.

A class-specific **DCGAN** generated synthetic MRI slices at increasing ratios:

| Dataset | Composition                              |
| ------- | ---------------------------------------- |
| D1      | 200 real pituitary images                |
| D2      | 200 real + 200 GAN                       |
| D3      | 200 real + 600 GAN                       |
| D4      | 200 real + 1000 GAN                      |
| D5      | 200 real + 600 traditional augmentations |

An **EfficientNet-B0** classifier was trained on all datasets over **five random seeds**.

### Key Findings

* Pituitary recall improved monotonically:

```text
0.880 → 0.984 (GAN)
0.994 (Traditional Augmentation)
```

However:

* FID remained extremely poor (**231–235**)
* Real-vs-Fake CNN achieved **99.8% accuracy**
* t-SNE embeddings formed completely separate clusters

These results prove the GAN failed to learn the true medical distribution despite apparent classification gains.

---

# 🚨 Main Scientific Contribution

## **Performance improvement alone is NOT evidence of successful medical image generation.**

This work demonstrates that:

> augmentation-ratio conclusions become scientifically unreliable when GAN quality is not rigorously verified.

We therefore propose a **multi-modal GAN verification protocol** for future medical-AI research.

---

# 🧠 Experimental Pipeline

<div align="center">

![Architecture](architecture_diagram.png)

</div>

---

# 🔁 Controlled D1–D5 Augmentation Sweep

## Step 1 — Train GAN

A class-specific DCGAN is trained exclusively on pituitary MRI slices.

## Step 2 — Create Minority Scenario

Only **200 real pituitary images** are retained.

## Step 3 — Controlled Augmentation

Synthetic images are added incrementally:

| Set | Synthetic Count                   |
| --- | --------------------------------- |
| D1  | 0                                 |
| D2  | 200                               |
| D3  | 600                               |
| D4  | 1000                              |
| D5  | Traditional augmentation baseline |

## Step 4 — Multi-Modal Evaluation

The framework evaluates:

* Classification performance
* FID score
* Real-vs-fake separability
* t-SNE feature overlap
* Grad-CAM attention localization
* Statistical significance across seeds

---

# 📊 Results

# 🏆 Classification Performance

| Dataset | Val Accuracy        | Val Pituitary Recall | Test Accuracy | Test Pituitary Recall |
| ------- | ------------------- | -------------------- | ------------- | --------------------- |
| D1      | 0.9714 ± 0.0091     | 0.8800 ± 0.0400      | 0.9194        | 0.94                  |
| D2      | 0.9711 ± 0.0068     | 0.9375 ± 0.0137      | 0.9156        | 0.89                  |
| D3      | 0.9716 ± 0.0068     | 0.9650 ± 0.0116      | 0.9137        | 0.90                  |
| D4      | **0.9802 ± 0.0036** | **0.9842 ± 0.0067**  | **0.9250**    | 0.92                  |
| D5      | 0.9777 ± 0.0020     | **0.9943 ± 0.0036**  | 0.9237        | **0.94**              |

---

# 🔍 GAN Quality Verification

| Metric            | Result            | Interpretation                       |
| ----------------- | ----------------- | ------------------------------------ |
| FID (GAN sets)    | 231–235           | Extremely poor realism               |
| FID (Traditional) | 113               | Better but still imperfect           |
| Real-vs-Fake CNN  | 99.8% accuracy    | Synthetic images trivially separable |
| t-SNE             | Disjoint clusters | No manifold overlap                  |

---

# 📈 Visual Outputs

All generated visualizations are automatically saved in:

```text
outputs/plots/
```

Including:

* GAN training progression
* FID vs augmentation ratio
* Accuracy vs recall trade-offs
* Confusion matrices
* Grad-CAM visualizations
* t-SNE embeddings
* Real vs synthetic feature distributions

---

# 🧪 Multi-Modal Quality Assurance Protocol

This repository introduces a reusable evaluation framework for safe medical-AI augmentation studies.

| Validation Stage      | Goal                            | Script                                 |
| --------------------- | ------------------------------- | -------------------------------------- |
| FID Score             | Distribution similarity         | `src/evaluation/fid_score.py`          |
| Real-vs-Fake CNN      | Feature overlap check           | `src/evaluation/real_fake_analysis.py` |
| t-SNE Analysis        | Latent manifold inspection      | `src/evaluation/real_fake_analysis.py` |
| Grad-CAM              | Anatomical attention validation | `src/explainability/gradcam.py`        |
| Multi-Seed Statistics | Reproducibility                 | `src/model/classifier_train.py`        |

---

# ⚙️ Reproducibility

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Shadhai/brain_tumor_gan_project.git
cd brain_tumor_gan_project
```

---

## 2️⃣ Create Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Download Dataset

Download:

[Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?utm_source=chatgpt.com)

Place into:

```text
dataset/raw/
```

Required structure:

```text
dataset/raw/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/
```

---

## 5️⃣ Run Complete Pipeline

```bash
python main.py
```

This single command performs:

1. GAN training
2. Synthetic image generation
3. D1–D5 dataset creation
4. EfficientNet-B0 training
5. Multi-seed evaluation
6. FID computation
7. Real-vs-fake analysis
8. t-SNE visualization
9. Grad-CAM explainability

---

# 🗂 Repository Structure

```text
brain_tumor_gan_project/
│
├── src/
│   ├── gan/
│   ├── model/
│   ├── evaluation/
│   └── explainability/
│
├── outputs/
│   └── plots/
│
├── experiments/
│
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

---

# 📚 Technical Report

The complete technical report is available here:

```text
Technical_Report_GAN_Augmentation.pdf
```

It contains:

* Methodology
* Statistical analysis
* GAN architecture
* Experimental protocol
* Failure analysis
* Discussion
* Future work

---

# 🧠 Research Impact

This work contributes to:

* Medical AI safety
* Reproducible augmentation research
* Explainable deep learning
* Synthetic data reliability
* Trustworthy healthcare AI

It also aligns with:

* **Society 5.0**
* Open Science
* Human-centered AI
* Reproducible research standards

---

# 🏅 Suggested Citation

```bibtex
@unpublished{shadhai2026controlled,
  author = {PANDU SHADHAI JOSEPH},
  title  = {A Controlled Augmentation Framework Reveals the Critical Role of GAN Quality:
            A Case Study on Minority Brain Tumor Detection},
  year   = {2026},
  note   = {Technical Report, under review}
}
```

---

# 📜 License

This project is released under the MIT License.

Dataset license belongs to the original Kaggle authors.

---

# 🤝 Acknowledgements

* Kaggle Brain MRI Dataset contributors
* PyTorch community
* Open-source medical AI ecosystem

All experiments were conducted on:

```text
NVIDIA RTX 3050 GPU
```

---

# 📬 Contact

## Alexander Joseph Shadhai

* GitHub: [Shadhai GitHub Profile](https://github.com/Shadhai)
* Repository: [brain_tumor_gan_project Repository](https://github.com/Shadhai/brain_tumor_gan_project.git)

---

<div align="center">

## ⭐ If this project helped your research, consider starring the repository.

### “Reproducibility is not optional in medical AI.”

</div>

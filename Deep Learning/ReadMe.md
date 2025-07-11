# Deep Learning: Autoencoders and Latent Space Optimization

This project explores the use of **autoencoders** and **latent space representations** for downstream image classification tasks using the **MNIST** and **CIFAR-10** datasets. It was developed as part of the "Deep Learning on Computational Accelerators" course at the Technion.

## 📌 Project Overview

The project investigates the structure and utility of learned latent representations through three main experiments:

1. **Self-Supervised Autoencoding**
   - Train autoencoders to compress input images into a 128-dimensional latent space.
   - Evaluate performance by training classifiers on the encoded representations.
   - Datasets: MNIST and CIFAR-10.

2. **Classification-Guided Encoding**
   - Jointly train encoder and classifier to optimize latent representations for classification, rather than reconstruction.
   - Skip the decoder entirely.

3. **Structured Latent Spaces**
   - Introduce structure to the latent space (e.g. via **contrastive learning** using SimCLR-style loss).
   - Evaluate latent space separability and classification accuracy.

## 🧠 Goals

- Learn compact and informative latent representations.
- Compare self-supervised vs supervised latent encoding strategies.
- Explore methods (e.g., contrastive learning) to enforce structure in the latent space.
- Visualize and analyze learned latent spaces using t-SNE.

## 🛠️ Technologies

- **Python**
- **PyTorch**
- **NumPy, Matplotlib**
- **t-SNE for latent space visualization**
- Datasets: `MNIST`, `CIFAR-10`

## 📈 Evaluation Metrics

- **Reconstruction Loss (MAE)** for autoencoders.
- **Classification Accuracy** on train, validation, and test sets.
- **t-SNE Visualizations** for comparing latent space quality.
- **Linear interpolation** in latent space to evaluate smoothness and semantic meaning.
- **Qualitative reconstruction results** (before and after improvements).


## 🖼️ Visuals

- Side-by-side comparisons of original and reconstructed images
- Latent space t-SNE plots
- Interpolation results in latent space

## 🔬 Key Concepts

- Autoencoders
- Latent space representation
- Contrastive learning (SimCLR)
- Classification over encoded features
- Dimensionality reduction
- Self-supervised vs supervised learning

## 👥 Authors

This project was developed as part of coursework in the Technion’s Deep Learning on Accelerators course.

## 📝 References

- [SimCLR Paper](https://arxiv.org/abs/2002.05709)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Contrastive Representation Learning Blog](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html)

---

> Feel free to clone, modify, and explore this repository to learn more about structured representations and deep learning on vision tasks.

**"Combating the Vanishing Gradient Problem in Deep CNNs: A Study of VGG38 with Batch Normalization and Residual Connections"**

## Overview

Training deep convolutional neural networks often leads to vanishing gradients, especially in very deep architectures. This project:
- Investigates the **Vanishing Gradient Problem (VGP)** in a 38-layer deep VGG model on the CIFAR-100 dataset.
- Compares the training dynamics of **VGG08** (shallow, stable) vs. **VGG38** (deep, unstable).
- Implements two common solutions to VGP: **Batch Normalization (BN)** and **Residual Connections (RC)**.
- Evaluates the performance impact of BN, RC, and their combination in solving the VGP in the deeper architecture.

## 🧠 Key Findings

- VGG38 without BN or RC fails to train effectively (accuracy ≈ 0.01) due to VGP.
- BN improves training stability and allows higher learning rates.
- RC allows better gradient flow and outperforms BN individually.
- Combining **BN + RC** provides the best performance, achieving ~59% accuracy on the test set.

## 📊 Results Summary

| Model             | Learning Rate | Val Loss | Val Accuracy |
|------------------|---------------|----------|--------------|
| VGG08            | 1e-3          | 1.95     | 46.84%       |
| VGG38            | 1e-3          | 4.61     | 00.01%       |
| VGG38 + BN       | 1e-3          | 2.01     | 45.08%       |
| VGG38 + RC       | 1e-3          | 1.84     | 52.32%       |
| VGG38 + BN + RC  | 1e-2          | 1.58     | **59.16%**    |


## 🧪 How to Run

# Train VGG38 with BN + RC

# Visualize gradient flow


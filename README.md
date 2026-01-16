# 🇸🇦 Arabic Sign Language Recognition System

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Accuracy](https://img.shields.io/badge/Accuracy-98.15%25-brightgreen.svg)](https://github.com/yourusername/arabic-sign-language)
[![Stars](https://img.shields.io/github/stars/yourusername/arabic-sign-language?style=social)](https://github.com/yourusername/arabic-sign-language)

**State-of-the-Art 98.15% Accuracy in Real-Time Arabic Sign Language Recognition**

[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Demo](#-live-demo) •
[Papers](#-research-papers)

</div>

---

## 📖 Overview

This project implements a high-accuracy **Arabic Sign Language Recognition System** using deep learning. The system achieves **98.15% accuracy** in recognizing 28 Arabic letters in real-time through webcam. It includes a complete pipeline from data preprocessing to deployment-ready inference.

### 🎯 **Key Achievements**

| Metric | Score | Status |
|--------|-------|--------|
| **Overall Accuracy** | **98.15%** | 🏆 **State-of-the-Art** |
| **Precision** | 98.32% | ⭐ **Excellent** |
| **Recall** | 98.15% | ⭐ **Excellent** |
| **F1-Score** | 98.11% | ⭐ **Excellent** |
| **Inference Speed** | 35 FPS | 🚀 **Real-time** |
| **Perfect Classes** | 16/28 | 💯 **Flawless** |

---

## ✨ Features

### 🔍 **Data Processing Pipeline**
- **Smart Blur Detection**: Automatic identification of low-quality images
- **Advanced Augmentation**: 7+ augmentation techniques preserving sign meaning
- **Dataset Analysis**: Comprehensive visualization and statistics
- **Class Balancing**: Optional undersampling/oversampling strategies

### 🧠 **Custom CNN Architecture**
- **4 Convolutional Blocks** with increasing filters (32→64→128→256)
- **Batch Normalization** after each convolution
- **Global Average Pooling** for better generalization
- **Strategic Dropout** to prevent overfitting
- **Optimized** for Arabic sign characteristics

### ⚡ **Real-time Inference**
- **Live Webcam Recognition**: 30+ FPS performance
- **Confidence Display**: Real-time prediction confidence
- **Multiple Controls**: Save snapshots, pause/resume, quit
- **Performance Stats**: FPS counter and accuracy metrics

### 📊 **Comprehensive Evaluation**
- **Confusion Matrices** (raw & normalized)
- **ROC Curves** with AUC scores
- **Per-class Metrics**: Precision, Recall, F1-score
- **Training Visualizations**: Loss and accuracy curves
- **Sample Predictions**: Visual verification of results

---

## 📁 Project Structure

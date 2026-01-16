# skin-cancer-detection-cnn-Python-
A Convolutional Neural Network (CNN) to detect skin cancer, built and trained from scratch using Keras/TensorFlow.

# 🔬 Skin Cancer Detection Using Neural Network

A neural network built **from scratch using NumPy** for detecting skin cancer from images.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Only-green.svg)

## 👥 Team Members
- **Muhammad Shan** - SU92-MSAIW-F25-033
- **Aiman Batool** - SU92-MSAIW-F25-018

## 📋 Project Overview

This project implements a fully-connected neural network from scratch to classify skin lesion images as **cancerous** or **non-cancerous**.

### Key Features
- ✅ Neural Network built with pure NumPy (no TensorFlow/PyTorch)
- ✅ Image preprocessing pipeline
- ✅ Data augmentation
- ✅ Class imbalance handling
- ✅ Comprehensive evaluation metrics

## 🗂️ Project Structure



**Project Structure**
Skin-Cancer-Detection-NN/
│
├── data/
│   ├── training/
│   │   ├── cancer/
│   │   └── non_cancer/
│   └── testing/
│       ├── cancer/
│       └── non_cancer/
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Image loading & processing
│   ├── neural_network.py       # NN from scratch
│   ├── train.py                # Training loop
│   ├── inference.py            # Predictions
│   └── evaluate.py             # Metrics
│
├── models/
│   └── (saved model)
│
├── docs/
│   ├── Project_Initial_Document.md
│   └── Project_Report.md
│
├── config.py
├── main.py
├── requirements.txt
└── README.md



Skin-Cancer-Detection-NN/
├── data/
│ ├── training/
│ │ ├── cancer/
│ │ └── non_cancer/
│ └── testing/
│ ├── cancer/
│ └── non_cancer/
├── src/
│ ├── data_preprocessing.py
│ ├── neural_network.py
│ ├── train.py
│ ├── evaluate.py
│ └── inference.py
├── models/
├── config.py
├── main.py
└── README.md

text


## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install numpy Pillow matplotlib
2. Prepare Dataset
Place your images in:

data/training/cancer/ - Cancer images
data/training/non_cancer/ - Non-cancer images
data/testing/cancer/ - Test cancer images
data/testing/non_cancer/ - Test non-cancer images
3. Run
Bash

python main.py
🧠 Model Architecture
text

Input (64×64×3 = 12,288)
        ↓
Hidden Layer 1 (128 neurons, ReLU)
        ↓
Hidden Layer 2 (64 neurons, ReLU)
        ↓
Hidden Layer 3 (32 neurons, ReLU)
        ↓
Output (1 neuron, Sigmoid)


📄 License
Educational purposes only.

text


---

## ✅ Setup Instructions

### Step 1: Create Project Folder
```bash
mkdir Skin-Cancer-Detection-NN
cd Skin-Cancer-Detection-NN
mkdir data src models docs
mkdir data/training data/testing
mkdir data/training/cancer data/training/non_cancer
mkdir data/testing/cancer data/testing/non_cancer
Step 2: Copy Your Images
Copy your images from Desktop to:

Training cancer → data/training/cancer/
Training non-cancer → data/training/non_cancer/
Testing cancer → data/testing/cancer/
Testing non-cancer → data/testing/non_cancer/
Step 3: Update Config Path (if needed)
In config.py, update the path if you want to use images directly from Desktop:

Python

DATA_DIR = r'C:\Users\YourUsername\Desktop\skin_cancer_dataset'
Step 4: Install & Run
Bash

pip install numpy Pillow matplotlib
python main.py




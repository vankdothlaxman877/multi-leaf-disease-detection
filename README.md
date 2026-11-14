# Multi-Leaf Disease Detection Using Federated Learning

A deep learning project for detecting plant leaf diseases across **Mango**, **Apple**, and **Tomato** crops. The model uses ensemble deep feature extractors (**VGG19**, **DenseNet121**, **ResNet50**) combined with Contrast-Limited Adaptive Histogram Equalization (**CLAHE**) and **Federated Learning** to classify diseased vs. healthy leaves while maintaining client data privacy.

---

## 🌿 Supported Crops & Diseases

- **Mango**: Healthy, Diseased
- **Apple**: Healthy, Rust, Scab
- **Tomato**: Bacterial Spot, Early Blight, Late Blight, Healthy, etc.

---

## 🛠️ Project Architecture

1. **Preprocessing**:
   - Contrast-Limited Adaptive Histogram Equalization (CLAHE) on the L-channel in LAB color space to enhance lesion contrast.
2. **Feature Extraction**:
   - Pre-trained CNN backbones on ImageNet:
     - **VGG19** (512-dim)
     - **DenseNet121** (1024-dim)
     - **ResNet50** (2048-dim)
   - Combined feature vector dimension: **3,584**.
3. **Federated Learning & Classification**:
   - Federated Averaging (FedAvg) across decentralized client nodes (Mango client, Apple client, Tomato client).
   - Local training & global aggregation for multi-crop generalizability.

---

## 🚀 Getting Started in VS Code

### 1. Prerequisites
- **Python 3.10 – 3.12** (Python 3.12 recommended)
- **VS Code** with the **Python** and **Jupyter** extensions installed.

### 2. Setup Virtual Environment

Open PowerShell in the project directory:

`powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
`

### 3. Running the Notebook
1. Open Main.ipynb in VS Code.
2. Select .venv as your Jupyter Kernel (top right).
3. Run the cells sequentially.

---

## 📁 Repository Structure

`	ext
├── datasets/             # Local extracted datasets (ignored in Git)
├── Main.ipynb            # Main Jupyter Notebook pipeline
├── requirements.txt      # Project dependencies
├── .gitignore            # Excludes datasets and virtual environments
└── README.md             # Project documentation
`

---

## 📜 License
MIT License

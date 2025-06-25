# 🧼 CleanVision using DIV2K  
**Deep Learning-Based Real-Time Denoising, Inpainting & Restoration of Noisy Images (Computer Vision)**

---

## 🧠 Project Overview

**CleanVision** is a deep learning-powered computer vision project designed to enhance, denoise, and restore degraded images in real time.  
Leveraging the **DIV2K** dataset, it addresses three core image restoration challenges:

- ✨ Denoising  
- 🧩 Inpainting  
- 📈 Super-Resolution / Restoration

---

## 🌟 Key Features

- 🧹 Real-time **image denoising** using CNNs or GAN-based models  
- 🖼️ Intelligent **inpainting** of missing or corrupted image regions  
- 🎨 Image **restoration and super-resolution** to enhance visual quality  
- 🧱 Modular and extensible architecture for training & inference  
- 📊 Evaluation with **PSNR**, **SSIM**, and visual comparisons  
- 🧪 Custom input support for **real-world test cases**

---

## 📂 Dataset: DIV2K

The project uses the **DIVerse 2K Resolution (DIV2K)** dataset, a high-quality dataset for image restoration tasks.  
It contains 2K resolution images with various degradations, making it ideal for:

- Denoising  
- Super-resolution  
- Compression artifact removal  
- Inpainting (with custom masks)  

🔗 [DIV2K Dataset – CVL ETH Zurich](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

---

## 🏗️ Model Architectures

| Task          | Model Options                               |
|---------------|----------------------------------------------|
| Denoising     | DnCNN, UNet, Residual Denoisers              |
| Inpainting    | Partial Convolutions, Masked Autoencoders    |
| Restoration   | SRCNN, ESRGAN                                |

- **Loss Functions**: MSE, Perceptual Loss, SSIM Loss  
- **Optimizers**: Adam, AdamW + learning rate schedulers  
- Supports: **PyTorch**, **TensorFlow**, *(optionally with PyTorch Lightning or Keras)*

---

## 🔬 Workflow

### 1️⃣ Data Preprocessing
- Load & augment DIV2K images
- Apply synthetic noise, blur, or masking

### 2️⃣ Model Training
- Train models for denoising, inpainting, and restoration
- Use early stopping, checkpointing, logging via TensorBoard / W&B

### 3️⃣ Inference & Visualization
- Pass noisy or damaged images through trained models
- Visualize side-by-side:
  - Original  
  - Corrupted  
  - Restored

---

## 🖼️ Example Use Case

| Input (Noisy) | Restored Output |
|---------------|------------------|
| `noisy.png`   | `clean.png`      |

---

## ⚙️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/your-username/CleanVision-DIV2K
cd CleanVision-DIV2K

# Install required packages
pip install -r requirements.txt

# Download & place DIV2K dataset
# Put it in the `data/` directory or update the config path
```

---

## 🚀 Train & Run Inference

```bash
# To train a model (example: denoising with DnCNN)
python train.py --task denoising --model dncnn

# To run inference on a noisy image
python infer.py --input sample_images/noisy.png --task denoising
```

---

## 📊 Evaluation Metrics

- 🧮 **PSNR** - Peak Signal-to-Noise Ratio  
- 🧮 **SSIM** - Structural Similarity Index  
- 👁️ **LPIPS** - Learned Perceptual Image Patch Similarity *(optional)*

---

## 📁 Folder Structure

```
CleanVision-DIV2K/
├── data/           # DIV2K and sample noisy images
├── models/         # Model architectures
├── outputs/        # Saved results and checkpoints
├── scripts/        # Utilities and evaluation tools
├── train.py        # Training script
├── infer.py        # Inference script
├── config.yaml     # Model and data configuration
└── README.md       # Project documentation
```

---

## 🧭 Roadmap / Future Work

- 🔴 Real-time restoration from webcam/image feed  
- 🟠 Vision Transformer support (ViT, MAE, etc.)  
- 🟡 Train on additional datasets (BSD500, ImageNet subsets)  
- 🟢 Evaluate robustness to adversarial noise

---

## 🤝 Contributions

Contributions are welcome!  
Feel free to fork, open issues, or submit pull requests for improvements.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgements

- [DIV2K Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) by **CVL ETH Zurich**  
- Open-source tools: **PyTorch**, **TensorFlow**, **OpenCV**  
- Research inspiration from: **ESRGAN**, **DnCNN**, and **NVIDIA Inpainting Models**

---

> 💡 *Developed with ❤️ by Abaha Mondal*

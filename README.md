# 🔬 STEM Image Denoising using Deep Learning

This repository provides a modular and reproducible framework for denoising high-resolution Scanning Transmission Electron Microscopy (STEM) images using a variety of UNet-based architectures.

---

## 📁 Project Structure

```
stem-denoising/
├── data/                   # Raw and processed datasets
│   ├── raw/                # Original noisy/clean image pairs
│   ├── processed/          # Patched and augmented data
│   └── dataset.py          # Preprocessing, patching, augmentation
├── models/
│   ├── unet.py             # UNet from scratch
│   ├── pretrained.py       # UNet with pretrained encoders (VGG16)
│   └── utils.py            # Utilities for saving images
├── training/
│   ├── train.py            # Training pipeline
│   ├── loss_functions.py   # Composite and perceptual loss functions
│   └── metrics.py          # PSNR, SSIM utilities
├── evaluation/
│   ├── eval.py             # Evaluation script
│   └── visualize.py        # Visualization, training plots, analysis
├── config.py               # Global configs (patch size, batch size)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🧠 Key Features

- 🔧 **UNet Variants**: Standard UNet, residual UNet, and pretrained UNet (VGG16)
- 🧪 **Loss Functions**: Custom hybrid loss including MSE, FFT, Poisson, SSIM, TV, and intensity loss
- 🔁 **Augmentation**: Geometric + physics-inspired distortions (salt & pepper, drift, atomic plane)
- 📊 **Metrics**: Real-time PSNR and SSIM logging during training
- 🧼 **Mixed Precision Training**: Optional float16 for speed and memory efficiency
- 📉 **Callbacks & Checkpoints**: Integrated learning rate scheduler, early stopping, model saving
- 🖼️ **Visualization**: Predictions, training metrics, and component-wise loss plots

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/stem-denoising.git
cd stem-denoising
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Data

Organize your data like:

```
data/raw/
├── noisy/
│   └── noisy1.png
├── clean/
│   └── clean1.png
```

Then run the following in Python:

```python
from data.dataset import patch_and_save, split_dataset, offline_augmentation
```

### 4. Train the Model

```python
from models.unet import build_unet
from training.train import train_model, get_callbacks

model = build_unet()
train_model(model, train_ds, val_ds,
            alpha=1.0, beta=0.3, gamma=0.3,
            delta=0.1, epsilon=0.05, mu=0.2,
            callbacks=get_callbacks())
```

---

## 📈 Visualizations and Evaluation

```python
from evaluation.visualize import visualize_predictions, plot_training_metrics
```

- Compare input vs. prediction vs. ground truth
- Plot epoch-wise training loss and metrics
- Save experiment summaries to CSV

---

## 📊 Sample Results

| Noisy Input | Ground Truth | Denoised Output |
|-------------|---------------|-----------------|
| ![noisy](samples/noisy.png) | ![clean](samples/clean.png) | ![output](samples/pred.png) |

---

## 📜 License

This repository is distributed under the MIT License. See `LICENSE` for details.

---

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements or new features.

---

## 🔬 Citation

If you use this project in your work, please consider citing:

```bibtex
@misc{yourproject2025,
  title={Deep Learning Denoising for High-Resolution STEM Images},
  author={Ali baghi Zadeh},
  year={2025},
  note={!git clone https://github.com/AliBaghizadeh/high-resolution-microscopy-denoising.git}
}
```

---

## 🙋 Contact

Questions, feedback, or collaborations? Reach out via GitHub Issues or [alibaghizade@gmail.com](mailto:alibaghizade@gmail.com)

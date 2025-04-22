# STEM Image Denoising with Deep Learning

A comprehensive research codebase for denoising high-resolution STEM (Scanning Transmission Electron Microscopy) images using deep learning approaches.

## Overview

This repository contains a modular and well-structured implementation of various UNet-based deep learning models for STEM image denoising, including:

- Standard UNet from scratch
- Residual UNet with skip connections
- Deeper UNet architectures
- UNet with pre-trained encoders (VGG16, ResNet50, EfficientNet)

The codebase includes data preprocessing pipelines, model training, evaluation metrics specific to microscopy, and comprehensive visualization tools.

## Project Structure

```
stem-denoising/
├── data/
│   ├── raw/                # Raw STEM images
│   ├── processed/          # Processed, patched images
│   └── dataset.py          # Dataset loading and preprocessing
├── models/
│   ├── unet.py             # UNet implementation
│   ├── pretrained.py       # Pre-trained CNN UNet variants
│   └── utils.py            # Model utilities
├── training/
│   ├── train.py            # Training logic
│   ├── loss_functions.py   # All loss functions
│   └── metrics.py          # Evaluation metrics
├── evaluation/
│   ├── eval.py             # Evaluation scripts
│   └── visualize.py        # Result visualization
├── config.py               # Configuration parameters
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

## Requirements

- Python 3.7+
- TensorFlow 2.4+
- numpy, pandas, matplotlib, scikit-image
- tqdm, patchify
- scipy

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

### Dataset Structure

The data should be organized as follows:

```
data/
├── raw/
│   ├── noisy/        # Original noisy STEM images
│   └── clean/        # Corresponding clean images (ground truth)
└── processed/        # Will contain processed and patched images
```

### Preprocessing Pipeline

1. **Image Patching**: Divides large STEM images into patches for training:

```python
from data.dataset import patch_and_save

# Extract patches from raw images
patch_and_save(
    source_noisy="data/raw/noisy",
    source_clean="data/raw/clean", 
    output_dir="data/processed/patches"
)
```

2. **Dataset Splitting**: Split data into train/validation/test sets:

```python
from data.dataset import split_dataset

# Split the dataset
split_dataset(
    source_dir="data/processed/patches",
    output_dir="data/processed/final",
    train_ratio=0.8,
    val_ratio=0.1
)
```

## Model Training

### Basic Training Example

```python
from data.dataset import DenoiseDataGenerator
from training.train import DenoiseTrainer

# Initialize data generators
data_dir = "data/processed/final"
batch_size = 8

data_generator = DenoiseDataGenerator(data_dir, batch_size=batch_size)
train_gen = data_generator.get_generator('train')
val_gen = data_generator.get_generator('val')

# Initialize the trainer
trainer = DenoiseTrainer(
    model_name='unet',
    input_size=(256, 256, 1),
    learning_rate=1e-4,
    batch_size=batch_size,
    output_dir='./output'
)

# Compile the model
trainer.compile()

# Train the model
trainer.train(train_gen, val_gen, epochs=100)
```

### Loss Functions

Several loss functions are implemented for denoising, including:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Structural Similarity Index (SSIM)
- Combined MSE + SSIM loss
- Perceptual loss (VGG16-based)
- Gradient loss for edge preservation

### Model Selection

Choose from various pre-implemented model architectures:

```python
# Standard UNet
trainer = DenoiseTrainer(model_name='unet')

# Deeper UNet
trainer = DenoiseTrainer(model_name='deep_unet')

# Residual UNet
trainer = DenoiseTrainer(model_name='residual_unet')

# Using pre-trained models
trainer = DenoiseTrainer(model_name='vgg16_unet')
trainer = DenoiseTrainer(model_name='resnet50_unet')
trainer = DenoiseTrainer(model_name='efficient_unet')
```

## Evaluation

### Basic Evaluation

```python
from evaluation.eval import DenoiseEvaluator

# Initialize evaluator
evaluator = DenoiseEvaluator(output_dir='./evaluation_results')

# Load models
evaluator.load_model('output/models/unet_model.h5', 'UNet')
evaluator.load_model('output/models/residual_unet_model.h5', 'ResidualUNet')

# Load test data
from data.dataset import DenoiseDataGenerator
data_generator = DenoiseDataGenerator("data/processed/final", batch_size=8)
test_gen = data_generator.get_generator('test')

# Evaluate individual models
evaluator.evaluate_model('UNet', test_gen, visualize=True)
evaluator.evaluate_model('ResidualUNet', test_gen, visualize=True)

# Compare models
evaluator.compare_models(['UNet', 'ResidualUNet'], test_gen)
```

### Metrics

The evaluation includes various metrics tailored for image quality assessment:

- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)
- Multi-Scale SSIM
- Edge Preservation
- Haar Wavelet-Based Perceptual Similarity Index (HaarPSI)
- Fréchet Inception Distance (FID)

### Visualizations

The evaluation tools generate comprehensive visualizations:

- Side-by-side comparisons of noisy, denoised, and ground truth images
- Edge detection visualizations to assess edge preservation
- Difference maps to highlight areas of improvement or remaining artifacts
- Residual noise analysis
- Model comparison visualizations

## Configuration

All project parameters are centralized in `config.py`. You can customize:

- Data preprocessing parameters
- Model architectures and hyperparameters
- Training settings
- Evaluation metrics and visualization options

## Ethical Considerations

This project carefully analyzes the impact of different loss functions on denoising quality, with a specific focus on preserving scientifically important features in STEM images. Ethical considerations include:

- Data integrity preservation
- Avoiding hallucination of non-existent features
- Transparency in reporting denoising artifacts
- Ensuring reproducibility of results

## Citation

If you use this code for your research, please cite:

```
@article{stem_denoising,
  title={Denoising High-Resolution STEM Images Using Deep Learning},
  author={Ali Baghi Zadeh},
  year={2025},
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This research acknowledges [relevant collaborators/institutions]
- Parts of the implementation were inspired by [relevant prior work] 

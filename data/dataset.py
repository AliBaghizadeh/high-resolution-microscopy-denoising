"""
STEM Denoising Model Training Demo for Google Colab
==================================================

This script demonstrates how to train a STEM image denoising model
using the high-resolution-microscopy-denoising framework.
"""

import os
import tensorflow as tf
import matplotlib.pyplot as plt
from google.colab import drive
import numpy as np

# Import our custom modules
from data.dataset import DenoiseDataGenerator
from training.train import DenoiseTrainer
from training.loss_functions import create_combined_loss_fn, ssim_loss, psnr_metric

# Mount Google Drive to access your data
drive.mount('/content/drive')

# Set up paths for data
drive_clean_dir = '/content/drive/MyDrive/dataset_converted/clean'
drive_noisy_dir = '/content/drive/MyDrive/dataset_converted/noisy'

# Check GPU availability
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# Create directories for processed data
os.makedirs('data/processed/train/clean', exist_ok=True)
os.makedirs('data/processed/train/noisy', exist_ok=True)
os.makedirs('data/processed/val/clean', exist_ok=True)
os.makedirs('data/processed/val/noisy', exist_ok=True)
os.makedirs('data/processed/test/clean', exist_ok=True)
os.makedirs('data/processed/test/noisy', exist_ok=True)

# Process and patch your data here
# This would typically involve:
# 1. Loading images from drive_clean_dir and drive_noisy_dir
# 2. Creating patches
# 3. Splitting into train/val/test
# 4. Saving to the directories created above

# For demo purposes, we'll assume data is already processed

# Create data generators
data_generator = DenoiseDataGenerator(
    data_dir='data/processed',
    batch_size=16,
    img_size=(128, 128),
    augmentation=True,
    grayscale=True  # Ensure images are grayscale (1 channel)
)

# Get datasets for training, validation, and testing
train_dataset = data_generator.get_generator(split='train')
val_dataset = data_generator.get_generator(split='val')
test_dataset = data_generator.get_generator(split='test')

# Create custom loss function with all required parameters
custom_loss = create_combined_loss_fn(
    alpha=0.84,  # MSE weight
    beta=0.06,   # FFT weight
    gamma=0.1,   # Poisson weight
    delta=0,     # SSIM weight
    epsilon=0,   # TV weight
    mu=0         # Mean intensity weight
)

# Initialize trainer
trainer = DenoiseTrainer(
    model_name='unet',
    input_size=(128, 128, 1),  # Make sure this matches your image dimensions
    learning_rate=1e-4,
    batch_size=16,
    output_dir='output',
    use_mixed_precision=True
)

# Compile the model
trainer.compile(
    loss_fn=custom_loss,
    metrics=[ssim_loss, psnr_metric]
)

# Train the model
history = trainer.train(
    train_generator=train_dataset,
    val_generator=val_dataset,
    epochs=50,
    early_stopping=True,
    patience=10
)

# Evaluate the model
evaluation_results = trainer.evaluate(
    test_generator=test_dataset,
    visualize=True,
    num_samples=5
)

# Save the model
model_path = os.path.join('output', 'models', 'unet_final.h5')
trainer.model.save(model_path)

# Download the model to your local machine
from google.colab import files
files.download(model_path)

print("Training and evaluation complete!") 

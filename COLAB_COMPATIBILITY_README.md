# Making the STEM Denoising Repository Compatible with Google Colab

This guide explains how to update your repository to make it fully compatible with Google Colab while preserving the original functionality of your notebooks.

## Files to Update

### 1. Update training/loss_functions.py (if needed)

Make sure your loss_functions.py contains the necessary functions from your original notebooks and also has the `get_loss_function` factory function at the end. If it's missing this function, you can add it:

```python
# Loss function factory for compatibility with Keras
def get_loss_function(alpha=1.0, beta=0.1, gamma=0.1, delta=0.1, epsilon=0.001, mu=0.1):
    """
    Creates a combined loss function with specified weights that can be passed to model.compile().
    
    Args:
        alpha (float): Weight for MSE loss (default: 1.0)
        beta (float): Weight for FFT loss (default: 0.1)
        gamma (float): Weight for Poisson loss (default: 0.1)
        delta (float): Weight for SSIM loss (default: 0.1)
        epsilon (float): Weight for TV loss (default: 0.001)
        mu (float): Weight for mean intensity loss (default: 0.1)
        
    Returns:
        function: A loss function compatible with Keras models
    """
    def loss_function(y_true, y_pred):
        return combined_loss(y_true, y_pred, alpha, beta, gamma, delta, epsilon, mu)
    
    return loss_function
```

### 2. Update training/train.py

Replace relative imports with absolute imports in training/train.py:

```python
# Change from relative imports:
from ..models.unet import unet_model, deep_unet_model, residual_unet_model
from ..models.pretrained import vgg16_unet, resnet50_unet, efficient_unet
from .loss_functions import combined_loss, ssim_loss, psnr_metric

# To absolute imports:
from models.unet import unet_model, deep_unet_model, residual_unet_model
from models.pretrained import vgg16_unet, resnet50_unet, efficient_unet
from training.loss_functions import combined_loss, ssim_loss, psnr_metric
```

### 3. Add colab_setup.py

Create a new file called `colab_setup.py` in the root directory with the following content:

```python
"""
Helper functions for setting up the STEM denoising project in Google Colab
"""
import sys
import os
import tensorflow as tf
import matplotlib.pyplot as plt

def setup_colab():
    """
    Set up the environment for Colab by adding the current directory to Python path
    and returning the working directory.
    """
    # Get current directory
    current_dir = os.getcwd()
    
    # Add the current directory to Python path
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    print(f"Repository path added to sys.path: {current_dir}")
    print(f"Available directories: {os.listdir(current_dir)}")
    
    # Set up output directories
    os.makedirs('output/models', exist_ok=True)
    os.makedirs('output/logs', exist_ok=True)
    os.makedirs('output/samples', exist_ok=True)
    
    # Check for GPU
    print("\nChecking GPU availability:")
    print("TensorFlow version:", tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU is available! Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  - {gpu}")
        try:
            # Set memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth enabled for GPUs")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")
    else:
        print("No GPU found. Training will be slow on CPU!")
    
    return current_dir

def prepare_data_for_training(data_gen, split='train'):
    """
    Convert generator to format suitable for model.fit in Colab.
    
    Args:
        data_gen: DenoiseDataGenerator instance
        split: Data split ('train', 'val', or 'test')
        
    Returns:
        generator_fn or tf.data.Dataset and steps_per_epoch
    """
    gen = data_gen.get_generator(split)
    
    if isinstance(gen, zip):
        print(f"Converting {split} zip generator to format suitable for training...")
        
        # Create a generator function that Keras can use
        def generator_fn():
            while True:
                # Get a new generator each time to avoid depletion
                gen = data_gen.get_generator(split)
                for noisy_batch, clean_batch in gen:
                    yield noisy_batch, clean_batch
        
        # Calculate steps
        dir_path = os.path.join(data_gen.data_dir, split, 'noisy')
        num_images = len(os.listdir(dir_path))
        steps = num_images // data_gen.batch_size
        
        print(f"Found {num_images} images in {split} set, using {steps} steps per epoch")
        return generator_fn(), steps
    else:
        # tf.data.Dataset can be used directly
        print(f"Using tf.data.Dataset for {split} set")
        return gen, None

def visualize_batch(noisy_batch, clean_batch, num_samples=4):
    """
    Visualize a batch of training data.
    
    Args:
        noisy_batch: Batch of noisy images
        clean_batch: Batch of clean images
        num_samples: Number of samples to visualize
    """
    plt.figure(figsize=(12, 6))
    
    for i in range(num_samples):
        if i < len(noisy_batch):
            # Display noisy image
            plt.subplot(2, num_samples, i+1)
            img = noisy_batch[i].squeeze()
            plt.imshow(img, cmap='gray')
            plt.title(f"Noisy {i+1}")
            plt.axis('off')

            # Display clean image
            plt.subplot(2, num_samples, i+num_samples+1)
            img = clean_batch[i].squeeze()
            plt.imshow(img, cmap='gray')
            plt.title(f"Clean {i+1}")
            plt.axis('off')

    plt.tight_layout()
    plt.show()
```

### 4. Ensure __init__.py files exist in all directories

- /\_\_init\_\_.py
- /data/\_\_init\_\_.py
- /models/\_\_init\_\_.py
- /training/\_\_init\_\_.py
- /evaluation/\_\_init\_\_.py

You can create them as empty files or add simple docstrings.

### 5. Create stem_denoising_colab_demo.ipynb

Create a Colab demo notebook with the following content:

```python
# Step 1: Clone the repository and set up the environment
!git clone https://github.com/YOUR_USERNAME/high-resolution-microscopy-denoising.git
%cd high-resolution-microscopy-denoising
!pip install -r requirements.txt

# Step 2: Import necessary modules and set up the environment
from colab_setup import setup_colab, prepare_data_for_training, visualize_batch
setup_colab()

# Step 3: Import the project modules
from data.dataset import DenoiseDataGenerator
from training.train import DenoiseTrainer
from training.loss_functions import get_loss_function, ssim_loss, psnr_metric

# Step 4: Initialize data generators
data_dir = "data/processed/final"  # Update to your data directory
batch_size = 8

data_generator = DenoiseDataGenerator(data_dir, batch_size=batch_size)
train_gen, train_steps = prepare_data_for_training(data_generator, 'train')
val_gen, val_steps = prepare_data_for_training(data_generator, 'val')
test_gen, test_steps = prepare_data_for_training(data_generator, 'test')

# Step 5: Visualize training data
if isinstance(train_gen, zip):
    noisy_batch, clean_batch = next(iter(train_gen))
    visualize_batch(noisy_batch, clean_batch)
    # Re-initialize the generator since we consumed it
    train_gen, train_steps = prepare_data_for_training(data_generator, 'train')
else:
    for noisy_batch, clean_batch in train_gen.take(1):
        visualize_batch(noisy_batch, clean_batch)

# Step 6: Initialize and train the model
output_dir = "output"
trainer = DenoiseTrainer(
    model_name="unet",
    input_size=(256, 256, 1),  # Update based on your image dimensions
    learning_rate=1e-4,
    batch_size=batch_size,
    output_dir=output_dir
)

# Custom loss function
loss_function = get_loss_function(
    alpha=1.0,    # MSE weight
    beta=0.1,     # FFT loss weight
    gamma=0.1,    # Poisson loss weight
    delta=0.5,    # SSIM loss weight
    epsilon=0.001, # TV loss weight
    mu=0.1        # Mean intensity loss weight
)

trainer.compile(
    loss=loss_function,
    metrics=[ssim_loss, psnr_metric]
)

# Train the model
history = trainer.train(
    train_generator=train_gen,
    val_generator=val_gen,
    epochs=50,
    early_stopping=True,
    patience=10
)

# Step 7: Evaluate the model and visualize results
metrics = trainer.evaluate(test_gen, visualize=True, num_samples=5)
print("Evaluation metrics:", metrics)
```

## Important Notes

1. These changes **do not** modify your original code functionality - they just make it compatible with Colab's import system.

2. The changes are minimal and focused on making imports work correctly in Colab, while preserving all of your original loss functions and training code.

3. The colab_setup.py file provides helper functions to deal with the ImageDataGenerator zip objects in Colab.

4. Always test these changes in Colab before finalizing them in your repository. 
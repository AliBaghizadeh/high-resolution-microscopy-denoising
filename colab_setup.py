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

def create_colab_demo_notebook():
    """
    Print code for a Colab demo notebook.
    """
    demo_code = """
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
"""
    print(demo_code)
    
if __name__ == "__main__":
    setup_colab()
    create_colab_demo_notebook() 
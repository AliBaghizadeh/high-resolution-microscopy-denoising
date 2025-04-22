import os
import numpy as np
from PIL import Image
from patchify import patchify
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
PATCH_SIZE = 256
STRIDE = 128

def load_image(image_path):
    """
    Loads an image from a given path and converts it to a NumPy array.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.array: The loaded image as a NumPy array.
    """
    return np.array(Image.open(image_path))

def create_patches(image, patch_size=PATCH_SIZE, stride=STRIDE):
    """
    Divides an image into structured overlapping patches.

    Args:
        image (np.array): Input image.
        patch_size (int): Size of each patch.
        stride (int): Stride step size.

    Returns:
        np.array: Patches reshaped into (total_patches, patch_size, patch_size).
    """
    return patchify(image, (patch_size, patch_size), step=stride).reshape(-1, patch_size, patch_size)

def patch_and_save(source_noisy, source_clean, output_dir):
    """
    Converts noisy-clean image pairs into patches and saves them.

    Args:
        source_noisy (str): Directory containing noisy images.
        source_clean (str): Directory containing clean images.
        output_dir (str): Directory where the patches will be saved.

    Returns:
        None
    """
    os.makedirs(os.path.join(output_dir, "noisy"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "clean"), exist_ok=True)

    noisy_files = sorted(os.listdir(source_noisy))
    clean_files = sorted(os.listdir(source_clean))

    for noisy_file, clean_file in zip(noisy_files, clean_files):
        noisy_img = load_image(os.path.join(source_noisy, noisy_file))
        clean_img = load_image(os.path.join(source_clean, clean_file))

        noisy_patches = create_patches(noisy_img)
        clean_patches = create_patches(clean_img)

        noisy_base = os.path.splitext(noisy_file)[0]
        clean_base = os.path.splitext(clean_file)[0]

        for i, (n_patch, c_patch) in enumerate(zip(noisy_patches, clean_patches)):
            Image.fromarray(n_patch.astype(np.uint8)).save(f"{output_dir}/noisy/{noisy_base}_patch_{i:03d}.png")
            Image.fromarray(c_patch.astype(np.uint8)).save(f"{output_dir}/clean/{clean_base}_patch_{i:03d}.png")

def split_dataset(source_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Splits a dataset into train, validation, and test sets.

    Args:
        source_dir (str): Path to dataset containing patched images.
        output_dir (str): Path where the split dataset will be stored.
        train_ratio (float): Ratio of training data. Default is 80%.
        val_ratio (float): Ratio of validation data. Default is 10%.

    Returns:
        None
    """
    all_files = sorted(os.listdir(os.path.join(source_dir, "noisy")))
    random.shuffle(all_files)

    total = len(all_files)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    splits = {
        "train": all_files[:train_end],
        "val": all_files[train_end:val_end],
        "test": all_files[val_end:]
    }

    # Create directories
    for split in splits:
        os.makedirs(f"{output_dir}/{split}/noisy", exist_ok=True)
        os.makedirs(f"{output_dir}/{split}/clean", exist_ok=True)

    # Copy files
    for split, files in splits.items():
        for file in files:
            shutil.copy(f"{source_dir}/noisy/{file}", f"{output_dir}/{split}/noisy/{file}")
            shutil.copy(f"{source_dir}/clean/{file}", f"{output_dir}/{split}/clean/{file}")

class DenoiseDataGenerator:
    """
    Data generator for STEM image denoising using TensorFlow.
    """
    
    def __init__(self, data_dir, batch_size=8, img_size=(256, 256), augmentation=True):
        """
        Initialize the data generator.
        
        Args:
            data_dir (str): Directory containing 'noisy' and 'clean' subdirectories
            batch_size (int): Batch size for training
            img_size (tuple): Image dimensions (height, width)
            augmentation (bool): Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.augmentation = augmentation
        
    def get_generator(self, split='train'):
        """
        Create a data generator for the specified split.
        
        Args:
            split (str): One of 'train', 'val', or 'test'
            
        Returns:
            tf.data.Dataset: Generator yielding (noisy_batch, clean_batch) pairs
        """
        noisy_dir = os.path.join(self.data_dir, split, 'noisy')
        clean_dir = os.path.join(self.data_dir, split, 'clean')
        
        if self.augmentation and split == 'train':
            # Setup augmentation parameters
            data_gen_args = dict(
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='reflect'
            )
            
            # Create image data generators
            image_gen = ImageDataGenerator(**data_gen_args)
            mask_gen = ImageDataGenerator(**data_gen_args)
            
            # Provide the same seed for both generators
            seed = 42
            
            noisy_gen = image_gen.flow_from_directory(
                os.path.dirname(noisy_dir),
                classes=[os.path.basename(noisy_dir)],
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode=None,
                seed=seed
            )
            
            clean_gen = mask_gen.flow_from_directory(
                os.path.dirname(clean_dir),
                classes=[os.path.basename(clean_dir)],
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode=None,
                seed=seed
            )
            
            # Combine generators
            return zip(noisy_gen, clean_gen)
        
        else:
            # Without augmentation, use simple loading
            noisy_files = sorted([os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir)])
            clean_files = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir)])
            
            def load_and_preprocess(noisy_path, clean_path):
                noisy_img = tf.image.decode_png(tf.io.read_file(noisy_path), channels=1)
                clean_img = tf.image.decode_png(tf.io.read_file(clean_path), channels=1)
                
                noisy_img = tf.image.resize(noisy_img, self.img_size)
                clean_img = tf.image.resize(clean_img, self.img_size)
                
                # Normalize to [0,1]
                noisy_img = tf.cast(noisy_img, tf.float32) / 255.0
                clean_img = tf.cast(clean_img, tf.float32) / 255.0
                
                return noisy_img, clean_img
            
            dataset = tf.data.Dataset.from_tensor_slices((noisy_files, clean_files))
            dataset = dataset.map(lambda x, y: tf.py_function(
                func=load_and_preprocess, inp=[x, y], Tout=[tf.float32, tf.float32]
            ))
            
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            
            return dataset 
import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from patchify import patchify
import cv2
import shutil

def load_image(path):
    '''
    Load an image from a given file path.

    Args:
        path (str): Path to the image file.

    Returns:
        np.ndarray: Loaded image as a NumPy array.
    '''
    return np.array(Image.open(path))

def create_patches(image, patch_size=256, stride=128):
    '''
    Divide an image into overlapping patches.

    Args:
        image (np.ndarray): Input image.
        patch_size (int): Size of square patches.
        stride (int): Step between patches.

    Returns:
        np.ndarray: Array of shape (num_patches, patch_size, patch_size)
    '''
    return patchify(image, (patch_size, patch_size), step=stride).reshape(-1, patch_size, patch_size)

def patch_and_save(source_noisy, source_clean, output_dir):
    '''
    Generate patches from paired noisy and clean images and save to disk.

    Args:
        source_noisy (str): Directory containing noisy images.
        source_clean (str): Directory containing clean images.
        output_dir (str): Output directory to save patches.
    '''
    os.makedirs(os.path.join(output_dir, "noisy"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "clean"), exist_ok=True)

    for nf, cf in zip(sorted(os.listdir(source_noisy)), sorted(os.listdir(source_clean))):
        noisy_img = load_image(os.path.join(source_noisy, nf))
        clean_img = load_image(os.path.join(source_clean, cf))

        n_patches = create_patches(noisy_img)
        c_patches = create_patches(clean_img)

        for i, (n, c) in enumerate(zip(n_patches, c_patches)):
            Image.fromarray(n.astype(np.uint8)).save(f"{output_dir}/noisy/{nf[:-4]}_patch_{i:03d}.png")
            Image.fromarray(c.astype(np.uint8)).save(f"{output_dir}/clean/{cf[:-4]}_patch_{i:03d}.png")

def split_dataset(source_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    '''
    Split images into train/val/test sets based on filename shuffling.

    Args:
        source_dir (str): Directory containing patched data.
        output_dir (str): Directory to save split datasets.
        train_ratio (float): Proportion of images for training.
        val_ratio (float): Proportion for validation.

    Remaining will be used as test set.
    '''
    files = sorted(os.listdir(os.path.join(source_dir, "noisy")))
    random.shuffle(files)
    total = len(files)
    t_end = int(total * train_ratio)
    v_end = int(total * (train_ratio + val_ratio))

    for split, split_files in zip(["train", "val", "test"], [files[:t_end], files[t_end:v_end], files[v_end:]]):
        for cat in ["noisy", "clean"]:
            os.makedirs(f"{output_dir}/{split}/{cat}", exist_ok=True)
        for f in split_files:
            for cat in ["noisy", "clean"]:
                shutil.copy(f"{source_dir}/{cat}/{f}", f"{output_dir}/{split}/{cat}/{f}")

# Augmentation functions
def add_salt_pepper_noise(image, prob=0.04):
    '''
    Add salt and pepper noise to a grayscale image.

    Args:
        image (np.ndarray): Input image normalized [0, 1].
        prob (float): Probability of pixel corruption.

    Returns:
        np.ndarray: Noisy image.
    '''
    image_np = image.copy()
    rand = np.random.uniform(size=image_np.shape)
    image_np[rand < prob / 2] = 0.0
    image_np[rand > (1 - prob / 2)] = 1.0
    return image_np.astype(np.float32)

def atomic_plane_distortion_fixed(image, frequency=10, intensity=0.05):
    '''
    Simulate fixed periodic distortion in atomic planes.

    Args:
        image (np.ndarray): Input image.
        frequency (int): Row frequency for distortion.
        intensity (float): Magnitude of distortion.

    Returns:
        np.ndarray: Distorted image.
    '''
    image_np = image.copy()
    rows, cols = image_np.shape
    sin_pattern = np.sin(np.linspace(0.0, np.pi * 2 * (rows // frequency), rows))[:, np.newaxis]
    return np.clip(image_np + sin_pattern * intensity, 0.0, 1.0).astype(np.float32)

def atomic_plane_distortion_random(image, probability=0.1, intensity=0.1):
    '''
    Apply random distortions to some atomic planes.

    Args:
        image (np.ndarray): Input image.
        probability (float): Fraction of rows affected.
        intensity (float): Strength of distortion.

    Returns:
        np.ndarray: Distorted image.
    '''
    image_np = image.copy()
    rows, cols = image_np.shape
    mask = np.random.uniform(0, 1, size=(rows,)) < probability
    noise = np.random.uniform(-intensity, intensity, size=(rows, cols))
    distorted_image = np.where(mask[:, np.newaxis], image_np + noise, image_np)
    return np.clip(distorted_image, 0.0, 1.0).astype(np.float32)

def scan_distortion(image, frequency=5, intensity=3):
    '''
    Introduce periodic scan-line distortions.

    Args:
        image (np.ndarray): Input image.
        frequency (int): Rows between each distortion.
        intensity (float): Max number of pixels to shift.

    Returns:
        np.ndarray: Distorted image.
    '''
    distorted_image = image.copy()
    rows, cols = image.shape
    for i in range(0, rows, frequency):
        shift = int(intensity * np.sin(i / frequency * np.pi))
        distorted_image[i, :] = np.roll(image[i, :], shift, axis=0)
    return distorted_image

def drift_distortion(image, frequency=6, intensity=2):
    '''
    Add drift-like noise in a staggered fashion across atomic rows.

    Args:
        image (np.ndarray): Input image.
        frequency (int): Row interval.
        intensity (int): Maximum shift range.

    Returns:
        np.ndarray: Drift-distorted image.
    '''
    distorted_image = image.copy()
    rows, cols = image.shape
    for i in range(0, rows, frequency):
        shift = np.random.randint(-intensity, intensity + 1)
        distorted_image[i, :] = np.roll(image[i, :], shift, axis=0)
    return distorted_image

def offline_augmentation(noisy_paths, clean_paths, save_dir):
    '''
    Apply a sequence of augmentations to noisy images while keeping clean images geometrically aligned.

    Args:
        noisy_paths (list): List of paths to noisy input images.
        clean_paths (list): List of paths to corresponding clean images.
        save_dir (str): Directory to save augmented images.

    Saves:
        Augmented pairs to 'noisy/' and 'clean/' folders under save_dir.
    '''
    augmented_noisy_dir = os.path.join(save_dir, "noisy")
    augmented_clean_dir = os.path.join(save_dir, "clean")
    os.makedirs(augmented_noisy_dir, exist_ok=True)
    os.makedirs(augmented_clean_dir, exist_ok=True)

    for i, (noisy_path, clean_path) in enumerate(zip(noisy_paths, clean_paths)):
        noisy_image = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE) / 255.0
        clean_image = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE) / 255.0

        if np.random.rand() > 0.5:
            noisy_image = np.fliplr(noisy_image)
            clean_image = np.fliplr(clean_image)
        if np.random.rand() > 0.5:
            noisy_image = np.flipud(noisy_image)
            clean_image = np.flipud(clean_image)

        k = np.random.randint(0, 4)
        noisy_image = np.rot90(noisy_image, k)
        clean_image = np.rot90(clean_image, k)

        if np.random.rand() < 0.5:
            noisy_image = add_salt_pepper_noise(noisy_image, np.random.uniform(0.01, 0.03))
        if np.random.rand() < 0.4:
            noisy_image = atomic_plane_distortion_fixed(noisy_image, np.random.randint(30, 60), np.random.uniform(0.03, 0.07))
        if np.random.rand() < 0.4:
            noisy_image = atomic_plane_distortion_random(noisy_image, np.random.uniform(0.05, 0.15), np.random.uniform(0.03, 0.08))
        if np.random.rand() < 0.3:
            noisy_image = scan_distortion(noisy_image, np.random.randint(30, 70), np.random.uniform(1, 3))
        if np.random.rand() < 0.3:
            noisy_image = drift_distortion(noisy_image, np.random.randint(30, 70), np.random.randint(2, 5))

        cv2.imwrite(os.path.join(augmented_noisy_dir, f"noisy_{i}.png"), noisy_image * 255)
        cv2.imwrite(os.path.join(augmented_clean_dir, f"clean_{i}.png"), clean_image * 255)

    print(f"Augmented dataset saved in {save_dir}")

def preprocess_validation_data(noisy_path, clean_path):
    '''
    Preprocess a single noisy/clean image pair without augmentation.

    Args:
        noisy_path (str): File path to noisy image.
        clean_path (str): File path to clean image.

    Returns:
        Tuple of (noisy_image, clean_image) as tf.Tensor.
    '''
    noisy_image = tf.io.read_file(noisy_path)
    clean_image = tf.io.read_file(clean_path)
    noisy_image = tf.image.decode_png(noisy_image, channels=1)
    clean_image = tf.image.decode_png(clean_image, channels=1)
    noisy_image = tf.image.convert_image_dtype(noisy_image, tf.float32)
    clean_image = tf.image.convert_image_dtype(clean_image, tf.float32)
    return noisy_image, clean_image

def load_pre_augmented_dataset(aug_noisy_dir, aug_clean_dir, batch_size=16):
    '''
    Load a directory of augmented noisy/clean image pairs into a tf.data.Dataset.

    Args:
        aug_noisy_dir (str): Directory of noisy images.
        aug_clean_dir (str): Directory of clean images.
        batch_size (int): Batch size for loading.

    Returns:
        tf.data.Dataset: Dataset object for training.
    '''
    noisy_files = sorted([os.path.join(aug_noisy_dir, f) for f in os.listdir(aug_noisy_dir)])
    clean_files = sorted([os.path.join(aug_clean_dir, f) for f in os.listdir(aug_clean_dir)])
    dataset = tf.data.Dataset.from_tensor_slices((noisy_files, clean_files))

    def load_images(noisy_path, clean_path):
        noisy_image = tf.io.read_file(noisy_path)
        clean_image = tf.io.read_file(clean_path)
        noisy_image = tf.image.decode_png(noisy_image, channels=1)
        clean_image = tf.image.decode_png(clean_image, channels=1)
        noisy_image = tf.image.convert_image_dtype(noisy_image, tf.float32)
        clean_image = tf.image.convert_image_dtype(clean_image, tf.float32)
        return noisy_image, clean_image

    return dataset.map(load_images).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def create_validation_dataset(noisy_paths, clean_paths, batch_size=32):
    '''
    Create a validation dataset from lists of image paths.

    Args:
        noisy_paths (list): List of file paths to noisy images.
        clean_paths (list): List of file paths to clean images.
        batch_size (int): Number of image pairs per batch.

    Returns:
        tf.data.Dataset: Prepared validation dataset.
    '''
    dataset = tf.data.Dataset.from_tensor_slices((noisy_paths, clean_paths))
    dataset = dataset.map(preprocess_validation_data).shuffle(500).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from skimage.feature import canny
from skimage.segmentation import mark_boundaries
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from ..training.metrics import psnr, ssim

def display_comparison(noisy, denoised, clean, save_path=None, title=None):
    """
    Display a side-by-side comparison of noisy, denoised, and clean images.
    
    Args:
        noisy: Noisy input image
        denoised: Denoised image (model output)
        clean: Clean reference image (ground truth)
        save_path: Path to save the figure (if None, just displays it)
        title: Title for the figure
    """
    # Ensure images are in the right shape
    noisy = np.squeeze(noisy)
    denoised = np.squeeze(denoised)
    clean = np.squeeze(clean)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display images
    axes[0].imshow(noisy, cmap='gray')
    axes[0].set_title('Noisy Input')
    axes[0].axis('off')
    
    axes[1].imshow(denoised, cmap='gray')
    axes[1].set_title('Denoised Output')
    axes[1].axis('off')
    
    axes[2].imshow(clean, cmap='gray')
    axes[2].set_title('Clean Reference')
    axes[2].axis('off')
    
    # Add metrics as text
    psnr_value = psnr(tf.convert_to_tensor([clean]), tf.convert_to_tensor([denoised])).numpy()
    ssim_value = ssim(tf.convert_to_tensor([clean]), tf.convert_to_tensor([denoised])).numpy()
    
    fig.text(0.5, 0.01, f'PSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.4f}', 
             ha='center', fontsize=12)
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def display_edge_comparison(noisy, denoised, clean, save_path=None, title=None):
    """
    Compare edge preservation in denoised images.
    
    Args:
        noisy: Noisy input image
        denoised: Denoised image (model output)
        clean: Clean reference image (ground truth)
        save_path: Path to save the figure (if None, just displays it)
        title: Title for the figure
    """
    # Ensure images are in the right shape
    noisy = np.squeeze(noisy)
    denoised = np.squeeze(denoised)
    clean = np.squeeze(clean)
    
    # Detect edges
    sigma = 2.0  # Edge detection parameter
    
    clean_edges = canny(clean, sigma=sigma)
    noisy_edges = canny(noisy, sigma=sigma)
    denoised_edges = canny(denoised, sigma=sigma)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Display original images
    axes[0, 0].imshow(noisy, cmap='gray')
    axes[0, 0].set_title('Noisy Input')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(denoised, cmap='gray')
    axes[0, 1].set_title('Denoised Output')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(clean, cmap='gray')
    axes[0, 2].set_title('Clean Reference')
    axes[0, 2].axis('off')
    
    # Display edge maps
    axes[1, 0].imshow(noisy_edges, cmap='gray')
    axes[1, 0].set_title('Noisy Edges')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(denoised_edges, cmap='gray')
    axes[1, 1].set_title('Denoised Edges')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(clean_edges, cmap='gray')
    axes[1, 2].set_title('Clean Edges')
    axes[1, 2].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def display_difference_map(denoised, clean, save_path=None, title=None):
    """
    Display difference map between denoised and clean images.
    
    Args:
        denoised: Denoised image (model output)
        clean: Clean reference image (ground truth)
        save_path: Path to save the figure (if None, just displays it)
        title: Title for the figure
    """
    # Ensure images are in the right shape
    denoised = np.squeeze(denoised)
    clean = np.squeeze(clean)
    
    # Calculate absolute difference
    diff = np.abs(denoised - clean)
    
    # Create a custom colormap - white for no difference, red for errors
    colors = [(1, 1, 1), (1, 0, 0)]  # White -> Red
    cmap = LinearSegmentedColormap.from_list('diff_cmap', colors, N=256)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display images
    axes[0].imshow(clean, cmap='gray')
    axes[0].set_title('Clean Reference')
    axes[0].axis('off')
    
    axes[1].imshow(denoised, cmap='gray')
    axes[1].set_title('Denoised Output')
    axes[1].axis('off')
    
    im = axes[2].imshow(diff, cmap=cmap, vmin=0, vmax=np.max(diff))
    axes[2].set_title('Difference Map')
    axes[2].axis('off')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Absolute Difference')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def display_residual_noise(noisy, denoised, save_path=None, title=None):
    """
    Display the residual noise removed by the model.
    
    Args:
        noisy: Noisy input image
        denoised: Denoised image (model output)
        save_path: Path to save the figure (if None, just displays it)
        title: Title for the figure
    """
    # Ensure images are in the right shape
    noisy = np.squeeze(noisy)
    denoised = np.squeeze(denoised)
    
    # Calculate residual (removed) noise
    residual = noisy - denoised
    
    # Create a custom colormap from blue (negative) to white (zero) to red (positive)
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue -> White -> Red
    cmap = LinearSegmentedColormap.from_list('residual_cmap', colors, N=256)
    
    # Determine max value for symmetric color scaling
    max_val = max(np.abs(np.min(residual)), np.abs(np.max(residual)))
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display images
    axes[0].imshow(noisy, cmap='gray')
    axes[0].set_title('Noisy Input')
    axes[0].axis('off')
    
    axes[1].imshow(denoised, cmap='gray')
    axes[1].set_title('Denoised Output')
    axes[1].axis('off')
    
    im = axes[2].imshow(residual, cmap=cmap, vmin=-max_val, vmax=max_val)
    axes[2].set_title('Residual Noise')
    axes[2].axis('off')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Residual Value')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def visualize_batch_results(model, test_batch, output_dir, prefix='sample'):
    """
    Visualize results for a batch of test images.
    
    Args:
        model: Trained model for denoising
        test_batch: Tuple of (noisy_images, clean_images)
        output_dir: Directory to save visualizations
        prefix: Prefix for saved files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Unpack the batch
    noisy_images, clean_images = test_batch
    
    # Generate predictions
    denoised_images = model.predict(noisy_images)
    
    # Visualize each sample
    for i in range(len(noisy_images)):
        noisy = noisy_images[i]
        denoised = denoised_images[i]
        clean = clean_images[i]
        
        # Basic comparison
        display_comparison(
            noisy, denoised, clean,
            save_path=os.path.join(output_dir, f'{prefix}_comparison_{i}.png'),
            title=f'Sample {i+1}'
        )
        
        # Edge comparison
        display_edge_comparison(
            noisy, denoised, clean,
            save_path=os.path.join(output_dir, f'{prefix}_edges_{i}.png'),
            title=f'Edge Preservation - Sample {i+1}'
        )
        
        # Difference map
        display_difference_map(
            denoised, clean,
            save_path=os.path.join(output_dir, f'{prefix}_diff_{i}.png'),
            title=f'Difference Map - Sample {i+1}'
        )
        
        # Residual noise
        display_residual_noise(
            noisy, denoised,
            save_path=os.path.join(output_dir, f'{prefix}_residual_{i}.png'),
            title=f'Residual Noise - Sample {i+1}'
        )

def create_metric_plots(histories, model_names, save_path=None):
    """
    Create comparison plots of training metrics for multiple models.
    
    Args:
        histories: List of training history objects from different models
        model_names: List of names for the models
        save_path: Path to save the figure (if None, just displays it)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot training loss
    for history, name in zip(histories, model_names):
        axes[0, 0].plot(history.history['loss'], label=name)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Plot validation loss
    for history, name in zip(histories, model_names):
        axes[0, 1].plot(history.history['val_loss'], label=name)
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    # Plot PSNR
    for history, name in zip(histories, model_names):
        axes[1, 0].plot(history.history['psnr_metric'], label=name)
    axes[1, 0].set_title('Training PSNR')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('PSNR (dB)')
    axes[1, 0].legend()
    
    # Plot validation PSNR
    for history, name in zip(histories, model_names):
        axes[1, 1].plot(history.history['val_psnr_metric'], label=name)
    axes[1, 1].set_title('Validation PSNR')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('PSNR (dB)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def visualize_model_comparison(models, model_names, test_batch, output_dir, sample_indices=None):
    """
    Compare denoising results from multiple models.
    
    Args:
        models: List of trained models for denoising
        model_names: List of names for the models
        test_batch: Tuple of (noisy_images, clean_images)
        output_dir: Directory to save visualizations
        sample_indices: List of sample indices to visualize (if None, uses the first 3)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Unpack the batch
    noisy_images, clean_images = test_batch
    
    # Default to first 3 samples if not specified
    if sample_indices is None:
        sample_indices = range(min(3, len(noisy_images)))
    
    # For each selected sample
    for idx in sample_indices:
        noisy = noisy_images[idx]
        clean = clean_images[idx]
        
        # Generate predictions from all models
        denoised_results = [model.predict(np.expand_dims(noisy, axis=0))[0] for model in models]
        
        # Create comparison figure
        n_models = len(models)
        fig = plt.figure(figsize=(4 * (n_models + 2), 4))
        gs = gridspec.GridSpec(1, n_models + 2)
        
        # Noisy image
        ax_noisy = fig.add_subplot(gs[0, 0])
        ax_noisy.imshow(np.squeeze(noisy), cmap='gray')
        ax_noisy.set_title('Noisy Input')
        ax_noisy.axis('off')
        
        # Clean image
        ax_clean = fig.add_subplot(gs[0, n_models + 1])
        ax_clean.imshow(np.squeeze(clean), cmap='gray')
        ax_clean.set_title('Clean Reference')
        ax_clean.axis('off')
        
        # Denoised results from each model
        for i, (denoised, model_name) in enumerate(zip(denoised_results, model_names)):
            ax = fig.add_subplot(gs[0, i + 1])
            ax.imshow(np.squeeze(denoised), cmap='gray')
            
            # Calculate metrics
            psnr_value = psnr(tf.convert_to_tensor([clean]), tf.convert_to_tensor([denoised])).numpy()
            ssim_value = ssim(tf.convert_to_tensor([clean]), tf.convert_to_tensor([denoised])).numpy()
            
            ax.set_title(f'{model_name}\nPSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}')
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f'model_comparison_sample_{idx}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def generate_magnified_view(noisy, denoised, clean, roi=None, magnification=2, save_path=None, title=None):
    """
    Generate magnified view of a region of interest to show denoising details.
    
    Args:
        noisy: Noisy input image
        denoised: Denoised image (model output)
        clean: Clean reference image (ground truth)
        roi: Region of interest as (x, y, width, height), if None, takes center
        magnification: Magnification factor
        save_path: Path to save the figure (if None, just displays it)
        title: Title for the figure
    """
    # Ensure images are in the right shape
    noisy = np.squeeze(noisy)
    denoised = np.squeeze(denoised)
    clean = np.squeeze(clean)
    
    # Default ROI to center if not specified
    if roi is None:
        h, w = clean.shape
        center_x, center_y = w // 2, h // 2
        roi_size = min(h, w) // 4
        roi = (center_x - roi_size//2, center_y - roi_size//2, roi_size, roi_size)
    
    # Extract ROI coordinates
    x, y, width, height = roi
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1])
    
    # Full images on top row
    ax_noisy = fig.add_subplot(gs[0, 0])
    ax_noisy.imshow(noisy, cmap='gray')
    ax_noisy.set_title('Noisy Input')
    ax_noisy.add_patch(plt.Rectangle((x, y), width, height, edgecolor='red', facecolor='none', linewidth=2))
    ax_noisy.axis('off')
    
    ax_denoised = fig.add_subplot(gs[0, 1])
    ax_denoised.imshow(denoised, cmap='gray')
    ax_denoised.set_title('Denoised Output')
    ax_denoised.add_patch(plt.Rectangle((x, y), width, height, edgecolor='red', facecolor='none', linewidth=2))
    ax_denoised.axis('off')
    
    ax_clean = fig.add_subplot(gs[0, 2])
    ax_clean.imshow(clean, cmap='gray')
    ax_clean.set_title('Clean Reference')
    ax_clean.add_patch(plt.Rectangle((x, y), width, height, edgecolor='red', facecolor='none', linewidth=2))
    ax_clean.axis('off')
    
    # Magnified ROIs on bottom row
    ax_noisy_roi = fig.add_subplot(gs[1, 0])
    ax_noisy_roi.imshow(noisy[y:y+height, x:x+width], cmap='gray', interpolation='nearest')
    ax_noisy_roi.set_title('Magnified Noisy')
    ax_noisy_roi.axis('off')
    
    ax_denoised_roi = fig.add_subplot(gs[1, 1])
    ax_denoised_roi.imshow(denoised[y:y+height, x:x+width], cmap='gray', interpolation='nearest')
    ax_denoised_roi.set_title('Magnified Denoised')
    ax_denoised_roi.axis('off')
    
    ax_clean_roi = fig.add_subplot(gs[1, 2])
    ax_clean_roi.imshow(clean[y:y+height, x:x+width], cmap='gray', interpolation='nearest')
    ax_clean_roi.set_title('Magnified Clean')
    ax_clean_roi.axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_noise_level_comparison(models, model_names, test_images, noise_levels, output_dir):
    """
    Compare model performance across different noise levels.
    
    Args:
        models: List of trained models for denoising
        model_names: List of names for the models
        test_images: Clean test images to which noise will be added
        noise_levels: List of noise standard deviations to test
        output_dir: Directory to save visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metrics storage
    psnr_values = {model_name: [] for model_name in model_names}
    ssim_values = {model_name: [] for model_name in model_names}
    
    # For each noise level
    for noise_std in noise_levels:
        # Add noise to the test images
        noisy_images = []
        for img in test_images:
            noise = np.random.normal(0, noise_std, img.shape)
            noisy = np.clip(img + noise, 0, 1)
            noisy_images.append(noisy)
        
        # Evaluate each model
        for i, (model, model_name) in enumerate(zip(models, model_names)):
            model_psnr = 0
            model_ssim = 0
            
            for noisy, clean in zip(noisy_images, test_images):
                # Get model prediction
                denoised = model.predict(np.expand_dims(noisy, axis=0))[0]
                
                # Calculate metrics
                model_psnr += psnr(tf.convert_to_tensor([clean]), tf.convert_to_tensor([denoised])).numpy()
                model_ssim += ssim(tf.convert_to_tensor([clean]), tf.convert_to_tensor([denoised])).numpy()
            
            # Average metrics
            model_psnr /= len(test_images)
            model_ssim /= len(test_images)
            
            # Store metrics
            psnr_values[model_name].append(model_psnr)
            ssim_values[model_name].append(model_ssim)
    
    # Plot PSNR vs noise level
    plt.figure(figsize=(12, 6))
    for model_name in model_names:
        plt.plot(noise_levels, psnr_values[model_name], marker='o', label=model_name)
    
    plt.title('PSNR vs Noise Level')
    plt.xlabel('Noise Standard Deviation')
    plt.ylabel('PSNR (dB)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'psnr_vs_noise.png'), dpi=300)
    plt.close()
    
    # Plot SSIM vs noise level
    plt.figure(figsize=(12, 6))
    for model_name in model_names:
        plt.plot(noise_levels, ssim_values[model_name], marker='o', label=model_name)
    
    plt.title('SSIM vs Noise Level')
    plt.xlabel('Noise Standard Deviation')
    plt.ylabel('SSIM')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ssim_vs_noise.png'), dpi=300)
    plt.close() 
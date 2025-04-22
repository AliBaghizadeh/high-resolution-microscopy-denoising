"""
Configuration parameters for STEM image denoising project.
"""

import os

# Data parameters
DATA_CONFIG = {
    # Directory paths
    'raw_data_dir': 'data/raw',
    'processed_data_dir': 'data/processed',
    'output_dir': 'output',
    
    # Patching parameters
    'patch_size': 256,
    'patch_stride': 128,
    
    # Dataset split ratios
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,  # Automatically calculated as 1 - (train_ratio + val_ratio)
    
    # Data augmentation parameters
    'use_augmentation': True,
    'rotation_range': 20,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'zoom_range': 0.1,
    'horizontal_flip': True,
    'vertical_flip': True,
    'brightness_range': (0.8, 1.2),  # Brightness adjustments
}

# Model parameters
MODEL_CONFIG = {
    # Basic UNet parameters
    'unet': {
        'input_size': (256, 256, 1),
        'filters': [64, 128, 256, 512, 1024],
        'batch_norm': True,
        'dropout_rates': [0.0, 0.0, 0.0, 0.0, 0.0],
        'use_transpose': True,
    },
    
    # Deep UNet parameters
    'deep_unet': {
        'input_size': (256, 256, 1),
        'filters': [64, 128, 256, 512, 1024, 2048],
        'batch_norm': True,
        'dropout_rates': [0.0, 0.0, 0.0, 0.2, 0.3, 0.4],
        'use_transpose': True,
    },
    
    # Residual UNet parameters
    'residual_unet': {
        'input_size': (256, 256, 1),
        'filters': [64, 128, 256, 512, 1024],
        'batch_norm': True,
        'dropout_rates': [0.0, 0.0, 0.0, 0.0, 0.0],
        'use_transpose': True,
    },
    
    # VGG16 UNet parameters
    'vgg16_unet': {
        'input_size': (256, 256, 1),
        'pretrained_weights': True,
        'freeze_encoder': True,
    },
    
    # ResNet50 UNet parameters
    'resnet50_unet': {
        'input_size': (256, 256, 1),
        'pretrained_weights': True,
        'freeze_encoder': True,
    },
    
    # EfficientNet UNet parameters
    'efficient_unet': {
        'input_size': (256, 256, 1),
        'efficient_net_version': 'B0',
        'pretrained_weights': True,
        'freeze_encoder': True,
    }
}

# Training parameters
TRAINING_CONFIG = {
    # Basic training parameters
    'batch_size': 8,
    'learning_rate': 1e-4,
    'epochs': 100,
    'early_stopping': True,
    'early_stopping_patience': 10,
    'use_mixed_precision': True,
    
    # Loss function parameters
    'loss_function': 'combined',  # Options: 'mse', 'mae', 'ssim', 'combined', 'perceptual'
    'ssim_weight': 0.84,  # Weight for SSIM loss in combined loss
    
    # Metrics to track
    'metrics': ['psnr', 'ssim', 'ms_ssim', 'edge_preservation'],
    
    # Optimizer parameters
    'optimizer': 'adam',  # Options: 'adam', 'sgd', 'rmsprop'
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'weight_decay': 1e-4,
    
    # Learning rate schedule
    'lr_schedule': 'reduce_on_plateau',  # Options: 'reduce_on_plateau', 'step_decay', 'cosine'
    'lr_reduce_factor': 0.5,
    'lr_reduce_patience': 5,
    'min_lr': 1e-6,
}

# Evaluation parameters
EVAL_CONFIG = {
    # Metrics to calculate
    'metrics': ['psnr', 'ssim', 'ms_ssim', 'edge_preservation', 'haarrpsi', 'fid'],
    
    # Visualization settings
    'generate_visualizations': True,
    'num_vis_samples': 5,
    'save_difference_maps': True,
    'save_edge_comparisons': True,
    
    # Noise analysis parameters
    'noise_analysis': {
        'enabled': True,
        'noise_levels': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    }
}

# Paths configuration
class PathConfig:
    def __init__(self, base_dir=None):
        # Set base directory
        if base_dir is None:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.base_dir = base_dir
        
        # Data directories
        self.raw_data_dir = os.path.join(self.base_dir, DATA_CONFIG['raw_data_dir'])
        self.processed_data_dir = os.path.join(self.base_dir, DATA_CONFIG['processed_data_dir'])
        
        # Output directories
        self.output_dir = os.path.join(self.base_dir, DATA_CONFIG['output_dir'])
        self.models_dir = os.path.join(self.output_dir, 'models')
        self.logs_dir = os.path.join(self.output_dir, 'logs')
        self.samples_dir = os.path.join(self.output_dir, 'samples')
        self.eval_dir = os.path.join(self.output_dir, 'evaluation')
        
        # Create directories if they don't exist
        self._create_directories()
    
    def _create_directories(self):
        """Create all required directories."""
        directories = [
            self.raw_data_dir,
            self.processed_data_dir,
            self.output_dir,
            self.models_dir,
            self.logs_dir,
            self.samples_dir,
            self.eval_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def get_model_path(self, model_name, timestamp=None):
        """Get path for saving/loading a model."""
        if timestamp:
            return os.path.join(self.models_dir, f"{model_name}_{timestamp}.h5")
        else:
            return os.path.join(self.models_dir, f"{model_name}.h5")
    
    def get_log_dir(self, model_name, timestamp=None):
        """Get directory for TensorBoard logs."""
        if timestamp:
            return os.path.join(self.logs_dir, f"{model_name}_{timestamp}")
        else:
            return os.path.join(self.logs_dir, model_name)

# Instantiate paths object
PATHS = PathConfig()

# Get configuration for a specific model
def get_model_config(model_name):
    """
    Get configuration parameters for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary of model configuration parameters
    """
    if model_name in MODEL_CONFIG:
        return MODEL_CONFIG[model_name]
    else:
        raise ValueError(f"Model '{model_name}' not found in configuration.")

# Get loss function based on configuration
def get_loss_function():
    """
    Get the configured loss function.
    
    Returns:
        Loss function based on configuration
    """
    from training.loss_functions import (combined_loss, ssim_loss, 
                                        perceptual_loss, l1_l2_loss)
    
    loss_type = TRAINING_CONFIG['loss_function']
    
    if loss_type == 'mse':
        return 'mse'
    elif loss_type == 'mae':
        return 'mae'
    elif loss_type == 'ssim':
        return ssim_loss
    elif loss_type == 'combined':
        return combined_loss(alpha=TRAINING_CONFIG['ssim_weight'])
    elif loss_type == 'perceptual':
        return perceptual_loss()
    elif loss_type == 'l1_l2':
        return l1_l2_loss(l1_ratio=0.5)
    else:
        raise ValueError(f"Unsupported loss function: {loss_type}") 
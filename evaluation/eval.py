import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

from ..models.unet import unet_model, deep_unet_model, residual_unet_model
from ..models.pretrained import vgg16_unet, resnet50_unet, efficient_unet
from ..training.metrics import psnr, ssim, ms_ssim, fid_score, edge_preservation, haarrpsi
from .visualize import (visualize_batch_results, display_comparison, 
                       display_edge_comparison, display_difference_map,
                       visualize_model_comparison)

class DenoiseEvaluator:
    """
    Class for evaluating and comparing STEM image denoising models.
    """
    
    def __init__(self, output_dir='./evaluation_results'):
        """
        Initialize the evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        self.models = {}
        self.results = {}
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, 'metrics'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
    
    def load_model(self, model_path, model_name=None):
        """
        Load a trained model for evaluation.
        
        Args:
            model_path: Path to the saved model
            model_name: Name to use for the model (defaults to filename)
        
        Returns:
            Loaded model
        """
        if model_name is None:
            model_name = os.path.basename(model_path).split('.')[0]
        
        print(f"Loading model '{model_name}' from {model_path}")
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects={
                'psnr_metric': psnr,
                'ssim_loss': ssim,
                'ms_ssim_loss': ms_ssim
            }
        )
        
        self.models[model_name] = model
        return model
    
    def create_model(self, model_type, model_name, weights_path=None, **kwargs):
        """
        Create a model from scratch with the given architecture.
        
        Args:
            model_type: Type of model to create ('unet', 'deep_unet', etc.)
            model_name: Name to use for the model
            weights_path: Optional path to model weights
            **kwargs: Additional arguments for model creation
            
        Returns:
            Created model
        """
        print(f"Creating model '{model_name}' of type {model_type}")
        
        if model_type == 'unet':
            model = unet_model(**kwargs)
        elif model_type == 'deep_unet':
            model = deep_unet_model(**kwargs)
        elif model_type == 'residual_unet':
            model = residual_unet_model(**kwargs)
        elif model_type == 'vgg16_unet':
            model = vgg16_unet(**kwargs)
        elif model_type == 'resnet50_unet':
            model = resnet50_unet(**kwargs)
        elif model_type == 'efficient_unet':
            model = efficient_unet(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load weights if provided
        if weights_path:
            model.load_weights(weights_path)
        
        self.models[model_name] = model
        return model
    
    def evaluate_model(self, model_name, test_data, batch_size=8, visualize=True, num_vis_samples=5):
        """
        Evaluate a model on test data.
        
        Args:
            model_name: Name of the model to evaluate
            test_data: Test dataset as a tuple of (x_test, y_test) or as a generator
            batch_size: Batch size for evaluation
            visualize: Whether to generate visualizations
            num_vis_samples: Number of samples to visualize
            
        Returns:
            Dictionary of evaluation metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Load or create it first.")
        
        model = self.models[model_name]
        print(f"Evaluating model '{model_name}'...")
        
        # Prepare visualization directory
        if visualize:
            vis_dir = os.path.join(self.output_dir, 'visualizations', model_name)
            os.makedirs(vis_dir, exist_ok=True)
        
        # Extract data from generator if needed
        if isinstance(test_data, tf.data.Dataset):
            # For TensorFlow datasets
            noisy_images, clean_images = [], []
            for x, y in test_data.take(batch_size):
                noisy_images.append(x)
                clean_images.append(y)
            x_test = tf.concat(noisy_images, axis=0)
            y_test = tf.concat(clean_images, axis=0)
        elif isinstance(test_data, tuple) and len(test_data) == 2:
            # For (x_test, y_test) tuples
            x_test, y_test = test_data
        else:
            raise ValueError("test_data must be a TensorFlow Dataset or a tuple of (x_test, y_test)")
        
        # Calculate predictions
        y_pred = model.predict(x_test, batch_size=batch_size)
        
        # Calculate metrics
        metrics = {
            'psnr': psnr(y_test, y_pred).numpy(),
            'ssim': ssim(y_test, y_pred).numpy(),
            'ms_ssim': ms_ssim(y_test, y_pred).numpy(),
            'edge_preservation': edge_preservation(y_test, y_pred).numpy(),
            'haarrpsi': haarrpsi(y_test, y_pred).numpy(),
        }
        
        try:
            metrics['fid'] = fid_score(y_test, y_pred)
        except Exception as e:
            print(f"Warning: Could not calculate FID score: {e}")
            metrics['fid'] = np.nan
        
        # Store results
        self.results[model_name] = metrics
        
        # Print results
        print(f"\nEvaluation results for '{model_name}':")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Generate visualizations
        if visualize and num_vis_samples > 0:
            num_vis_samples = min(num_vis_samples, len(x_test))
            for i in range(num_vis_samples):
                # Basic comparison
                display_comparison(
                    x_test[i], y_pred[i], y_test[i],
                    save_path=os.path.join(vis_dir, f'sample_{i}_comparison.png'),
                    title=f'{model_name} - Sample {i+1}'
                )
                
                # Edge comparison
                display_edge_comparison(
                    x_test[i], y_pred[i], y_test[i],
                    save_path=os.path.join(vis_dir, f'sample_{i}_edges.png'),
                    title=f'{model_name} - Edge Preservation - Sample {i+1}'
                )
                
                # Difference map
                display_difference_map(
                    y_pred[i], y_test[i],
                    save_path=os.path.join(vis_dir, f'sample_{i}_diff.png'),
                    title=f'{model_name} - Difference Map - Sample {i+1}'
                )
        
        return metrics
    
    def compare_models(self, model_names, test_data, batch_size=8, visualize=True, num_vis_samples=3):
        """
        Compare multiple models on the same test data.
        
        Args:
            model_names: List of model names to compare
            test_data: Test dataset as a tuple of (x_test, y_test) or as a generator
            batch_size: Batch size for evaluation
            visualize: Whether to generate comparison visualizations
            num_vis_samples: Number of samples to visualize
            
        Returns:
            Pandas DataFrame with comparison results
        """
        # Validate models
        for model_name in model_names:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found. Load or create it first.")
        
        # Evaluate each model if not already evaluated
        for model_name in model_names:
            if model_name not in self.results:
                self.evaluate_model(model_name, test_data, batch_size, False)
        
        # Extract data from generator if needed for visualization
        if visualize:
            if isinstance(test_data, tf.data.Dataset):
                # For TensorFlow datasets
                noisy_images, clean_images = [], []
                for x, y in test_data.take(batch_size):
                    noisy_images.append(x)
                    clean_images.append(y)
                x_test = tf.concat(noisy_images, axis=0)
                y_test = tf.concat(clean_images, axis=0)
            elif isinstance(test_data, tuple) and len(test_data) == 2:
                # For (x_test, y_test) tuples
                x_test, y_test = test_data
            else:
                raise ValueError("test_data must be a TensorFlow Dataset or a tuple of (x_test, y_test)")
            
            # Generate comparison visualizations
            comp_dir = os.path.join(self.output_dir, 'comparisons')
            os.makedirs(comp_dir, exist_ok=True)
            
            models = [self.models[name] for name in model_names]
            
            # Limit number of visualization samples
            num_vis_samples = min(num_vis_samples, len(x_test))
            
            # Generate and save comparison visualizations
            visualize_model_comparison(
                models, model_names, (x_test[:num_vis_samples], y_test[:num_vis_samples]), 
                comp_dir, range(num_vis_samples)
            )
        
        # Create comparison DataFrame
        comparison = pd.DataFrame([self.results[model_name] for model_name in model_names], 
                                 index=model_names)
        
        # Save comparison to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.output_dir, 'metrics', f'model_comparison_{timestamp}.csv')
        comparison.to_csv(csv_path)
        
        # Generate comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # PSNR comparison
        comparison['psnr'].plot(kind='bar', ax=axes[0, 0], color='skyblue')
        axes[0, 0].set_title('PSNR Comparison')
        axes[0, 0].set_ylabel('PSNR (dB)')
        axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # SSIM comparison
        comparison['ssim'].plot(kind='bar', ax=axes[0, 1], color='lightgreen')
        axes[0, 1].set_title('SSIM Comparison')
        axes[0, 1].set_ylabel('SSIM')
        axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Edge preservation comparison
        comparison['edge_preservation'].plot(kind='bar', ax=axes[1, 0], color='coral')
        axes[1, 0].set_title('Edge Preservation Comparison')
        axes[1, 0].set_ylabel('Edge Preservation Score')
        axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # HaarPSI comparison
        comparison['haarrpsi'].plot(kind='bar', ax=axes[1, 1], color='purple')
        axes[1, 1].set_title('HaarPSI Comparison')
        axes[1, 1].set_ylabel('HaarPSI Score')
        axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metrics', f'metric_comparison_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print comparison table
        print("\nModel Comparison:")
        print(comparison.to_string())
        
        return comparison
    
    def save_results(self, filename=None):
        """
        Save evaluation results to a file.
        
        Args:
            filename: Filename for the results (default: results_YYYYMMDD_HHMMSS.json)
        """
        if not self.results:
            print("No results to save.")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"
        
        # Convert numpy values to Python types for JSON serialization
        results_json = {}
        for model_name, metrics in self.results.items():
            results_json[model_name] = {k: float(v) for k, v in metrics.items()}
        
        # Save to file
        results_path = os.path.join(self.output_dir, 'metrics', filename)
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"Results saved to {results_path}")
    
    def load_results(self, filename):
        """
        Load evaluation results from a file.
        
        Args:
            filename: Path to the results file
        """
        # Load from file
        results_path = os.path.join(self.output_dir, 'metrics', filename)
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        
        print(f"Results loaded from {results_path}")
        
        # Print loaded results
        for model_name, metrics in self.results.items():
            print(f"\nResults for '{model_name}':")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
    
    def analyze_noise_levels(self, model_names, clean_images, noise_levels, output_dir=None):
        """
        Analyze model performance across different noise levels.
        
        Args:
            model_names: List of model names to compare
            clean_images: Dataset of clean images (no noise)
            noise_levels: List of noise standard deviations to test
            output_dir: Directory to save analysis results (default: 'noise_analysis')
        """
        # Validate models
        for model_name in model_names:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found. Load or create it first.")
        
        # Default output directory
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, 'noise_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results storage
        noise_results = {
            model_name: {
                'psnr': [],
                'ssim': [],
                'edge_preservation': []
            } for model_name in model_names
        }
        
        # For each noise level
        for noise_std in tqdm(noise_levels, desc="Processing noise levels"):
            # Create noisy images
            noisy_images = []
            for img in clean_images:
                noise = np.random.normal(0, noise_std, img.shape)
                noisy = np.clip(img + noise, 0, 1)
                noisy_images.append(noisy)
            
            noisy_batch = np.stack(noisy_images)
            clean_batch = np.stack(clean_images)
            
            # Test each model
            for model_name in model_names:
                model = self.models[model_name]
                
                # Generate predictions
                denoised_batch = model.predict(noisy_batch)
                
                # Calculate metrics
                noise_results[model_name]['psnr'].append(
                    psnr(clean_batch, denoised_batch).numpy()
                )
                noise_results[model_name]['ssim'].append(
                    ssim(clean_batch, denoised_batch).numpy()
                )
                noise_results[model_name]['edge_preservation'].append(
                    edge_preservation(clean_batch, denoised_batch).numpy()
                )
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(output_dir, f'noise_analysis_{timestamp}.json')
        
        # Convert results for JSON serialization
        json_results = {
            'noise_levels': noise_levels.tolist() if isinstance(noise_levels, np.ndarray) else noise_levels,
            'models': {}
        }
        
        for model_name, metrics in noise_results.items():
            json_results['models'][model_name] = {
                k: [float(val) for val in v] for k, v in metrics.items()
            }
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Generate plots
        for metric in ['psnr', 'ssim', 'edge_preservation']:
            plt.figure(figsize=(12, 6))
            
            for model_name in model_names:
                plt.plot(noise_levels, noise_results[model_name][metric], marker='o', label=model_name)
            
            plt.title(f'{metric.upper()} vs Noise Level')
            plt.xlabel('Noise Standard Deviation')
            plt.ylabel(metric.upper())
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f'{metric}_vs_noise_{timestamp}.png'), dpi=300)
            plt.close()
        
        return noise_results

def main():
    """
    Example usage of the DenoiseEvaluator.
    """
    # Initialize the evaluator
    evaluator = DenoiseEvaluator(output_dir='./evaluation_results')
    
    # Load models
    evaluator.load_model('models/unet_model.h5', 'UNet')
    evaluator.load_model('models/residual_unet_model.h5', 'ResidualUNet')
    evaluator.load_model('models/vgg16_unet_model.h5', 'VGG16-UNet')
    
    # Load test data (example)
    from ..data.dataset import DenoiseDataGenerator
    data_dir = "path/to/dataset"
    batch_size = 8
    
    data_generator = DenoiseDataGenerator(data_dir, batch_size=batch_size)
    test_gen = data_generator.get_generator('test')
    
    # Evaluate each model
    for model_name in evaluator.models:
        evaluator.evaluate_model(model_name, test_gen, visualize=True)
    
    # Compare models
    evaluator.compare_models(list(evaluator.models.keys()), test_gen)
    
    # Save results
    evaluator.save_results()
    
if __name__ == "__main__":
    main() 
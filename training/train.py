import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, TensorBoard
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
from models.unet import unet_model, deep_unet_model, residual_unet_model
from models.pretrained import vgg16_unet, resnet50_unet, efficient_unet
from training.loss_functions import combined_loss, ssim_loss, psnr_metric, ms_ssim_loss
from training.metrics import psnr, ssim, fid_score

class DenoiseTrainer:
    """
    Trainer class for STEM image denoising models.
    """
    
    def __init__(self, 
                 model_name='unet',
                 input_size=(256, 256, 1),
                 learning_rate=1e-4,
                 batch_size=8,
                 output_dir='./output',
                 use_mixed_precision=True):
        """
        Initialize the trainer.
        
        Args:
            model_name: Model architecture to use ('unet', 'deep_unet', 'residual_unet',
                       'vgg16_unet', 'resnet50_unet', 'efficient_unet')
            input_size: Input image dimensions (height, width, channels)
            learning_rate: Initial learning rate
            batch_size: Batch size for training
            output_dir: Directory to save models and logs
            use_mixed_precision: Whether to use mixed precision training
        """
        self.model_name = model_name
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.output_dir = output_dir
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
        
        # Set up mixed precision if requested
        if use_mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Create model
        self.model = self._create_model()
        
    def _create_model(self):
        """
        Create the denoising model based on the specified architecture.
        
        Returns:
            Keras Model object
        """
        # Initialize the model based on the selected architecture
        if self.model_name == 'unet':
            model = unet_model(input_size=self.input_size)
        elif self.model_name == 'deep_unet':
            model = deep_unet_model(input_size=self.input_size)
        elif self.model_name == 'residual_unet':
            model = residual_unet_model(input_size=self.input_size)
        elif self.model_name == 'vgg16_unet':
            model = vgg16_unet(input_size=self.input_size)
        elif self.model_name == 'resnet50_unet':
            model = resnet50_unet(input_size=self.input_size)
        elif self.model_name == 'efficient_unet':
            model = efficient_unet(input_size=self.input_size)
        else:
            raise ValueError(f"Unsupported model architecture: {self.model_name}")
            
        return model
    
    def compile(self, loss_fn=None, metrics=None):
        """
        Compile the model with optimizer, loss, and metrics.
        
        Args:
            loss_fn: Loss function to use (defaults to combined SSIM+MSE loss)
            metrics: List of metrics to track during training
        """
        if loss_fn is None:
            loss_fn = combined_loss(alpha=0.84)
        
        if metrics is None:
            metrics = [ssim_loss, psnr_metric]
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=loss_fn,
            metrics=metrics
        )
        
        # Print model summary
        self.model.summary()
        
    def _get_callbacks(self, checkpoint_path, early_stopping=True, patience=10):
        """
        Set up training callbacks.
        
        Args:
            checkpoint_path: Path to save model checkpoints
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping and LR reduction
            
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping callback
        if early_stopping:
            early_stopping_callback = EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping_callback)
        
        # Learning rate reduction callback
        reduce_lr_callback = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(reduce_lr_callback)
        
        # TensorBoard callback
        log_dir = os.path.join(
            self.output_dir, 
            'logs', 
            f"{self.model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        tensorboard_callback = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq='epoch'
        )
        callbacks.append(tensorboard_callback)
        
        return callbacks
    
    def train(self, train_generator, val_generator, epochs=100, early_stopping=True, patience=10):
        """
        Train the denoising model.
        
        Args:
            train_generator: Generator for training data
            val_generator: Generator for validation data
            epochs: Number of training epochs
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping
            
        Returns:
            Training history object
        """
        # Create checkpoint path
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_path = os.path.join(
            self.output_dir, 
            'models', 
            f"{self.model_name}_{timestamp}.h5"
        )
        
        # Get callbacks
        callbacks = self._get_callbacks(checkpoint_path, early_stopping, patience)
        
        # Train the model
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the final model
        final_model_path = os.path.join(
            self.output_dir, 
            'models', 
            f"{self.model_name}_final_{timestamp}.h5"
        )
        self.model.save(final_model_path)
        
        # Plot and save training history
        self._plot_training_history(history)
        
        return history
    
    def _plot_training_history(self, history):
        """
        Plot and save training history metrics.
        
        Args:
            history: Training history object from model.fit()
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Plot loss
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot PSNR
        plt.subplot(1, 2, 2)
        plt.plot(history.history['psnr_metric'], label='Training PSNR')
        plt.plot(history.history['val_psnr_metric'], label='Validation PSNR')
        plt.title('PSNR')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (dB)')
        plt.legend()
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(
            self.output_dir, 
            'logs', 
            f"{self.model_name}_history_{timestamp}.png"
        )
        plt.savefig(plot_path)
        plt.close()
        
    def evaluate(self, test_generator, visualize=True, num_samples=5):
        """
        Evaluate the model on test data.
        
        Args:
            test_generator: Generator for test data
            visualize: Whether to visualize and save sample results
            num_samples: Number of samples to visualize
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Evaluate the model
        results = self.model.evaluate(test_generator)
        metrics = {m.name: v for m, v in zip(self.model.metrics, results)}
        
        # Print results
        print("\nTest Results:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        # Visualize samples if requested
        if visualize:
            self._visualize_results(test_generator, num_samples)
            
        return metrics
    
    def _visualize_results(self, test_generator, num_samples=5):
        """
        Visualize and save sample denoising results.
        
        Args:
            test_generator: Generator for test data
            num_samples: Number of samples to visualize
        """
        # Get a batch of test data
        noisy_images, clean_images = next(iter(test_generator))
        
        # Limit to the number of samples to visualize
        num_samples = min(num_samples, len(noisy_images))
        noisy_images = noisy_images[:num_samples]
        clean_images = clean_images[:num_samples]
        
        # Generate predictions
        predictions = self.model.predict(noisy_images)
        
        # Create figure
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        
        for i in range(num_samples):
            # Display noisy image
            axes[i, 0].imshow(noisy_images[i].squeeze(), cmap='gray')
            axes[i, 0].set_title('Noisy')
            axes[i, 0].axis('off')
            
            # Display predicted (denoised) image
            axes[i, 1].imshow(predictions[i].squeeze(), cmap='gray')
            axes[i, 1].set_title('Denoised')
            axes[i, 1].axis('off')
            
            # Display clean (ground truth) image
            axes[i, 2].imshow(clean_images[i].squeeze(), cmap='gray')
            axes[i, 2].set_title('Ground Truth')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        # Save the visualization
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        vis_path = os.path.join(
            self.output_dir, 
            'samples', 
            f"{self.model_name}_samples_{timestamp}.png"
        )
        plt.savefig(vis_path)
        plt.close()
        
def main():
    """
    Main function to demonstrate model training.
    """
    from data.dataset import DenoiseDataGenerator
    
    # Initialize data generators
    data_dir = "/path/to/dataset_final"
    batch_size = 8
    
    data_generator = DenoiseDataGenerator(data_dir, batch_size=batch_size)
    train_gen = data_generator.get_generator('train')
    val_gen = data_generator.get_generator('val')
    test_gen = data_generator.get_generator('test')
    
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
    
    # Evaluate the model
    trainer.evaluate(test_gen)
    
if __name__ == "__main__":
    main() 
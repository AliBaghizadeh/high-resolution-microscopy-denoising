import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from scipy.linalg import sqrtm

def psnr(y_true, y_pred):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between ground truth and predicted images.
    
    Args:
        y_true: Ground truth images
        y_pred: Predicted images
    
    Returns:
        PSNR value
    """
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

def ssim(y_true, y_pred):
    """
    Calculate Structural Similarity Index (SSIM) between ground truth and predicted images.
    
    Args:
        y_true: Ground truth images
        y_pred: Predicted images
    
    Returns:
        SSIM value
    """
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def ms_ssim(y_true, y_pred):
    """
    Calculate Multi-Scale Structural Similarity Index (MS-SSIM) between ground truth and predicted images.
    
    Args:
        y_true: Ground truth images
        y_pred: Predicted images
    
    Returns:
        MS-SSIM value
    """
    return tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, max_val=1.0))

def rmse(y_true, y_pred):
    """
    Calculate Root Mean Square Error (RMSE) between ground truth and predicted images.
    
    Args:
        y_true: Ground truth images
        y_pred: Predicted images
    
    Returns:
        RMSE value
    """
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

def fid_score(real_images, generated_images, batch_size=8):
    """
    Calculate Fréchet Inception Distance (FID) between real and generated images.
    FID measures the similarity between two datasets of images.
    
    Args:
        real_images: Dataset of real images
        generated_images: Dataset of generated images
        batch_size: Batch size for processing
    
    Returns:
        FID score (lower is better)
    """
    # Load Inception model for feature extraction
    inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    
    # Helper function to extract features
    def extract_features(images):
        # Resize images to 299x299 as required by InceptionV3
        images_resized = tf.image.resize(images, (299, 299))
        
        # Convert from grayscale to RGB if needed
        if images_resized.shape[-1] == 1:
            images_rgb = tf.tile(images_resized, [1, 1, 1, 3])
        else:
            images_rgb = images_resized
        
        # Scale from [0,1] to [0,255]
        images_scaled = images_rgb * 255.0
        
        # Preprocess images for inception
        images_preprocessed = preprocess_input(images_scaled)
        
        # Extract features in batches
        features_list = []
        for i in range(0, images_preprocessed.shape[0], batch_size):
            batch = images_preprocessed[i:i+batch_size]
            features = inception_model.predict(batch)
            features_list.append(features)
        
        return np.vstack(features_list)
    
    # Extract features from both sets of images
    real_features = extract_features(real_images)
    generated_features = extract_features(generated_images)
    
    # Calculate mean and covariance for both feature sets
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_generated = np.mean(generated_features, axis=0)
    sigma_generated = np.cov(generated_features, rowvar=False)
    
    # Calculate FID
    squared_diff = np.sum((mu_real - mu_generated) ** 2)
    covmean = sqrtm(sigma_real.dot(sigma_generated))
    
    # Check if covmean contains complex values
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = squared_diff + np.trace(sigma_real + sigma_generated - 2 * covmean)
    
    return fid

def edge_preservation(y_true, y_pred):
    """
    Calculate edge preservation metric to evaluate how well edges are preserved in denoised images.
    
    Args:
        y_true: Ground truth images
        y_pred: Predicted images
    
    Returns:
        Edge preservation score
    """
    # Calculate gradients for ground truth
    true_grad_x = tf.abs(tf.image.sobel_edges(y_true)[:,:,:,:,0])
    true_grad_y = tf.abs(tf.image.sobel_edges(y_true)[:,:,:,:,1])
    
    # Calculate gradients for predicted images
    pred_grad_x = tf.abs(tf.image.sobel_edges(y_pred)[:,:,:,:,0])
    pred_grad_y = tf.abs(tf.image.sobel_edges(y_pred)[:,:,:,:,1])
    
    # Calculate correlation between true and predicted gradients
    correlation_x = tf.reduce_mean(
        tf.image.ssim(true_grad_x, pred_grad_x, max_val=tf.reduce_max(true_grad_x))
    )
    correlation_y = tf.reduce_mean(
        tf.image.ssim(true_grad_y, pred_grad_y, max_val=tf.reduce_max(true_grad_y))
    )
    
    # Average the correlations
    return (correlation_x + correlation_y) / 2.0

def haarrpsi(y_true, y_pred, batch_size=8):
    """
    Calculate Haar Wavelet-Based Perceptual Similarity Index (HaarPSI).
    
    This is a simplified implementation of HaarPSI to work in TensorFlow.
    For the original algorithm, see the paper:
    "HaarPSI: A simple and yet accurate full-reference perceptual image quality assessor"
    
    Args:
        y_true: Ground truth images
        y_pred: Predicted images
        batch_size: Batch size for processing
    
    Returns:
        HaarPSI score (higher is better)
    """
    # Define Haar wavelet filters
    haar_filter_x = tf.constant([[[[-1, 1], [-1, 1]]]], dtype=tf.float32)
    haar_filter_y = tf.constant([[[[-1, -1], [1, 1]]]], dtype=tf.float32)
    
    def process_batch(true_batch, pred_batch):
        # Apply Haar wavelet transform
        true_x = tf.nn.conv2d(true_batch, haar_filter_x, strides=[1, 2, 2, 1], padding='VALID')
        true_y = tf.nn.conv2d(true_batch, haar_filter_y, strides=[1, 2, 2, 1], padding='VALID')
        
        pred_x = tf.nn.conv2d(pred_batch, haar_filter_x, strides=[1, 2, 2, 1], padding='VALID')
        pred_y = tf.nn.conv2d(pred_batch, haar_filter_y, strides=[1, 2, 2, 1], padding='VALID')
        
        # Calculate magnitudes
        true_mag = tf.sqrt(tf.square(true_x) + tf.square(true_y) + 1e-8)
        pred_mag = tf.sqrt(tf.square(pred_x) + tf.square(pred_y) + 1e-8)
        
        # Calculate local similarity
        c = 0.001  # Small constant to avoid division by zero
        similarity = (2 * true_mag * pred_mag + c) / (tf.square(true_mag) + tf.square(pred_mag) + c)
        
        # Weight the similarity by the magnitude (important features have higher magnitude)
        weights = tf.maximum(true_mag, pred_mag)
        weighted_similarity = similarity * weights
        
        # Calculate HaarPSI
        score = tf.reduce_sum(weighted_similarity) / (tf.reduce_sum(weights) + 1e-8)
        
        return score
    
    # Process images in batches
    scores = []
    for i in range(0, y_true.shape[0], batch_size):
        true_batch = y_true[i:i+batch_size]
        pred_batch = y_pred[i:i+batch_size]
        score = process_batch(true_batch, pred_batch)
        scores.append(score)
    
    return tf.reduce_mean(scores)

class CustomMetrics(tf.keras.callbacks.Callback):
    """
    Custom callback to calculate additional metrics during training.
    """
    
    def __init__(self, validation_data, metrics_list=['edge_preservation', 'haarrpsi']):
        """
        Initialize the callback.
        
        Args:
            validation_data: Tuple of (x_val, y_val) for validation data
            metrics_list: List of metric names to calculate
        """
        super().__init__()
        self.validation_data = validation_data
        self.metrics_list = metrics_list
        self.metrics_values = {metric: [] for metric in metrics_list}
    
    def on_epoch_end(self, epoch, logs=None):
        """
        Calculate metrics at the end of each epoch.
        
        Args:
            epoch: Current epoch
            logs: Dictionary to which we'll add metrics
        """
        x_val, y_val = self.validation_data
        y_pred = self.model.predict(x_val)
        
        logs = logs or {}
        
        for metric in self.metrics_list:
            if metric == 'edge_preservation':
                value = edge_preservation(y_val, y_pred).numpy()
            elif metric == 'haarrpsi':
                value = haarrpsi(y_val, y_pred).numpy()
            else:
                continue
            
            self.metrics_values[metric].append(value)
            logs[metric] = value
            
            print(f" - val_{metric}: {value:.4f}") 
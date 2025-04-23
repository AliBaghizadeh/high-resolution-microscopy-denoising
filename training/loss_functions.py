import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

# Initialize TensorFlow Metrics for Logging Losses
mse_metric = tf.keras.metrics.Mean(name="mse_loss")
fft_metric = tf.keras.metrics.Mean(name="fft_loss")
poisson_metric = tf.keras.metrics.Mean(name="poisson_loss")
ssim_metric = tf.keras.metrics.Mean(name="ssim_loss")
tv_metric = tf.keras.metrics.Mean(name="tv_loss")
mean_intensity_metric = tf.keras.metrics.Mean(name="mean_intensity_loss")

def ssim_loss(y_true, y_pred):
    """
    Structural Similarity Index (SSIM) loss.
    
    Args:
        y_true: Ground truth images
        y_pred: Predicted images
    
    Returns:
        SSIM loss value
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Only convert if the image has 3 channels (RGB)
    if y_true.shape[-1] == 3:
        y_true = tf.image.rgb_to_grayscale(y_true)
    if y_pred.shape[-1] == 3:
        y_pred = tf.image.rgb_to_grayscale(y_pred)

    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def psnr_metric(y_true, y_pred):
    """
    Peak Signal-to-Noise Ratio (PSNR) metric.
    
    Args:
        y_true: Ground truth images
        y_pred: Predicted images
    
    Returns:
        PSNR value
    """
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

# Standard Mean Squared Error (MSE) Loss
def mse_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.square(y_true - y_pred))

# FFT-Based Loss (Frequency Domain)
def fft_loss(y_true, y_pred):
    """
    Compares the frequency domain of the true and predicted images using FFT.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    fft_true = tf.signal.fft2d(tf.cast(y_true, tf.complex64))  # Convert to complex
    fft_pred = tf.signal.fft2d(tf.cast(y_pred, tf.complex64))  # Convert to complex

    # Take absolute values to remove imaginary components
    log_fft_true = tf.math.log(1.0 + tf.abs(fft_true))
    log_fft_pred = tf.math.log(1.0 + tf.abs(fft_pred))

    loss = tf.reduce_mean(tf.square(log_fft_true - log_fft_pred))  # Compute Mean Squared Error in log-space
    return loss

def normalized_tv_loss(y_true, y_pred):
    """ TV loss normalized to prevent large values dominating the loss. Accepts y_true for compatibility. """
    tv = tf.reduce_mean(tf.image.total_variation(tf.cast(y_pred, tf.float32)))  # Work in float32
    image_size = tf.cast(tf.size(y_pred), tf.float32)  # Total number of pixels as float32
    return tv / image_size

def mean_intensity_loss(y_true, y_pred):
    """ Penalize deviation from mean intensity of the target image """
    mean_true = tf.reduce_mean(tf.cast(y_true, tf.float32))  # Cast to float32
    mean_pred = tf.reduce_mean(tf.cast(y_pred, tf.float32))  # Cast to float32
    return tf.abs(mean_true - mean_pred)  # Always returns float32

# Poisson-Based Loss (for Noise Modeling)
def poisson_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    epsilon = 1e-8  # Prevent log(0)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
    return tf.reduce_mean(y_pred - y_true * tf.math.log(y_pred))

def combined_loss(y_true, y_pred, alpha, beta, gamma, delta, epsilon, mu):
    """
    Combined loss function: MSE + FFT Loss + Poisson Loss + SSIM Loss + Normalized TV + Mean Intensity Loss.

    Tracks individual loss components separately for better visualization.
    """

    # Compute each loss term only if its weight is non-zero
    mse = mse_loss(y_true, y_pred) if alpha > 0 else 0
    fft = fft_loss(y_true, y_pred) if beta > 0 else 0
    poisson = poisson_loss(y_true, y_pred) if gamma > 0 else 0
    ssim = ssim_loss(y_true, y_pred) if delta > 0 else 0
    tv = normalized_tv_loss(y_true, y_pred) if epsilon > 0 else 0
    mean_intensity = mean_intensity_loss(y_true, y_pred) if mu > 0 else 0  # Intensity preservation

    # Update metric tracking only for non-zero components
    if alpha > 0:
        mse_metric.update_state(mse)
    if beta > 0:
        fft_metric.update_state(fft)
    if gamma > 0:
        poisson_metric.update_state(poisson)
    if delta > 0:
        ssim_metric.update_state(ssim)
    if epsilon > 0:
        tv_metric.update_state(tv)
    if mu > 0:
        mean_intensity_metric.update_state(mean_intensity)

    # Total weighted sum (now includes mean intensity)
    total_loss = (alpha * mse + beta * fft + gamma * poisson + delta * ssim
                  + epsilon * tv + mu * mean_intensity)

    return total_loss

# Loss function factory for compatibility with Keras (added for Colab compatibility)
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
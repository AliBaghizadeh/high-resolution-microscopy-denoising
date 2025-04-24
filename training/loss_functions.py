
import tensorflow as tf

# Initialize TensorFlow Metrics for Logging Losses
mse_metric = tf.keras.metrics.Mean(name='mse_loss')
fft_metric = tf.keras.metrics.Mean(name='fft_loss')
poisson_metric = tf.keras.metrics.Mean(name='poisson_loss')
ssim_metric = tf.keras.metrics.Mean(name='ssim_loss')
tv_metric = tf.keras.metrics.Mean(name='tv_loss')
mean_intensity_metric = tf.keras.metrics.Mean(name='mean_intensity_loss')

def ssim_loss(y_true, y_pred):
    '''
    Structural Similarity Index Measure (SSIM) loss.

    Args:
        y_true (tf.Tensor): Ground truth images
        y_pred (tf.Tensor): Predicted images

    Returns:
        tf.Tensor: 1 - mean SSIM score across the batch
    '''
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def tv_loss(y_true, y_pred):
    '''
    Total Variation (TV) loss, non-normalized.

    Args:
        y_true (tf.Tensor): Unused
        y_pred (tf.Tensor): Prediction

    Returns:
        tf.Tensor: Scalar TV loss
    '''
    y_pred = (y_pred - tf.reduce_min(y_pred)) / (tf.reduce_max(y_pred) - tf.reduce_min(y_pred) + 1e-8)
    return tf.reduce_mean(tf.image.total_variation(y_pred))

def normalized_tv_loss(y_true, y_pred):
    '''
    Normalized Total Variation loss by image size.

    Args:
        y_true (tf.Tensor): Unused
        y_pred (tf.Tensor): Prediction

    Returns:
        tf.Tensor: Normalized TV loss
    '''
    tv = tf.reduce_mean(tf.image.total_variation(tf.cast(y_pred, tf.float32)))
    image_size = tf.cast(tf.size(y_pred), tf.float32)
    return tv / image_size

def mse_loss(y_true, y_pred):
    '''
    Mean Squared Error loss.

    Args:
        y_true (tf.Tensor): Ground truth
        y_pred (tf.Tensor): Prediction

    Returns:
        tf.Tensor: Scalar MSE loss
    '''
    return tf.reduce_mean(tf.square(tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32)))

def fft_loss(y_true, y_pred):
    '''
    Log-scaled FFT loss comparing frequency domain differences.

    Args:
        y_true (tf.Tensor): Ground truth
        y_pred (tf.Tensor): Prediction

    Returns:
        tf.Tensor: Scalar FFT loss
    '''
    fft_true = tf.signal.fft2d(tf.cast(y_true, tf.complex64))
    fft_pred = tf.signal.fft2d(tf.cast(y_pred, tf.complex64))
    log_fft_true = tf.math.log(1.0 + tf.abs(fft_true))
    log_fft_pred = tf.math.log(1.0 + tf.abs(fft_pred))
    return tf.reduce_mean(tf.square(log_fft_true - log_fft_pred))

def poisson_loss(y_true, y_pred):
    '''
    Poisson loss for intensity modeling.

    Args:
        y_true (tf.Tensor): Ground truth
        y_pred (tf.Tensor): Prediction

    Returns:
        tf.Tensor: Scalar Poisson loss
    '''
    y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)
    return tf.reduce_mean(y_pred - y_true * tf.math.log(y_pred))

def mean_intensity_loss(y_true, y_pred):
    '''
    Penalizes deviation from the average intensity.

    Args:
        y_true (tf.Tensor): Ground truth
        y_pred (tf.Tensor): Prediction

    Returns:
        tf.Tensor: Mean intensity difference
    '''
    return tf.abs(tf.reduce_mean(y_true) - tf.reduce_mean(y_pred))

def combined_loss(y_true, y_pred, alpha, beta, gamma, delta, epsilon, mu):
    '''
    Weighted combination of multiple loss functions.

    Args:
        y_true (tf.Tensor): Ground truth images
        y_pred (tf.Tensor): Predicted images
        alpha (float): Weight for MSE
        beta (float): Weight for FFT loss
        gamma (float): Weight for Poisson loss
        delta (float): Weight for SSIM loss
        epsilon (float): Weight for TV loss
        mu (float): Weight for Mean Intensity loss

    Returns:
        tf.Tensor: Final weighted loss
    '''
    mse = mse_loss(y_true, y_pred) if alpha > 0 else 0
    fft = fft_loss(y_true, y_pred) if beta > 0 else 0
    poisson = poisson_loss(y_true, y_pred) if gamma > 0 else 0
    ssim = ssim_loss(y_true, y_pred) if delta > 0 else 0
    tv = normalized_tv_loss(y_true, y_pred) if epsilon > 0 else 0
    mean_intensity = mean_intensity_loss(y_true, y_pred) if mu > 0 else 0

    if alpha > 0: mse_metric.update_state(mse)
    if beta > 0: fft_metric.update_state(fft)
    if gamma > 0: poisson_metric.update_state(poisson)
    if delta > 0: ssim_metric.update_state(ssim)
    if epsilon > 0: tv_metric.update_state(tv)
    if mu > 0: mean_intensity_metric.update_state(mean_intensity)

    return alpha*mse + beta*fft + gamma*poisson + delta*ssim + epsilon*tv + mu*mean_intensity

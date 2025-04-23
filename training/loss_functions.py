import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, binary_crossentropy
from tensorflow.keras.applications import VGG16
import numpy as np

def ssim_loss(y_true, y_pred):
    """
    Structural Similarity Index (SSIM) loss.
    
    Args:
        y_true: Ground truth images
        y_pred: Predicted images
    
    Returns:
        SSIM loss value
    """
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

def ms_ssim_loss(y_true, y_pred):
    """
    Multi-Scale Structural Similarity Index (MS-SSIM) loss.
    
    Args:
        y_true: Ground truth images
        y_pred: Predicted images
    
    Returns:
        MS-SSIM loss value
    """
    return 1 - tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, max_val=1.0))

def combined_loss(alpha=0.84):
    """
    Combined loss with weighted MSE and SSIM components.
    
    Args:
        alpha: Weight for SSIM loss (1-alpha for MSE)
    
    Returns:
        Combined loss function
    """
    def loss(y_true, y_pred):
        mse_loss = MeanSquaredError()(y_true, y_pred)
        ssim_l = ssim_loss(y_true, y_pred)
        return alpha * ssim_l + (1 - alpha) * mse_loss
    
    return loss

def l1_l2_loss(l1_ratio=0.5):
    """
    Combination of L1 and L2 losses.
    
    Args:
        l1_ratio: Weight for L1 loss (1-l1_ratio for L2)
    
    Returns:
        Combined L1-L2 loss function
    """
    def loss(y_true, y_pred):
        l1_loss = MeanAbsoluteError()(y_true, y_pred)
        l2_loss = MeanSquaredError()(y_true, y_pred)
        return l1_ratio * l1_loss + (1 - l1_ratio) * l2_loss
    
    return loss

def perceptual_loss(input_shape=(256, 256, 1)):
    """
    Perceptual loss using VGG16 features.
    
    Args:
        input_shape: Input image shape
    
    Returns:
        Perceptual loss function
    """
    # Create VGG model for feature extraction
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
    
    # Use specific layers for feature comparison
    layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']
    layer_outputs = [vgg.get_layer(name).output for name in layer_names]
    
    # Create feature extraction model
    feature_model = tf.keras.Model(inputs=vgg.input, outputs=layer_outputs)
    feature_model.trainable = False
    
    def loss(y_true, y_pred):
        # Convert grayscale to RGB by repeating channels
        y_true_rgb = tf.tile(y_true, [1, 1, 1, 3])
        y_pred_rgb = tf.tile(y_pred, [1, 1, 1, 3])
        
        # Extract features
        true_features = feature_model(y_true_rgb)
        pred_features = feature_model(y_pred_rgb)
        
        # Calculate MSE between features
        loss_value = 0
        for true_feat, pred_feat in zip(true_features, pred_features):
            loss_value += MeanSquaredError()(true_feat, pred_feat)
        
        return loss_value
    
    return loss

def ssim_perceptual_loss(alpha=0.5, input_shape=(256, 256, 1)):
    """
    Combined SSIM and perceptual loss.
    
    Args:
        alpha: Weight for SSIM loss
        input_shape: Input image shape
    
    Returns:
        Combined loss function
    """
    perceptual = perceptual_loss(input_shape)
    
    def loss(y_true, y_pred):
        p_loss = perceptual(y_true, y_pred)
        s_loss = ssim_loss(y_true, y_pred)
        return alpha * s_loss + (1 - alpha) * p_loss
    
    return loss

def content_aware_loss(edge_weight=2.0):
    """
    Content-aware loss with higher weights for edges.
    
    Args:
        edge_weight: Weight multiplier for edge regions
    
    Returns:
        Content-aware loss function
    """
    def loss(y_true, y_pred):
        # Detect edges in ground truth using Sobel operator
        edges_x = tf.abs(tf.image.sobel_edges(y_true)[:,:,:,:,0])
        edges_y = tf.abs(tf.image.sobel_edges(y_true)[:,:,:,:,1])
        edge_mask = tf.clip_by_value(edges_x + edges_y, 0, 1)
        
        # Create weighted mask (edges have higher weight)
        weights = 1.0 + (edge_weight - 1.0) * edge_mask
        
        # Calculate weighted MSE
        squared_diff = tf.square(y_true - y_pred)
        weighted_squared_diff = weights * squared_diff
        
        return tf.reduce_mean(weighted_squared_diff)
    
    return loss

def gradient_loss():
    """
    Gradient-based loss that compares image gradients.
    
    Returns:
        Gradient loss function
    """
    def loss(y_true, y_pred):
        # Calculate gradients
        true_grad_x = tf.image.sobel_edges(y_true)[:,:,:,:,0]
        true_grad_y = tf.image.sobel_edges(y_true)[:,:,:,:,1]
        pred_grad_x = tf.image.sobel_edges(y_pred)[:,:,:,:,0]
        pred_grad_y = tf.image.sobel_edges(y_pred)[:,:,:,:,1]
        
        # Calculate MSE for gradients
        grad_x_loss = MeanSquaredError()(true_grad_x, pred_grad_x)
        grad_y_loss = MeanSquaredError()(true_grad_y, pred_grad_y)
        
        return grad_x_loss + grad_y_loss
    
    return loss

def total_variation_loss(y_true, y_pred):
    """
    Total Variation loss for encouraging spatial smoothness.
    
    Args:
        y_true: Ground truth images (not used in this loss)
        y_pred: Predicted images
    
    Returns:
        Total variation loss value
    """
    return tf.reduce_mean(tf.image.total_variation(y_pred))

def boundary_mse_loss(boundary_weight=2.0):
    """
    MSE loss with higher weights for object boundaries.
    
    Args:
        boundary_weight: Weight for boundary regions
    
    Returns:
        Boundary-weighted MSE loss function
    """
    def loss(y_true, y_pred):
        # Create boundary mask using Laplacian
        laplacian = tf.abs(tf.nn.depthwise_conv2d(
            y_true, 
            tf.reshape(tf.constant([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=tf.float32), [3, 3, 1, 1]), 
            strides=[1, 1, 1, 1], 
            padding='SAME'
        ))
        boundary_mask = tf.clip_by_value(laplacian, 0, 1)
        
        # Create weights
        weights = 1.0 + (boundary_weight - 1.0) * boundary_mask
        
        # Calculate weighted MSE
        squared_diff = tf.square(y_true - y_pred)
        weighted_loss = weights * squared_diff
        
        return tf.reduce_mean(weighted_loss)
    
    return loss

def multiscale_gradient_loss(scales=[1, 2, 4]):
    """
    Multi-scale gradient loss comparing gradients at different scales.
    
    Args:
        scales: List of scales to compute gradients
    
    Returns:
        Multi-scale gradient loss function
    """
    def loss(y_true, y_pred):
        loss_value = 0
        
        for scale in scales:
            if scale > 1:
                # Downscale images
                true_scaled = tf.nn.avg_pool2d(y_true, ksize=scale, strides=scale, padding='VALID')
                pred_scaled = tf.nn.avg_pool2d(y_pred, ksize=scale, strides=scale, padding='VALID')
            else:
                true_scaled = y_true
                pred_scaled = y_pred
            
            # Calculate gradients
            true_grad_x = tf.image.sobel_edges(true_scaled)[:,:,:,:,0]
            true_grad_y = tf.image.sobel_edges(true_scaled)[:,:,:,:,1]
            pred_grad_x = tf.image.sobel_edges(pred_scaled)[:,:,:,:,0]
            pred_grad_y = tf.image.sobel_edges(pred_scaled)[:,:,:,:,1]
            
            # Calculate MSE for gradients
            grad_x_loss = MeanSquaredError()(true_grad_x, pred_grad_x)
            grad_y_loss = MeanSquaredError()(true_grad_y, pred_grad_y)
            
            loss_value += grad_x_loss + grad_y_loss
            
        return loss_value / len(scales)
    
    return loss

def create_combined_loss_fn(alpha=0.84, beta=0.06, gamma=0.1, delta=0, epsilon=0, mu=0):
    """
    Creates a Keras-compatible wrapper for the multi-parameter combined_loss function.
    
    Args:
        alpha: Weight for MSE loss
        beta: Weight for FFT loss
        gamma: Weight for Poisson loss
        delta: Weight for SSIM loss
        epsilon: Weight for TV loss
        mu: Weight for Mean Intensity loss
        
    Returns:
        A function that takes only y_true and y_pred (Keras-compatible)
    """
    def loss_fn(y_true, y_pred):
        return combined_loss(y_true, y_pred, alpha, beta, gamma, delta, epsilon, mu)
    
    return loss_fn 

import os
import numpy as np
from PIL import Image

def save_prediction_image(tensor, save_path):
    '''
    Saves a single image tensor (batch size 1) as PNG.

    Args:
        tensor (np.ndarray or tf.Tensor): Image tensor with values in [0, 1]
        save_path (str): Path to save the image
    '''
    if hasattr(tensor, 'numpy'):
        tensor = tensor.numpy()
    img = np.squeeze(tensor)  # remove batch and channel dims
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(save_path)

def ensure_dir(path):
    '''
    Ensure a directory exists; if not, create it.

    Args:
        path (str): Path to check/create.
    '''
    if not os.path.exists(path):
        os.makedirs(path)

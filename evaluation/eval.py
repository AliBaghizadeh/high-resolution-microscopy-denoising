import os
from models.utils import save_prediction_image
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def evaluate_model(model, dataset, save_dir=None, prefix="eval"):
    '''
    Evaluates a model on a given dataset and optionally saves the results.

    Args:
        model (tf.keras.Model): Trained model to evaluate.
        dataset (tf.data.Dataset): Dataset containing (noisy, clean) pairs.
        save_dir (str): If provided, predictions will be saved here.
        prefix (str): Prefix for saved file names.

    Returns:
        list: PSNR and SSIM for each prediction
    '''
    psnrs, ssims = [], []

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i, (noisy, clean) in enumerate(tqdm(dataset, desc="Evaluating")):
        pred = model.predict(noisy)
        psnr = tf.image.psnr(clean, pred, max_val=1.0).numpy().mean()
        ssim = tf.image.ssim(clean, pred, max_val=1.0).numpy().mean()

        psnrs.append(psnr)
        ssims.append(ssim)

        if save_dir:
            save_prediction_image(pred[0], os.path.join(save_dir, f"{prefix}_pred_{i:03d}.png"))
            save_prediction_image(noisy[0], os.path.join(save_dir, f"{prefix}_noisy_{i:03d}.png"))
            save_prediction_image(clean[0], os.path.join(save_dir, f"{prefix}_clean_{i:03d}.png"))

    return psnrs, ssims

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from models.utils import save_prediction_image

def check_dataset(dataset, num_batches=1):
    '''
    Displays a few batches of images from a dataset.

    Args:
        dataset (tf.data.Dataset): The dataset to inspect.
        num_batches (int): Number of batches to display.
    '''
    for i, (noisy, clean) in enumerate(dataset.take(num_batches)):
        fig, axs = plt.subplots(noisy.shape[0], 2, figsize=(5, 2 * noisy.shape[0]))
        for j in range(noisy.shape[0]):
            axs[j, 0].imshow(tf.squeeze(noisy[j]), cmap='gray')
            axs[j, 0].set_title("Noisy")
            axs[j, 1].imshow(tf.squeeze(clean[j]), cmap='gray')
            axs[j, 1].set_title("Clean")
        plt.tight_layout()
        plt.show()

def visualize_predictions(model, dataset, n_samples=3):
    '''
    Shows predictions for a few samples from the dataset.

    Args:
        model (tf.keras.Model): The trained model.
        dataset (tf.data.Dataset): Dataset of (noisy, clean) pairs.
        n_samples (int): Number of samples to visualize.
    '''
    for i, (noisy, clean) in enumerate(dataset.take(n_samples)):
        pred = model.predict(noisy)
        for j in range(noisy.shape[0]):
            fig, axs = plt.subplots(1, 3, figsize=(10, 3))
            axs[0].imshow(tf.squeeze(noisy[j]), cmap='gray')
            axs[0].set_title("Noisy")
            axs[1].imshow(tf.squeeze(clean[j]), cmap='gray')
            axs[1].set_title("Ground Truth")
            axs[2].imshow(tf.squeeze(pred[j]), cmap='gray')
            axs[2].set_title("Denoised")
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            plt.show()

def log_experiment_results(history, save_path="experiment_log.csv"):
    '''
    Saves the final metrics of a training session to a CSV file.

    Args:
        history (History object): Training history returned from model.fit().
        save_path (str): Path to the CSV file.
    '''
    final_epoch = {k: v[-1] for k, v in history.history.items()}
    df = pd.DataFrame([final_epoch])
    df.to_csv(save_path, index=False)

def save_training_history(history, filename='history_full.csv'):
    '''
    Save the full training history (all epochs) to CSV.

    Args:
        history (History object): Keras training history.
        filename (str): Destination CSV filename.
    '''
    df = pd.DataFrame(history.history)
    df.to_csv(filename, index=False)

def plot_training_metrics(history):
    '''
    Plots loss and metric curves from Keras training history.

    Args:
        history (History object): Training history.
    '''
    metrics = list(history.history.keys())
    plt.figure(figsize=(10, 6))
    for m in metrics:
        plt.plot(history.history[m], label=m)
    plt.title("Training Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_weighted_loss_components(metric_logs, weights):
    '''
    Plots weighted loss component curves.

    Args:
        metric_logs (dict): Dictionary of lists (each metric's loss per epoch).
        weights (dict): Dictionary of weights per loss component.
    '''
    plt.figure(figsize=(10, 6))
    for name, values in metric_logs.items():
        weighted = [v * weights.get(name, 1) for v in values]
        plt.plot(weighted, label=f"{name} (weighted)")
    plt.title("Weighted Loss Components")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_model(model, dataset, save_dir="results", prefix="sample"):
    '''
    Generates predictions, computes PSNR/SSIM, saves images.

    Args:
        model (tf.keras.Model): Trained model.
        dataset (tf.data.Dataset): Dataset for evaluation.
        save_dir (str): Directory to save predictions.
        prefix (str): Prefix for saved files.

    Returns:
        list: PSNR and SSIM scores.
    '''
    os.makedirs(save_dir, exist_ok=True)
    psnr_list, ssim_list = [], []

    for i, (noisy, clean) in enumerate(tqdm(dataset, desc="Evaluating")):
        pred = model.predict(noisy)

        psnr = tf.image.psnr(clean, pred, max_val=1.0).numpy().mean()
        ssim = tf.image.ssim(clean, pred, max_val=1.0).numpy().mean()
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        save_prediction_image(pred[0], os.path.join(save_dir, f"{prefix}_denoised_{i}.png"))
        save_prediction_image(noisy[0], os.path.join(save_dir, f"{prefix}_noisy_{i}.png"))
        save_prediction_image(clean[0], os.path.join(save_dir, f"{prefix}_clean_{i}.png"))

    return psnr_list, ssim_list

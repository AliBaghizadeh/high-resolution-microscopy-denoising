# Data Preprocessing
PATCH_SIZE = 256
STRIDE = 128
BATCH_SIZE = 16
AUGMENTATION_PROBABILITIES = {
    "salt_pepper": 0.5,
    "plane_fixed": 0.4,
    "plane_random": 0.4,
    "scan": 0.3,
    "drift": 0.3
}

# Model Settings
INPUT_SHAPE = (256, 256, 1)
DROPOUT_RATE = 0.35
L2_STRENGTH = 2e-4
USE_MIXED_PRECISION = True

# Loss Function Weights
ALPHA = 1.0      # MSE
BETA = 0.1       # FFT
GAMMA = 0.1      # Poisson
DELTA = 0.2      # SSIM
EPSILON = 0.05   # TV
MU = 0.05        # Mean Intensity

# Training Settings
EPOCHS = 50
INITIAL_LR = 1e-4
PATIENCE = 7
MODEL_PATH = "model_best.h5"

# Evaluation
SAVE_RESULTS_DIR = "results"

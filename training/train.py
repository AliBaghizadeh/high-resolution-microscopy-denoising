import tensorflow as tf
from training.loss_functions import combined_loss
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import mixed_precision

# Enable mixed precision training globally
mixed_precision.set_global_policy('mixed_float16')

def get_callbacks(model_path='model_best.h5', patience=7):
    '''
    Create standard Keras callbacks for training.

    Args:
        model_path (str): File path to save the best model.
        patience (int): Number of epochs to wait before early stopping.

    Returns:
        list: List of Keras callbacks.
    '''
    return [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

def train_model(model, train_ds, val_ds, alpha, beta, gamma, delta, epsilon, mu,
                epochs=50, callbacks=None):
    '''
    Compile and train a model using a custom composite loss function.

    Args:
        model (tf.keras.Model): Model to train.
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        alpha, beta, gamma, delta, epsilon, mu (float): Weights for each loss component.
        epochs (int): Number of training epochs.
        callbacks (list): List of Keras callbacks.

    Returns:
        tf.keras.Model: Trained model.
    '''
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss_fn = lambda y_true, y_pred: combined_loss(y_true, y_pred, alpha, beta, gamma, delta, epsilon, mu)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mae'])
    model.fit(train_ds,
              validation_data=val_ds,
              epochs=epochs,
              callbacks=callbacks)
    return model

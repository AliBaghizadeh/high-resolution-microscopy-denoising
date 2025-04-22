"""
Utility functions for deep learning models.
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

def count_parameters(model):
    """
    Count the total number of trainable parameters in a model.
    
    Args:
        model: Keras model
        
    Returns:
        Total number of trainable parameters
    """
    return sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)

def visualize_model_architecture(model, to_file=None, show_shapes=True, show_layer_names=True):
    """
    Visualize the model architecture.
    
    Args:
        model: Keras model
        to_file: Path to save the visualization (if None, just displays it)
        show_shapes: Whether to show input/output shapes
        show_layer_names: Whether to show layer names
    """
    # Use tf.keras.utils.plot_model to generate the visualization
    if to_file:
        tf.keras.utils.plot_model(
            model,
            to_file=to_file,
            show_shapes=show_shapes,
            show_layer_names=show_layer_names,
            dpi=96
        )
        print(f"Model architecture saved to {to_file}")
    else:
        try:
            from IPython.display import display, Image
            
            # Create a temporary file
            temp_file = 'temp_model_architecture.png'
            tf.keras.utils.plot_model(
                model,
                to_file=temp_file,
                show_shapes=show_shapes,
                show_layer_names=show_layer_names,
                dpi=96
            )
            
            # Display the image
            display(Image(temp_file))
            
            # Clean up
            os.remove(temp_file)
        except ImportError:
            print("Unable to display model architecture. Install pydot and graphviz for visualization.")
            print("Model summary:")
            model.summary()

def save_model_summary_to_file(model, file_path):
    """
    Save a model summary to a text file.
    
    Args:
        model: Keras model
        file_path: Path to save the summary
    """
    # Create a string buffer to capture the summary
    from io import StringIO
    import sys
    
    # Store the default stdout
    default_stdout = sys.stdout
    
    # Create a StringIO object to capture summary
    string_io = StringIO()
    sys.stdout = string_io
    
    # Print the model summary
    model.summary()
    
    # Reset stdout to original
    sys.stdout = default_stdout
    
    # Write the captured output to file
    with open(file_path, 'w') as f:
        f.write(string_io.getvalue())
    
    print(f"Model summary saved to {file_path}")

def check_nan_weights(model):
    """
    Check if a model contains NaN weights.
    
    Args:
        model: Keras model
        
    Returns:
        Boolean indicating if any weights are NaN
    """
    has_nan = False
    
    for layer in model.layers:
        for weight in layer.weights:
            if np.isnan(weight.numpy()).any():
                print(f"NaN found in layer: {layer.name}, weight shape: {weight.shape}")
                has_nan = True
    
    return has_nan

def compare_model_predictions(models, model_names, sample_input, ground_truth=None):
    """
    Compare the predictions of multiple models on a single input.
    
    Args:
        models: List of Keras models
        model_names: List of names for the models
        sample_input: Input sample for prediction
        ground_truth: Ground truth for comparison (optional)
    """
    # Ensure sample_input is ready for prediction (add batch dimension if needed)
    if len(sample_input.shape) == 3:  # (height, width, channels)
        sample_input = np.expand_dims(sample_input, axis=0)
    
    # Get predictions from all models
    predictions = [model.predict(sample_input)[0] for model in models]
    
    # Create visualization
    n_models = len(models)
    fig_width = 4 * (n_models + (1 if ground_truth is not None else 0))
    
    plt.figure(figsize=(fig_width, 4))
    
    # Display input image
    plt.subplot(1, n_models + 1 + (1 if ground_truth is not None else 0), 1)
    plt.imshow(np.squeeze(sample_input[0]), cmap='gray')
    plt.title('Input')
    plt.axis('off')
    
    # Display predictions
    for i, (pred, name) in enumerate(zip(predictions, model_names)):
        plt.subplot(1, n_models + 1 + (1 if ground_truth is not None else 0), i + 2)
        plt.imshow(np.squeeze(pred), cmap='gray')
        plt.title(name)
        plt.axis('off')
    
    # Display ground truth if provided
    if ground_truth is not None:
        plt.subplot(1, n_models + 2, n_models + 2)
        plt.imshow(np.squeeze(ground_truth), cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def freeze_layers(model, layers_to_freeze=None, freeze_before_layer=None):
    """
    Freeze specific layers in a model.
    
    Args:
        model: Keras model
        layers_to_freeze: List of layer names to freeze (if None, use freeze_before_layer)
        freeze_before_layer: Freeze all layers before this layer name
    
    Returns:
        Model with frozen layers
    """
    if layers_to_freeze is not None:
        # Freeze specific layers
        for layer in model.layers:
            if layer.name in layers_to_freeze:
                layer.trainable = False
                print(f"Froze layer: {layer.name}")
    
    elif freeze_before_layer is not None:
        # Freeze all layers before a specific layer
        freeze = True
        for layer in model.layers:
            if layer.name == freeze_before_layer:
                freeze = False
            
            if freeze:
                layer.trainable = False
                print(f"Froze layer: {layer.name}")
    
    return model

def create_model_ensemble(models, input_size=(256, 256, 1), ensemble_method='average'):
    """
    Create an ensemble from multiple models.
    
    Args:
        models: List of Keras models for the ensemble
        input_size: Input shape for the ensemble model
        ensemble_method: Method to combine predictions ('average' or 'weighted')
        
    Returns:
        Ensemble model
    """
    if ensemble_method not in ['average', 'weighted']:
        raise ValueError(f"Unsupported ensemble method: {ensemble_method}")
    
    # Create input layer
    inputs = tf.keras.layers.Input(shape=input_size)
    
    # Get predictions from each model
    predictions = [model(inputs) for model in models]
    
    if ensemble_method == 'average':
        # Simple averaging of predictions
        output = tf.keras.layers.Average()(predictions)
    else:  # weighted
        # Add trainable weights for each model's prediction
        weighted_predictions = []
        for pred in predictions:
            # Initialize with weight close to 1/n_models
            weight = tf.keras.layers.Dense(1, activation='sigmoid', 
                                         bias_initializer=tf.keras.initializers.Constant(0.5))(
                tf.keras.layers.GlobalAveragePooling2D()(pred)
            )
            weighted_pred = tf.keras.layers.Multiply()([pred, weight])
            weighted_predictions.append(weighted_pred)
        
        # Sum the weighted predictions
        output = tf.keras.layers.Add()(weighted_predictions)
    
    # Create the ensemble model
    ensemble_model = tf.keras.Model(inputs=inputs, outputs=output)
    
    return ensemble_model 
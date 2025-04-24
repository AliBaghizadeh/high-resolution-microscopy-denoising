import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Lambda, Conv2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model

def unet_with_pretrained_encoder(name='vgg16', input_shape=(256, 256, 1), dropout_rate=0.4):
    '''
    Build a UNet model using a VGG16 encoder with custom layer freezing.

    Args:
        name (str): Encoder base name (e.g., 'vgg16').
        input_shape (tuple): Input image shape (default is (256, 256, 1)).
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        tf.keras.Model: UNet model with partially frozen VGG16 encoder.
    '''
    inputs = Input(shape=input_shape)

    # Convert grayscale to RGB (3 channels for VGG16)
    x = Lambda(lambda x: tf.repeat(x, 3, axis=-1))(inputs)

    # Load VGG16 base model
    base_model = VGG16(include_top=False, weights='imagenet', input_tensor=x)
    base_model.trainable = True  # Enable trainable for manual control

    # Capture skip connections
    skips = [
        base_model.get_layer("block1_conv2").output,
        base_model.get_layer("block2_conv2").output,
        base_model.get_layer("block3_conv3").output,
        base_model.get_layer("block4_conv3").output,
    ]

    # Encoder output for the decoder input
    x = skips[-1]

    # Define decoder block
    def decoder_block(x, skip, filters):
        x = Conv2D(filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = concatenate([x, skip])
        x = Conv2D(filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
        x = Conv2DTranspose(filters, 3, strides=2, padding='same', activation='relu')(x)
        return x

    # Decode
    x = decoder_block(x, skips[2], 512)
    x = decoder_block(x, skips[1], 256)
    x = decoder_block(x, skips[0], 128)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    outputs = Conv2D(1, 1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    # Apply partial freezing strategy
    encoder_start_index = 2  # Lambda layer is at index 1
    encoder_end_index = 15   # Typically block4_conv3 output layer index

    encoder_layers = model.layers[encoder_start_index:encoder_end_index]

    for layer in encoder_layers[:2]:
        layer.trainable = False

    for layer in encoder_layers[2:]:
        layer.trainable = True

    for layer in model.layers[encoder_end_index:]:
        layer.trainable = True

    return model


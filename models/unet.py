import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Conv2DTranspose
from tensorflow.keras.layers import concatenate, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def conv_block(input_tensor, num_filters, kernel_size=3, batch_norm=True, dropout_rate=0.0):
    """
    Creates a convolutional block with optional batch normalization and dropout.
    
    Args:
        input_tensor: Input tensor
        num_filters: Number of filters in the convolutional layer
        kernel_size: Size of the kernel for convolution
        batch_norm: Whether to apply batch normalization
        dropout_rate: Dropout rate (0 = no dropout)
    
    Returns:
        Tensor after applying convolution, normalization and activation
    """
    x = Conv2D(num_filters, (kernel_size, kernel_size), 
               padding='same', 
               kernel_regularizer=l2(1e-4))(input_tensor)
    
    if batch_norm:
        x = BatchNormalization()(x)
    
    x = Activation('relu')(x)
    
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    
    x = Conv2D(num_filters, (kernel_size, kernel_size), 
               padding='same',
               kernel_regularizer=l2(1e-4))(x)
    
    if batch_norm:
        x = BatchNormalization()(x)
    
    x = Activation('relu')(x)
    
    return x

def unet_model(input_size=(256, 256, 1), 
               filters=[64, 128, 256, 512, 1024], 
               batch_norm=True, 
               dropout_rates=[0.0, 0.0, 0.0, 0.0, 0.0], 
               use_transpose=True):
    """
    Creates a U-Net model for image denoising.
    
    Args:
        input_size: Input image dimensions (height, width, channels)
        filters: List of filter numbers for each level
        batch_norm: Whether to use batch normalization
        dropout_rates: List of dropout rates for each level
        use_transpose: Whether to use transposed convolution for upsampling
    
    Returns:
        Keras Model object
    """
    inputs = Input(input_size)
    
    # Contracting path (encoder)
    conv_layers = []
    x = inputs
    
    for i, f in enumerate(filters[:-1]):
        x = conv_block(x, f, batch_norm=batch_norm, dropout_rate=dropout_rates[i])
        conv_layers.append(x)
        x = MaxPooling2D((2, 2))(x)
    
    # Bridge
    x = conv_block(x, filters[-1], batch_norm=batch_norm, dropout_rate=dropout_rates[-1])
    
    # Expansive path (decoder)
    for i in reversed(range(len(filters)-1)):
        if use_transpose:
            x = Conv2DTranspose(filters[i], (2, 2), strides=(2, 2), padding='same')(x)
        else:
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(filters[i], (2, 2), padding='same')(x)
            x = Activation('relu')(x)
        
        x = concatenate([x, conv_layers[i]])
        x = conv_block(x, filters[i], batch_norm=batch_norm, dropout_rate=dropout_rates[i])
    
    # Output layer
    outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

def deep_unet_model(input_size=(256, 256, 1), 
                    filters=[64, 128, 256, 512, 1024, 2048], 
                    batch_norm=True,
                    dropout_rates=[0.0, 0.0, 0.0, 0.2, 0.3, 0.4],
                    use_transpose=True):
    """
    Creates a deeper U-Net model for image denoising.
    
    Args:
        input_size: Input image dimensions (height, width, channels)
        filters: List of filter numbers for each level
        batch_norm: Whether to use batch normalization
        dropout_rates: List of dropout rates for each level
        use_transpose: Whether to use transposed convolution for upsampling
    
    Returns:
        Keras Model object
    """
    inputs = Input(input_size)
    
    # Contracting path (encoder)
    conv_layers = []
    x = inputs
    
    for i, f in enumerate(filters[:-1]):
        x = conv_block(x, f, batch_norm=batch_norm, dropout_rate=dropout_rates[i])
        conv_layers.append(x)
        x = MaxPooling2D((2, 2))(x)
    
    # Bridge
    x = conv_block(x, filters[-1], batch_norm=batch_norm, dropout_rate=dropout_rates[-1])
    
    # Expansive path (decoder)
    for i in reversed(range(len(filters)-1)):
        if use_transpose:
            x = Conv2DTranspose(filters[i], (2, 2), strides=(2, 2), padding='same')(x)
        else:
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(filters[i], (2, 2), padding='same')(x)
            x = Activation('relu')(x)
        
        x = concatenate([x, conv_layers[i]])
        x = conv_block(x, filters[i], batch_norm=batch_norm, dropout_rate=dropout_rates[i])
    
    # Output layer
    outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

def residual_unet_model(input_size=(256, 256, 1), 
                       filters=[64, 128, 256, 512, 1024], 
                       batch_norm=True, 
                       dropout_rates=[0.0, 0.0, 0.0, 0.0, 0.0], 
                       use_transpose=True):
    """
    Creates a U-Net model with residual connections for image denoising.
    
    Args:
        input_size: Input image dimensions (height, width, channels)
        filters: List of filter numbers for each level
        batch_norm: Whether to use batch normalization
        dropout_rates: List of dropout rates for each level
        use_transpose: Whether to use transposed convolution for upsampling
    
    Returns:
        Keras Model object
    """
    inputs = Input(input_size)
    
    # Contracting path (encoder)
    conv_layers = []
    x = inputs
    
    for i, f in enumerate(filters[:-1]):
        x_in = x
        x = conv_block(x, f, batch_norm=batch_norm, dropout_rate=dropout_rates[i])
        
        # Add residual connection if input channels match
        if i > 0:
            # Make sure dimensions match with a 1x1 conv if needed
            x_res = Conv2D(f, (1, 1), padding='same')(x_in)
            x = tf.keras.layers.add([x, x_res])
        
        conv_layers.append(x)
        x = MaxPooling2D((2, 2))(x)
    
    # Bridge
    x_bridge = x
    x = conv_block(x, filters[-1], batch_norm=batch_norm, dropout_rate=dropout_rates[-1])
    
    # Add residual connection in the bridge
    x_res = Conv2D(filters[-1], (1, 1), padding='same')(x_bridge)
    x = tf.keras.layers.add([x, x_res])
    
    # Expansive path (decoder)
    for i in reversed(range(len(filters)-1)):
        if use_transpose:
            x = Conv2DTranspose(filters[i], (2, 2), strides=(2, 2), padding='same')(x)
        else:
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(filters[i], (2, 2), padding='same')(x)
            x = Activation('relu')(x)
        
        x = concatenate([x, conv_layers[i]])
        
        x_in = x
        x = conv_block(x, filters[i], batch_norm=batch_norm, dropout_rate=dropout_rates[i])
        
        # Add residual connection in decoder
        x_res = Conv2D(filters[i], (1, 1), padding='same')(x_in)
        x = tf.keras.layers.add([x, x_res])
    
    # Output layer
    outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model 
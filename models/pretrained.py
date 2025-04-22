import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, Conv2DTranspose, Lambda
from tensorflow.keras.layers import concatenate, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def vgg16_unet(input_size=(256, 256, 1), pretrained_weights=True, freeze_encoder=True):
    """
    Creates a U-Net model using VGG16 as the encoder for STEM image denoising.
    
    Args:
        input_size: Input image dimensions (height, width, channels)
        pretrained_weights: Whether to use pretrained weights for the encoder
        freeze_encoder: Whether to freeze the encoder during training
    
    Returns:
        Keras Model object
    """
    # Prepare input - replicate single channel to 3 channels
    inputs = Input(input_size)
    input_3channel = Lambda(lambda x: tf.tile(x, [1, 1, 1, 3]))(inputs)
    
    # Load VGG16 model as encoder
    vgg16 = VGG16(include_top=False, weights='imagenet' if pretrained_weights else None, 
                 input_tensor=input_3channel)
    
    # Freeze encoder layers if required
    if freeze_encoder:
        for layer in vgg16.layers:
            layer.trainable = False
    
    # Get intermediate layer outputs for skip connections
    # Block 1
    block1_conv2 = vgg16.get_layer('block1_conv2').output  # 64 filters
    # Block 2
    block2_conv2 = vgg16.get_layer('block2_conv2').output  # 128 filters
    # Block 3
    block3_conv3 = vgg16.get_layer('block3_conv3').output  # 256 filters
    # Block 4
    block4_conv3 = vgg16.get_layer('block4_conv3').output  # 512 filters
    # Block 5 (bridge)
    bridge = vgg16.get_layer('block5_conv3').output       # 512 filters
    
    # Decoder path
    # Upsampling block 5
    up5 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bridge)
    up5 = concatenate([up5, block4_conv3])
    up5 = Conv2D(512, (3, 3), padding='same', activation='relu')(up5)
    up5 = BatchNormalization()(up5)
    up5 = Conv2D(512, (3, 3), padding='same', activation='relu')(up5)
    up5 = BatchNormalization()(up5)
    
    # Upsampling block 4
    up4 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(up5)
    up4 = concatenate([up4, block3_conv3])
    up4 = Conv2D(256, (3, 3), padding='same', activation='relu')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Conv2D(256, (3, 3), padding='same', activation='relu')(up4)
    up4 = BatchNormalization()(up4)
    
    # Upsampling block 3
    up3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(up4)
    up3 = concatenate([up3, block2_conv2])
    up3 = Conv2D(128, (3, 3), padding='same', activation='relu')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Conv2D(128, (3, 3), padding='same', activation='relu')(up3)
    up3 = BatchNormalization()(up3)
    
    # Upsampling block 2
    up2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(up3)
    up2 = concatenate([up2, block1_conv2])
    up2 = Conv2D(64, (3, 3), padding='same', activation='relu')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Conv2D(64, (3, 3), padding='same', activation='relu')(up2)
    up2 = BatchNormalization()(up2)
    
    # Output layer
    outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(up2)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

def resnet50_unet(input_size=(256, 256, 1), pretrained_weights=True, freeze_encoder=True):
    """
    Creates a U-Net model using ResNet50 as the encoder for STEM image denoising.
    
    Args:
        input_size: Input image dimensions (height, width, channels)
        pretrained_weights: Whether to use pretrained weights for the encoder
        freeze_encoder: Whether to freeze the encoder during training
    
    Returns:
        Keras Model object
    """
    # Prepare input - replicate single channel to 3 channels
    inputs = Input(input_size)
    input_3channel = Lambda(lambda x: tf.tile(x, [1, 1, 1, 3]))(inputs)
    
    # Load ResNet50 model as encoder
    resnet = ResNet50(include_top=False, weights='imagenet' if pretrained_weights else None, 
                     input_tensor=input_3channel)
    
    # Freeze encoder layers if required
    if freeze_encoder:
        for layer in resnet.layers:
            layer.trainable = False
    
    # Get intermediate layer outputs for skip connections
    # These are the activation layers after each stage
    stage1_out = resnet.get_layer('conv1_relu').output        # 64 filters
    stage2_out = resnet.get_layer('conv2_block3_out').output  # 256 filters
    stage3_out = resnet.get_layer('conv3_block4_out').output  # 512 filters
    stage4_out = resnet.get_layer('conv4_block6_out').output  # 1024 filters
    bridge = resnet.get_layer('conv5_block3_out').output      # 2048 filters
    
    # Decoder path
    # Upsampling block 5
    up5 = Conv2DTranspose(1024, (2, 2), strides=(2, 2), padding='same')(bridge)
    up5 = concatenate([up5, stage4_out])
    up5 = Conv2D(1024, (3, 3), padding='same', activation='relu')(up5)
    up5 = BatchNormalization()(up5)
    up5 = Conv2D(1024, (3, 3), padding='same', activation='relu')(up5)
    up5 = BatchNormalization()(up5)
    
    # Upsampling block 4
    up4 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(up5)
    up4 = concatenate([up4, stage3_out])
    up4 = Conv2D(512, (3, 3), padding='same', activation='relu')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Conv2D(512, (3, 3), padding='same', activation='relu')(up4)
    up4 = BatchNormalization()(up4)
    
    # Upsampling block 3
    up3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(up4)
    up3 = concatenate([up3, stage2_out])
    up3 = Conv2D(256, (3, 3), padding='same', activation='relu')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Conv2D(256, (3, 3), padding='same', activation='relu')(up3)
    up3 = BatchNormalization()(up3)
    
    # Upsampling block 2
    up2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(up3)
    up2 = concatenate([up2, stage1_out])
    up2 = Conv2D(64, (3, 3), padding='same', activation='relu')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Conv2D(64, (3, 3), padding='same', activation='relu')(up2)
    up2 = BatchNormalization()(up2)
    
    # Final upsampling to original size
    up1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(up2)
    up1 = Conv2D(32, (3, 3), padding='same', activation='relu')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Conv2D(32, (3, 3), padding='same', activation='relu')(up1)
    up1 = BatchNormalization()(up1)
    
    # Output layer
    outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(up1)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

def efficient_unet(input_size=(256, 256, 1), efficient_net_version='B0', 
                 pretrained_weights=True, freeze_encoder=True):
    """
    Creates a U-Net model using EfficientNet as the encoder for STEM image denoising.
    
    Args:
        input_size: Input image dimensions (height, width, channels)
        efficient_net_version: EfficientNet version ('B0' to 'B7')
        pretrained_weights: Whether to use pretrained weights for the encoder
        freeze_encoder: Whether to freeze the encoder during training
    
    Returns:
        Keras Model object
    """
    # Import EfficientNet dynamically to avoid errors if it's not installed
    try:
        if tf.__version__ < '2.3':
            # For older TF versions
            from efficientnet.tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2
            from efficientnet.tfkeras import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
            efficient_nets = {
                'B0': EfficientNetB0,
                'B1': EfficientNetB1,
                'B2': EfficientNetB2,
                'B3': EfficientNetB3,
                'B4': EfficientNetB4,
                'B5': EfficientNetB5,
                'B6': EfficientNetB6,
                'B7': EfficientNetB7
            }
        else:
            # For TF 2.3+
            from tensorflow.keras.applications import (
                EfficientNetB0, EfficientNetB1, EfficientNetB2, 
                EfficientNetB3, EfficientNetB4, EfficientNetB5,
                EfficientNetB6, EfficientNetB7
            )
            efficient_nets = {
                'B0': EfficientNetB0,
                'B1': EfficientNetB1,
                'B2': EfficientNetB2,
                'B3': EfficientNetB3,
                'B4': EfficientNetB4,
                'B5': EfficientNetB5,
                'B6': EfficientNetB6,
                'B7': EfficientNetB7
            }
    except ImportError:
        raise ImportError("EfficientNet not found. Please install it with: pip install efficientnet")
    
    # Get the appropriate EfficientNet model
    EfficientNet = efficient_nets.get(efficient_net_version.upper())
    if EfficientNet is None:
        raise ValueError(f"Invalid EfficientNet version: {efficient_net_version}. Choose from B0-B7")
    
    # Prepare input - replicate single channel to 3 channels
    inputs = Input(input_size)
    input_3channel = Lambda(lambda x: tf.tile(x, [1, 1, 1, 3]))(inputs)
    
    # Load EfficientNet model as encoder
    encoder = EfficientNet(include_top=False, weights='imagenet' if pretrained_weights else None, 
                         input_tensor=input_3channel)
    
    # Freeze encoder layers if required
    if freeze_encoder:
        for layer in encoder.layers:
            layer.trainable = False
    
    # Extract intermediate layers for skip connections
    # The exact layer names will depend on the EfficientNet version
    # These are examples for EfficientNetB0
    skips = []
    for i, layer in enumerate(encoder.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) and layer.strides == (2, 2):
            if i > 0:  # Skip the first conv layer
                skips.append(encoder.layers[i-1].output)
    
    # Add the final activation as bridge
    bridge = encoder.outputs[0]
    
    # Add the final activation as first skip connection
    skips.append(bridge)
    skips = list(reversed(skips))
    
    # Decoder path with skip connections
    x = bridge
    
    # Number of filters for decoder blocks
    filters = [512, 256, 128, 64, 32]
    
    # Build decoder
    for i in range(len(skips) - 1):
        x = Conv2DTranspose(filters[i], (2, 2), strides=(2, 2), padding='same')(x)
        x = concatenate([x, skips[i+1]])
        x = Conv2D(filters[i], (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters[i], (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
    
    # Output layer
    outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model 
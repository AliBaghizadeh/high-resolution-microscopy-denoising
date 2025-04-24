from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras import mixed_precision

# Enable mixed precision
mixed_precision.set_global_policy('mixed_float16')

def build_unet(input_shape=(256, 256, 1), dropout_rate=0.35, l2_strength=2e-4):
    '''
    Build a UNet model from scratch using functional API, L2 regularization and mixed precision.

    Args:
        input_shape (tuple): Input shape of images, default (256, 256, 1).
        dropout_rate (float): Dropout rate to apply between layers.
        l2_strength (float): L2 regularization weight.

    Returns:
        tf.keras.Model: Constructed UNet model.
    '''
    inputs = Input(shape=input_shape)

    def conv_block(x, filters):
        x = Conv2D(filters, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_strength))(x)
        x = Conv2D(filters, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_strength))(x)
        x = Dropout(dropout_rate)(x)
        return x

    # Encoder
    c1 = conv_block(inputs, 64)
    p1 = MaxPooling2D()(c1)
    c2 = conv_block(p1, 128)
    p2 = MaxPooling2D()(c2)
    c3 = conv_block(p2, 256)
    p3 = MaxPooling2D()(c3)
    c4 = conv_block(p3, 512)
    p4 = MaxPooling2D()(c4)

    # Bridge
    c5 = conv_block(p4, 1024)

    # Decoder
    u6 = UpSampling2D()(c5)
    u6 = concatenate([u6, c4])
    c6 = conv_block(u6, 512)
    u7 = UpSampling2D()(c6)
    u7 = concatenate([u7, c3])
    c7 = conv_block(u7, 256)
    u8 = UpSampling2D()(c7)
    u8 = concatenate([u8, c2])
    c8 = conv_block(u8, 128)
    u9 = UpSampling2D()(c8)
    u9 = concatenate([u9, c1])
    c9 = conv_block(u9, 64)

    outputs = Conv2D(1, 1, activation='sigmoid', dtype='float32')(c9)
    return Model(inputs, outputs)

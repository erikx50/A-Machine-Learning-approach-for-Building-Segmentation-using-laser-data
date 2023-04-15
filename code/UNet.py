from tensorflow.keras import layers, optimizers, backend, Model
from tensorflow.keras.applications import EfficientNetB4, EfficientNetV2S, ResNet50V2, DenseNet201


# U-Net
def unet(input_shape=(512, 512, 3)):
    """
    Creates a neural network using the U-Net architecture.
    Args:
        input_shape: The size of the input image.
    Returns:
        A U-Net model.
    """
    # Encoder Part
    # Layer 1
    inputs = layers.Input(input_shape)
    inputs_rescaled = layers.Lambda(lambda x: x / 255)(inputs)  # Rescale input pixel values to floating point values
    c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs_rescaled)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    # Layer 2
    c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Layer 3
    c3 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Layer 4
    c4 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Layer 5
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)

    # Decoder Part
    # Layer 6
    u6 = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)

    # Layer 7
    u7 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    c7 = layers.BatchNormalization()(c7)

    # Layer 8
    u8 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    c8 = layers.BatchNormalization()(c8)

    # Layer 9
    u9 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    c9 = layers.BatchNormalization()(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    # Compiling model
    model = Model(inputs=[inputs], outputs=[outputs], name='UNet')
    return model


################################################################
# U-Net using different backbones
def conv_block(input, num_filters):
    """
    Creates a block consisting of two convolutional layers.
    Args:
        input: The output of the previous up-convolution.
        num_filters: Number of filters to be used in the convolution.
    Returns:
        The result of the final calculation of the current layer.
    """
    x = layers.Conv2D(num_filters, 3, kernel_initializer='he_normal',  padding='same')(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(num_filters, 3, kernel_initializer='he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def decoder_block(input, skip_features, num_filters):
    """
    Create a decoder block used in the decoder part of the network.
    Args:
        input: The output of the previous layer.
        skip_features: The features of the same size from the encoder part of the network.
        num_filters: Number of filters to be used in the convolution.
    Returns:
        The result of the final calculation of the current layer.
    """
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(input)
    x = layers.concatenate([x, skip_features])
    x = conv_block(x, num_filters)
    return x


# EfficientNetB4 UNet
def EfficientNetB4_unet(input_shape=(512, 512, 3), weight='imagenet'):
    """
    Creates a neural network using the U-Net architecture and EfficientNetB4 as backbone.
    Args:
        input_shape: The size of the input image.
        weight: Pre-trained weights.
    Returns:
        A U-Net model using EfficientNetB4 as backbone.
    """
    # Input
    inputs = layers.Input(input_shape)

    # Loading pre trained model
    EffNetB4 = EfficientNetB4(include_top=False, weights=weight, input_tensor=inputs)

    # Encoder
    s1 = EffNetB4.get_layer('rescaling').output  # 512 x 512
    s2 = EffNetB4.get_layer('block2a_expand_activation').output  # 256 x 256
    s3 = EffNetB4.get_layer('block3a_expand_activation').output  # 128 x 128
    s4 = EffNetB4.get_layer('block4a_expand_activation').output  # 64 x 64

    # Bottleneck
    b1 = EffNetB4.get_layer('block6a_expand_activation').output  # 32 x 32

    # Decoder
    d1 = decoder_block(b1, s4, 512)  # 64 x 64
    d2 = decoder_block(d1, s3, 256)    # 128 x 128
    d3 = decoder_block(d2, s2, 128)   # 256 x 256
    d4 = decoder_block(d3, s1, 64)   # 512 x 512

    # Output
    outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(d4)

    model = Model(inputs, outputs, name='EfficientNetB4_U-Net')
    return model


# EfficientNetV2S UNet
def EfficientNetV2S_unet(input_shape=(512, 512, 3), weight='imagenet'):
    """
    Creates a neural network using the U-Net architecture and EfficientNetV2S as backbone.
    Args:
        input_shape: The size of the input image.
        weight: Pre-trained weights.
    Returns:
        A U-Net model using EfficientNetB4 as backbone.
    """
    # Input
    inputs = layers.Input(input_shape)

    # Loading pre trained model
    EffNetV2S = EfficientNetV2S(include_top=False, weights=weight, input_tensor=inputs)

    # Encoder
    s1 = EffNetV2S.get_layer('rescaling').output  # 512 x 512
    s2 = EffNetV2S.get_layer('block1b_add').output  # 256 x 256
    s3 = EffNetV2S.get_layer('block2d_add').output  # 128 x 128
    s4 = EffNetV2S.get_layer('block4a_expand_activation').output  # 64 x 64

    # Bottleneck
    b1 = EffNetV2S.get_layer('block6a_expand_activation').output  # 32 x 32

    # Decoder
    d1 = decoder_block(b1, s4, 512)  # 64 x 64
    d2 = decoder_block(d1, s3, 256)    # 128 x 128
    d3 = decoder_block(d2, s2, 128)   # 256 x 256
    d4 = decoder_block(d3, s1, 64)   # 512 x 512

    # Output
    outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(d4)

    model = Model(inputs, outputs, name='EfficientNetV2S_U-Net')
    return model


# ResNet50V2 UNet
def ResNet50V2_unet(input_shape=(512, 512, 3), weight='imagenet'):
    """
    Creates a neural network using the U-Net architecture and ResNet50V2 as backbone.
    Args:
        input_shape: The size of the input image.
        weight: Pre-trained weights.
    Returns:
        A U-Net model using EfficientNetB4 as backbone.
    """
    # Input
    inputs = layers.Input(input_shape)

    # Loading pre trained model
    ResNet50 = ResNet50V2(include_top=False, weights=weight, input_tensor=inputs)

    # Encoder
    s1 = ResNet50.get_layer('input_1').output  # 512 x 512
    s2 = ResNet50.get_layer('conv1_conv').output  # 256 x 256
    s3 = ResNet50.get_layer('conv2_block3_1_relu').output  # 128 x 128
    s4 = ResNet50.get_layer('conv3_block4_1_relu').output  # 64 x 64

    # Bottleneck
    b1 = ResNet50.get_layer('conv4_block6_1_relu').output  # 32 x 32

    # Decoder
    d1 = decoder_block(b1, s4, 512)  # 64 x 64
    d2 = decoder_block(d1, s3, 256)    # 128 x 128
    d3 = decoder_block(d2, s2, 128)   # 256 x 256
    d4 = decoder_block(d3, s1, 64)   # 512 x 512

    # Output
    outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(d4)

    model = Model(inputs, outputs, name='ResNet50V2_U-Net')
    return model


# DenseNet201 UNet
def DenseNet201_unet(input_shape=(512, 512, 3), weight='imagenet'):
    """
    Creates a neural network using the U-Net architecture and DenseNet201 as backbone.
    Args:
        input_shape: The size of the input image.
        weight: Pre-trained weights.
    Returns:
        A U-Net model using EfficientNetB4 as backbone.
    """
    # Input
    inputs = layers.Input(input_shape)

    # Loading pre trained model
    DenseNet = DenseNet201(include_top=False, weights=weight, input_tensor=inputs)

    # Encoder
    s1 = DenseNet.get_layer('input_1').output  # 512 x 512
    s2 = DenseNet.get_layer('conv1/relu').output  # 256 x 256
    s3 = DenseNet.get_layer('pool2_conv').output  # 128 x 128
    s4 = DenseNet.get_layer('pool3_conv').output  # 64 x 64

    # Bottleneck
    b1 = DenseNet.get_layer('pool4_conv').output  # 32 x 32

    # Decoder
    d1 = decoder_block(b1, s4, 512)  # 64 x 64
    d2 = decoder_block(d1, s3, 256)    # 128 x 128
    d3 = decoder_block(d2, s2, 128)   # 256 x 256
    d4 = decoder_block(d3, s1, 64)   # 512 x 512

    # Output
    outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(d4)

    model = Model(inputs, outputs, name='DenseNet201_U-Net')
    return model

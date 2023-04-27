from tensorflow.keras import layers, optimizers, backend, Model
from tensorflow.keras.applications import EfficientNetB4, EfficientNetV2S, ResNet50V2, DenseNet201


# Functions for block types of CT-Unet
def DB_block(input1, num_filters):
    """
    Creates a Dense Block.
    Args:
        input1: The output of the first encoder or SCAB block.
        num_filters: Number of filters to be used in the convolution.
    Returns:
        The input for the next calculation.
    """
    x = layers.BatchNormalization()(input1)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, (1, 1), kernel_initializer='he_normal', padding='same')(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    c1 = layers.Conv2D(num_filters, (3, 3), kernel_initializer='he_normal', padding='same')(x)

    x = layers.concatenate([input1, c1])

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, (1, 1), kernel_initializer='he_normal', padding='same')(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, (3, 3), kernel_initializer='he_normal', padding='same')(x)

    x = layers.concatenate([input1, x, c1])
    return x


def DBB_block(input1, input2, num_filters):
    """
    Creates a Dense Boundary Block.
    Args:
        input1: The output of encoder block.
        input2: The output of the previous stage Boundary Block.
        num_filters: Number of filters to be used in the convolution.
    Returns:
        The input for the next calculation.
    """
    x = DB_block(input1, num_filters)

    x = layers.Conv2D(num_filters, (1, 1), kernel_initializer='he_normal', padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(num_filters, (2, 2), kernel_initializer='he_normal', strides=2, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)

    # Rescale input2 to have the same filter dimension as x
    input2 = layers.Conv2D(num_filters, (1, 1), kernel_initializer='he_normal', padding="same")(input2)
    input2 = layers.Activation("relu")(input2)
    input2 = layers.BatchNormalization()(input2)
    ####################################################

    x = layers.Add()([x, input2])

    x = layers.Conv2D(num_filters, (1, 1), kernel_initializer='he_normal', padding='same')(x)

    # Reduce size of x to match its original size before Conv2DTranspose#
    x = layers.MaxPooling2D((2, 2))(x)
    ####################################################
    return x


def SCAB_block(input1, input2, num_filters, final=False):
    """
    Creates a Spatial Channel Attention Block.
    Args:
        input1: The output of the previous decoder layer.
        input2: The output of the DB/DBB in the skip connection at the same level.
        num_filters: Number of filters to be used in the convolution.
        final: Indicator if this is the final SCAB block in the network.
    Returns:
        The input for the next calculation.
    """
    x = layers.Conv2D(num_filters, (1, 1), kernel_initializer='he_normal', padding='same')(input1)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, (1, 1), kernel_initializer='glorot_normal', padding='same')(x)
    x = layers.Activation('sigmoid')(x)

    if final:
        input2 = layers.Conv2D(num_filters, (1, 1), kernel_initializer='he_normal', padding='same')(input2)

    x = layers.Multiply()([x, input2])
    x = layers.concatenate([input1, x])
    c1 = layers.Conv2D(num_filters, (1, 1), kernel_initializer='he_normal', padding='same')(x)

    x = layers.GlobalAveragePooling2D(keepdims=True)(c1)

    x = layers.Conv2D(num_filters, (1, 1), kernel_initializer='he_normal', padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, (1, 1), kernel_initializer='glorot_normal', padding='same')(x)
    x = layers.Activation('sigmoid')(x)

    x = layers.Multiply()([x, c1])
    x = layers.Add()([x, c1])
    return x


def conv_block(input, num_filters):
    """
    Creates a block consisting of a convolutional layer.
    Args:
        input: The output of the previous calculation.
        num_filters: Number of filters to be used in the convolution.
    Returns:
        The input for the next calculation.
    """
    x = layers.Conv2D(num_filters, 3, activation='relu', kernel_initializer='he_normal',  padding='same')(input)
    x = layers.BatchNormalization()(x)
    return x


def deconv_block(input, num_filters):
    """
    Creates a block consisting of a de-convolution layer.
    Args:
        input: The output of the previous calculation.
        num_filters: Number of filters to be used in the convolution.
    Returns:
        The input for the next calculation.
    """
    x = layers.Conv2DTranspose(num_filters, 1, activation='relu', kernel_initializer='he_normal', padding='same')(input)
    x = layers.BatchNormalization()(x)
    return x


def decoder_block(input, skip_output, num_filters, final=False):
    """
    Creates a block for the decoder par of the network.
    Args:
        input: The output of the previous calculation.
        skip_output: The output of the skip connection(DB/DBB block) of the same size.
        num_filters: Number of filters to be used in the convolution.
        final: Indicator if this is the final decoder layer.
    Returns:
        The input for the decoder block. If final=True -> Input for the output layer.
    """
    x = layers.Conv2DTranspose(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', strides=2, padding='same')(input)
    x = layers.BatchNormalization()(x)
    x = SCAB_block(x, skip_output, num_filters, final)
    x = DB_block(x, num_filters)
    x = conv_block(x, num_filters)
    return x


def bottleneck(input, num_filters):
    """
    Creates the bottleneck layer for the CT-UNet architecture.
    Args:
        input: The output of the last encoder layer.
    Returns:
        The input of the first decoder layer.
    """
    x = layers.MaxPooling2D((2, 2))(input)
    x = conv_block(x, num_filters)
    return x


# EfficientNetB4 CT-UNet
def EfficientNetB4_CTUnet(input_shape=(512, 512, 3), weight='imagenet'):
    # Input
    inputs = layers.Input(input_shape)

    # Loading pre trained model
    EffNetB4 = EfficientNetB4(include_top=False, weights=weight, input_tensor=inputs)

    # Encoder
    res0 = EffNetB4.get_layer('rescaling_1').output  # 512 x 512
    res1 = EffNetB4.get_layer('block2a_expand_activation').output  # 256 x 256
    res2 = EffNetB4.get_layer('block3a_expand_activation').output  # 128 x 128
    res3 = EffNetB4.get_layer('block4a_expand_activation').output  # 64 x 64
    res4 = EffNetB4.get_layer('block6a_expand_activation').output  # 32 x 32

    # Skip connection blocks
    skip0 = DB_block(res0, 16)
    skip1 = DBB_block(res1, skip0, 32)
    skip2 = DBB_block(res2, skip1, 64)
    skip3 = DBB_block(res3, skip2, 128)
    skip4 = DBB_block(res4, skip3, 256)

    # Bottleneck
    b1 = bottleneck(res4, 512)  # 16 x 16

    # Decoder
    d1 = decoder_block(b1, skip4, 256)  # 32 x 32
    d2 = decoder_block(d1, skip3, 128)    # 64 x 64
    d3 = decoder_block(d2, skip2, 64)   # 128 x 128
    d4 = decoder_block(d3, skip1, 32)   # 256 x 256
    d5 = decoder_block(d4, skip0, 16, True)   # 512 x 512

    # Output
    outputs = conv_block(d5, 16)
    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(outputs)

    model = Model(inputs, outputs, name="EfficientNetB4_CTU-Net")
    return model
'''
def EfficientNetB4_CTUnet(input_shape=(512, 512, 3), weight='imagenet'):
    # Input
    inputs = layers.Input(input_shape)

    # Loading pre trained model
    EffNetB4 = EfficientNetB4(include_top=False, weights=weight, input_tensor=inputs)

    # Encoder
    res0 = EffNetB4.get_layer('rescaling_1').output  # 512 x 512
    res1 = EffNetB4.get_layer('block2a_expand_activation').output  # 256 x 256
    res2 = EffNetB4.get_layer('block3a_expand_activation').output  # 128 x 128
    res3 = EffNetB4.get_layer('block4a_expand_activation').output  # 64 x 64
    res4 = EffNetB4.get_layer('block6a_expand_activation').output  # 32 x 32

    # Skip connection blocks
    skip0 = DB_block(res0, 32)
    skip1 = DBB_block(res1, skip0, 64)
    skip2 = DBB_block(res2, skip1, 128)
    skip3 = DBB_block(res3, skip2, 256)
    skip4 = DBB_block(res4, skip3, 512)

    # Bottleneck
    b1 = bottleneck(res4, 512)  # 16 x 16

    # Decoder
    d1 = decoder_block(b1, skip4, 512)  # 32 x 32
    d2 = decoder_block(d1, skip3, 256)    # 64 x 64
    d3 = decoder_block(d2, skip2, 128)   # 128 x 128
    d4 = decoder_block(d3, skip1, 64)   # 256 x 256
    d5 = decoder_block(d4, skip0, 32, True)   # 512 x 512

    # Output
    outputs = conv_block(d5, 32)
    outputs = layers.Conv2D(1, 1, kernel_initializer='glorot_normal', padding="same", activation="sigmoid")(outputs)

    model = Model(inputs, outputs, name="EfficientNetB4_CTU-Net")
    return model
'''

# EfficientNetV2S CT-UNet
def EfficientNetV2S_CTUnet(input_shape=(512, 512, 3), weight='imagenet'):
    """
    Creates a neural network using the CT-UNet architecture and EfficientNetV2S as backbone.
    Args:
        input_shape: The size of the input image.
        weight: Pre-trained weights.
    Returns:
        A CT-UNet model using EfficientNetV2S as backbone.
    """
    # Input
    inputs = layers.Input(input_shape)

    # Loading pre trained model
    EffNetV2S = EfficientNetV2S(include_top=False, weights=weight, input_tensor=inputs)

    # Encoder
    res0 = EffNetV2S.get_layer('rescaling').output  # 512 x 512
    res1 = EffNetV2S.get_layer('block1b_add').output  # 256 x 256
    res2 = EffNetV2S.get_layer('block2d_add').output  # 128 x 128
    res3 = EffNetV2S.get_layer('block4a_expand_activation').output  # 64 x 64
    res4 = EffNetV2S.get_layer('block6a_expand_activation').output  # 32 x 32

    # Skip connection blocks
    skip0 = DB_block(res0, 32)
    skip1 = DBB_block(res1, skip0, 64)
    skip2 = DBB_block(res2, skip1, 128)
    skip3 = DBB_block(res3, skip2, 256)
    skip4 = DBB_block(res4, skip3, 512)

    # Bottleneck
    b1 = bottleneck(res4, 512)  # 16 x 16

    # Decoder
    d1 = decoder_block(b1, skip4, 512)  # 32 x 32
    d2 = decoder_block(d1, skip3, 256)    # 64 x 64
    d3 = decoder_block(d2, skip2, 128)   # 128 x 128
    d4 = decoder_block(d3, skip1, 64)   # 256 x 256
    d5 = decoder_block(d4, skip0, 32, True)   # 512 x 512

    # Output
    outputs = conv_block(d5, 32)
    outputs = layers.Conv2D(1, 1, kernel_initializer='glorot_normal', padding="same", activation="sigmoid")(outputs)

    model = Model(inputs, outputs, name='EfficientNetV2S_CTU-Net')

    return model


# ResNet50V2 CT-UNet
def ResNet50V2_CTUnet(input_shape=(512, 512, 3), weight='imagenet'):
    """
    Creates a neural network using the CT-UNet architecture and ResNet50V2 as backbone.
    Args:
        input_shape: The size of the input image.
        weight: Pre-trained weights.
    Returns:
        A CT-UNet model using ResNet50V2 as backbone.
    """
    # Input
    inputs = layers.Input(input_shape)

    # Loading pre trained model
    ResNet50 = ResNet50V2(include_top=False, weights=weight, input_tensor=inputs)

    # Encoder
    res0 = ResNet50.get_layer('input_1').output  # 512 x 512
    res1 = ResNet50.get_layer('conv1_conv').output  # 256 x 256
    res2 = ResNet50.get_layer('conv2_block3_1_relu').output  # 128 x 128
    res3 = ResNet50.get_layer('conv3_block4_1_relu').output  # 64 x 64
    res4 = ResNet50.get_layer('conv4_block6_1_relu').output  # 32 x 32

    # Skip connection blocks
    skip0 = DB_block(res0, 32)
    skip1 = DBB_block(res1, skip0, 64)
    skip2 = DBB_block(res2, skip1, 128)
    skip3 = DBB_block(res3, skip2, 256)
    skip4 = DBB_block(res4, skip3, 512)

    # Bottleneck
    b1 = bottleneck(res4, 512)  # 16 x 16

    # Decoder
    d1 = decoder_block(b1, skip4, 512)  # 32 x 32
    d2 = decoder_block(d1, skip3, 256)    # 64 x 64
    d3 = decoder_block(d2, skip2, 128)   # 128 x 128
    d4 = decoder_block(d3, skip1, 64)   # 256 x 256
    d5 = decoder_block(d4, skip0, 32, True)   # 512 x 512

    # Output
    outputs = conv_block(d5, 32)
    outputs = layers.Conv2D(1, 1, kernel_initializer='glorot_normal', padding="same", activation="sigmoid")(outputs)

    model = Model(inputs, outputs, name='ResNet50V2_CTU-Net')
    return model


# DenseNet201 CT-UNet
def DenseNet201_CTUnet(input_shape=(512, 512, 3), weight='imagenet'):
    """
    Creates a neural network using the CT-UNet architecture and DenseNet201 as backbone.
    Args:
        input_shape: The size of the input image.
        weight: Pre-trained weights.
    Returns:
        A CT-UNet model using DenseNet201 as backbone.
    """
    # Input
    inputs = layers.Input(input_shape)

    # Loading pre trained model
    DenseNet = DenseNet201(include_top=False, weights=weight, input_tensor=inputs)

    # Encoder
    res0 = DenseNet.get_layer('input_1').output  # 512 x 512
    res1 = DenseNet.get_layer('conv1/relu').output  # 256 x 256
    res2 = DenseNet.get_layer('pool2_conv').output  # 128 x 128
    res3 = DenseNet.get_layer('pool3_conv').output  # 64 x 64
    res4 = DenseNet.get_layer('pool4_conv').output  # 32 x 32

    # Skip connection blocks
    skip0 = DB_block(res0, 32)
    skip1 = DBB_block(res1, skip0, 64)
    skip2 = DBB_block(res2, skip1, 128)
    skip3 = DBB_block(res3, skip2, 256)
    skip4 = DBB_block(res4, skip3, 512)

    # Bottleneck
    b1 = bottleneck(res4, 512)  # 16 x 16

    # Decoder
    d1 = decoder_block(b1, skip4, 512)  # 32 x 32
    d2 = decoder_block(d1, skip3, 256)    # 64 x 64
    d3 = decoder_block(d2, skip2, 128)   # 128 x 128
    d4 = decoder_block(d3, skip1, 64)   # 256 x 256
    d5 = decoder_block(d4, skip0, 32, True)   # 512 x 512

    # Output
    outputs = conv_block(d5, 32)
    outputs = layers.Conv2D(1, 1, kernel_initializer='glorot_normal', padding="same", activation="sigmoid")(outputs)

    model = Model(inputs, outputs, name='DenseNet201_CTU-Net')
    return model

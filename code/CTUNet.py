import tensorflow as tf
from tensorflow.keras import layers, optimizers, backend, Model
from tensorflow.keras.applications import EfficientNetB4
from UNet import jaccard_coef, jaccard_coef_loss, dice_coef_loss


def DB_block(input1, num_filters):
    x = layers.BatchNormalization()(input1)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(num_filters, (1, 1), padding="same")(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    c1 = layers.Conv2D(num_filters, (3, 3), padding="same")(x)

    x = layers.concatenate([input1,c1])

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(num_filters, (1, 1), padding="same")(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(num_filters, (3, 3), padding="same")(x)

    x = layers.concatenate([input1,x,c1])
    return x


def DBB_block(input1, input2, num_filters):
    x = DB_block(input1, num_filters)

    x = layers.Conv2D(num_filters, (1, 1), padding="same")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.BatchNormalization()(x)

    # Rescale input2 to have the same filter dimension as x #
    input2 = layers.Conv2D(num_filters, (1, 1), padding="same")(input2)
    ###################################################
    x = layers.Add()([x, input2])

    x = layers.Conv2D(num_filters, (1, 1), padding="same")(x)
    # Reduce size of x to match its original size before Conv2DTranspose#
    x = layers.MaxPooling2D((2,2))(x)
    ###################################################
    return x


def SCAB_block(input1, input2, num_filters, final=False):
    x = layers.Conv2D(num_filters, (1, 1), padding="same")(input1)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(num_filters, (1, 1), padding="same")(x)

    x = layers.Activation("sigmoid")(x)

    if final:
        input2 = layers.Conv2D(num_filters, (1, 1), padding="same")(input2)

    x = layers.Multiply()([x, input2])

    x = layers.concatenate([input1,x])

    c1 = layers.Conv2D(num_filters, (1, 1), padding="same")(x)

    x = layers.GlobalAveragePooling2D(keepdims=True)(c1)

    x = layers.Conv2D(num_filters, (1, 1), padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(num_filters, (1, 1), padding="same")(x)

    x = layers.Activation("sigmoid")(x)

    x = layers.Multiply()([x, c1])

    x = layers.Add()([x, c1])
    return x


def conv_block(input, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same")(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def deconv_block(input, num_filters):
    x = layers.Conv2DTranspose(num_filters, 1, padding="same")(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def decoder_block(input, skip_output, num_filters, final=False):
    x = conv_block(input, num_filters)
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = SCAB_block(x, skip_output, num_filters, final)
    x = DB_block(x, num_filters)
    return x


def bottleneck(input):
    x = layers.MaxPooling2D((2,2))(input)
    #x = layers.GlobalAveragePooling2D()(x)
    return x


def EfficientNetB4_CTUnet(input_shape=(512, 512, 3)):
    # Input
    inputs = layers.Input(input_shape)

    # Loading pre trained model
    EffNetB4 = EfficientNetB4(include_top=False, weights="imagenet", input_tensor=inputs)

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
    b1 = bottleneck(res4)  # 16 x 16

    # Decoder
    d1 = decoder_block(b1, skip4, 512)  # 32 x 32
    d2 = decoder_block(d1, skip3, 256)    # 64 x 64
    d3 = decoder_block(d2, skip2, 128)   # 128 x 128
    d4 = decoder_block(d3, skip1, 64)   # 256 x 256
    d5 = decoder_block(d4, skip0, 32, True)   # 512 x 512

    # Output
    outputs = conv_block(d5, 16)
    outputs = deconv_block(outputs, 16)
    outputs = conv_block(outputs, 16)
    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(outputs)

    model = Model(inputs, outputs, name="EfficientNetB4_CTU-Net")
    model.compile(optimizer=optimizers.Adam(learning_rate=0.000015), loss=[dice_coef_loss], metrics=[jaccard_coef, 'accuracy'])
    return model


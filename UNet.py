from tensorflow.keras import models, layers, optimizers, callbacks, losses


def basic_unet(input_size=(512, 512, 3)):
    # Encoder Part
    # Layer 1
    inputs = layers.Input(input_size)
    inputs_rescaled = layers.Lambda(lambda x: x / 255)(inputs) # Rescale input pixel values to floating point values
    c1 = layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(inputs_rescaled)
    c1 = layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c1)
    p1 = layers.MaxPooling2D((2,2))(c1)

    # Layer 2
    c2 = layers.Conv2D(128, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p1)
    c2 = layers.Conv2D(128, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c2)
    p2 = layers.MaxPooling2D((2,2))(c2)

    # Layer 3
    c3 = layers.Conv2D(256, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p2)
    c3 = layers.Conv2D(256, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c3)
    p3 = layers.MaxPooling2D((2,2))(c3)

    # Layer 4
    c4 = layers.Conv2D(512, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p3)
    c4 = layers.Conv2D(512, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c4)
    p4 = layers.MaxPooling2D((2,2))(c4)

    # Layer 5
    c5 = layers.Conv2D(1024, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p4)
    c5 = layers.Conv2D(1024, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c5)


    # Decoder Part
    # Layer 6
    u6 = layers.Conv2DTranspose(512, (2,2), strides=(2,2), padding = 'same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u6)
    c6 = layers.Conv2D(512, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c6)

    # Layer 7
    u7 = layers.Conv2DTranspose(256, (2,2), strides=(2,2), padding = 'same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u7)
    c7 = layers.Conv2D(256, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c7)

    # Layer 8
    u8 = layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding = 'same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u8)
    c8 = layers.Conv2D(128, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c8)

    # Layer 9
    u9 = layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding = 'same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u9)
    c9 = layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c9)

    outputs = layers.Conv2D(1, (1,1), activation = 'sigmoid')(c9)

    # Compiling model
    model = models.Model([inputs], [outputs])
    model.compile(optimizer = 'adam', loss = losses.BinaryCrossentropy(from_logits=False), metrics = ['accuracy'])
    return model


def unet_dropout(input_size=(512, 512, 3)):
    # Encoder Part
    # Layer 1
    inputs = layers.Input(input_size)
    inputs_rescaled = layers.Lambda(lambda x: x / 255)(inputs) # Rescale input pixel values to floating point values
    c1 = layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(inputs_rescaled)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c1)
    p1 = layers.MaxPooling2D((2,2))(c1)

    # Layer 2
    c2 = layers.Conv2D(128, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(128, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c2)
    p2 = layers.MaxPooling2D((2,2))(c2)

    # Layer 3
    c3 = layers.Conv2D(256, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p2)
    c3 = layers.Dropout(0.1)(c3)
    c3 = layers.Conv2D(256, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c3)
    p3 = layers.MaxPooling2D((2,2))(c3)

    # Layer 4
    c4 = layers.Conv2D(512, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p3)
    c4 = layers.Dropout(0.1)(c4)
    c4 = layers.Conv2D(512, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c4)
    p4 = layers.MaxPooling2D((2,2))(c4)

    # Layer 5
    c5 = layers.Conv2D(1024, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p4)
    c5 = layers.Dropout(0.1)(c5)
    c5 = layers.Conv2D(1024, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c5)


    # Decoder Part
    # Layer 6
    u6 = layers.Conv2DTranspose(512, (2,2), strides=(2,2), padding = 'same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u6)
    c6 = layers.Dropout(0.1)(c6)
    c6 = layers.Conv2D(512, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c6)

    # Layer 7
    u7 = layers.Conv2DTranspose(256, (2,2), strides=(2,2), padding = 'same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u7)
    c7 = layers.Dropout(0.1)(c7)
    c7 = layers.Conv2D(256, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c7)

    # Layer 8
    u8 = layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding = 'same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u8)
    c8 = layers.Dropout(0.1)(c8)
    c8 = layers.Conv2D(128, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c8)

    # Layer 9
    u9 = layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding = 'same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u9)
    c9 = layers.Dropout(0.1)(c9)
    c9 = layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c9)

    outputs = layers.Conv2D(1, (1,1), activation = 'sigmoid', padding = "same")(c9)

    # Compiling model
    model = models.Model([inputs], [outputs])
    #model.compile(optimizer = 'adam', loss = losses.BinaryFocalCrossentropy(gamma = 5.0), metrics = ['accuracy'])
    model.compile(optimizer = 'adam', loss = losses.BinaryCrossentropy(from_logits=False), metrics = ['accuracy'], shuffle = true)
    return model


from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Activation, Dense, Flatten
from tensorflow.keras.models import Model

def build():
    input_shape = (32, 32, 3)
    img_input = Input(shape=input_shape)

    x = img_input
    x = Conv2D(64, (3, 3),
                            strides=(1, 1),
                            padding='same')(x)
    x = Activation('approx_activation')(x)

    x = Conv2D(64, (3, 3),
                            strides=(1, 1),
                            padding='same')(x)
    x = Activation('approx_activation')(x)

    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3),
                            strides=(1, 1),
                            padding='same')(x)
    x = Activation('approx_activation')(x)

    x = Conv2D(64, (3, 3),
                            strides=(1, 1),
                            padding='same')(x)
    x = Activation('approx_activation')(x)

    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3),
                            strides=(1, 1),
                            padding='same')(x)
    x = Activation('approx_activation')(x)

    x = Conv2D(64, (1, 1),
                            strides=(1, 1),
                            padding='valid')(x)
    x = Activation('approx_activation')(x)

    x = Conv2D(16, (1, 1),
                            strides=(1, 1),
                            padding='valid')(x)
    x = Activation('approx_activation')(x)

    x = Flatten()(x)
    x = Dense(10, activation='softmax')(x)
    return Model(img_input, x)

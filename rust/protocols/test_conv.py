import tensorflow as tf
import keras
import sys
from keras.layers import Input, Conv2D, Activation, AveragePooling2D, Flatten, Dense, Lambda
from keras.utils.generic_utils import get_custom_objects
from keras.models import Model
from keras import backend as K
import itertools
import numpy as np


def approx_activation(x):
    return Activation('relu')(x)
    #return .1992 + .5002*x + .1997*x**2

get_custom_objects().update({'approx_activation': keras.layers.Activation(approx_activation)})
np.set_printoptions(threshold=sys.maxsize)

# Build minionn test model
def build_mini():
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
    #x = Lambda((lambda x: tf.Print(x, [x], summarize=-1)))(x)
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
    x = Dense(10)(x)
    ##x = Lambda((lambda x: K.print_tensor(x, message="Pre-softmax: ")))(x)
    return Model(img_input, x)

# Build test model
def build_test():
    img_input = Input(shape=(4, 4, 1))
    x = img_input                                      
    x = Conv2D(3, (3, 3), padding='same')(x)
    x = AveragePooling2D((2, 2), 2, padding='valid')(x)
    x = Activation('relu')(x)
    x = Conv2D(16, (1, 1), padding='valid')(x)

    x = Activation('approx_activation')(x)

    x = Flatten()(x)
    x = Dense(10, bias_initializer='random_uniform')(x)

    model = Model(img_input, x)

    return model

#model = build_test()
model = build_mini()
model.load_weights("/home/ryan/research/delphi/system/python/minionn/pretrained/mini_relu")

### Set random int weights for easier debugging
#for layer in model.layers:
#    if "conv2d" in layer.name:
#        A, b = layer.get_weights()
#        layer.set_weights([np.random.randint(-2, 2, np.prod(A.shape)).reshape(A.shape),
#                           np.random.randint(-2, 2, A.shape[3])])
    #if "dense" in layer.name:
    #    A, b = layer.get_weights()
    #    layer.set_weights([np.random.randint(0, 1, np.prod(A.shape)).reshape(A.shape),
    #                       np.random.randint(0, 2, A.shape[3])])


# All the weights need to be flattened into a single array for rust interopt
network_weights = np.array([])
for i, layer in enumerate(model.layers):
    if "conv2d" in layer.name:
        A, b = layer.get_weights()
        # Keras stores the filter as the first two dimensions and the
        # channels as the 3rd and 4th. PyTorch does the opposite so flip
        # everything around
        _, _, inp_c, out_c = A.shape
        py_tensor = [[A[:, :, i, o] for i in range(inp_c)] for o in range(out_c)]
        A = np.array(py_tensor)
        #print(f"LAYER {i}: {A.shape} {b.shape}")
        #print(A)
        #print(b)
    elif "dense" in layer.name:
        A, b = layer.get_weights()
        A = A.T
        # Get the shape of last layer output to transform the FC
        # weights correctly since we don't flatten input to FC in Delphi
        inp_chans = 1
        for prev_i in range(i, 0, -1):
            layer_name = model.layers[prev_i].name
            if ("conv2d" in layer_name) or ("average_pooling2d" in layer_name) or prev_i == 0:
                inp_chans = model.layers[prev_i].output_shape[3]
                break
        # Remap to PyTorch shape
        fc_h, fc_w = A.shape
        channel_cols = [np.hstack((A[:, [i]] for i in range(chan, fc_w, inp_chans)))
            for chan in range(inp_chans)]
        A = np.hstack(channel_cols)
        #print(f"LAYER {i}: {A.shape} {b.shape}")
        #print(A)
        #print(b)
    else:
        continue
    layer_weights = np.concatenate((A.flatten(), b.flatten()))
    network_weights = np.concatenate((network_weights, layer_weights))

np.save("test.npy", network_weights.astype(np.float64))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# TODO
#inp = np.ones((1, 32, 32, 3))
inp = np.load("../../python/minionn/test_data/image_0.npy").reshape(3, 32, 32)
#inp = np.expand_dims(np.array([inp[c, :, :] for c in range(3)]).reshape(32, 32, 3), axis=0)
inp = np.expand_dims(np.array([[inp[:, i, j] for j in range(32)] for i in range(32)]).reshape(32, 32, 3), axis=0)
# Static random input for easier debugging
#inp = np.array([3, 0, 1, 1, 2, 0, 2, 1, 1, 3, 2, 1, 1, 0, 1, 0, 1, 1, 3, 1, 3, 1, 1,
#    1, 0, 2, 1, 2, 1, 1, 3, 0, 2, 0, 2, 2, 2, 0, 0, 1, 3, 0, 3, 1, 0, 2, 3, 0,
#    1, 1, 3, 3, 3, 3, 3, 3, 2, 0, 2, 2, 2, 1, 2, 1, 2, 0, 2, 2, 0, 3, 0, 1, 2,
#    3, 2]).reshape(1, 5, 5, 3)

x = model.predict(inp)
print(x)
s = softmax(x[0])
print(s)
print(f"Out: {np.argmax(s)}")

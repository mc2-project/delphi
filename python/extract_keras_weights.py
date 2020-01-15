"""
Serialize Keras model weights for loading in Rust
"""
import argparse
import numpy as np
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from minionn import minionn_model
from resnet import resnet32_model
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import get_custom_objects
from os import path

def build_model(model_builder, approx):
    """Construct model following given architecture and approx layers"""
    ACTIVATION_NUM = 0
    def approx_activation(x):
        nonlocal ACTIVATION_NUM
        if ACTIVATION_NUM in approx:
            x = .1992 + .5002*x + .1997*x**2
            print(f"ACTIVATION {ACTIVATION_NUM}: Approx")
        else:
            x = tf.keras.activations.relu(x, max_value=6.0)
            print(f"ACTIVATION {ACTIVATION_NUM}: ReLU")
        ACTIVATION_NUM += 1
        return x
    get_custom_objects().update({'approx_activation': tf.keras.layers.Activation(approx_activation)})

    # Load the model_builder module
    model = model_builder.build()
    # This is necessary since quantization will rebuild the network
    ACTIVATION_NUM = 0
    return model


def test_accuracy(model, dataset):
    """Evaluate the accuracy of given model on provided dataset"""
    # Normalize dataset
    (_, _), (x_test, y_test) = dataset.load_data()
    x_test = x_test.astype('float32')
    x_test /= 255
    # Test accuracy
    correct = 0
    for i in range(len(y_test)):
        correct += 1 if y_test[i] == np.argmax(model.predict(x_test[[i]])) else 0
    return 100 * (correct / len(y_test))


def quantize(model):
    """Quantize given model with TFLite and convert back to Keras"""
    # Run TFLite quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_quant_model = converter.convert()
   
    # Convert quantized weights back to keras model
    interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    all_layers_details = interpreter.get_tensor_details() 

    for quant_layer in all_layers_details:   
        name = quant_layer['name']
        if 'model' in name:
            layer_name = re.split('/', name)[1]
            layer_part = re.split('/', name)[2]
            shape = quant_layer['shape']
            quant = quant_layer['quantization']
            # If layer is a conv or dense reshape and load the weights
            if ("conv2d" in layer_name) or ("dense" in layer_name):
                layer = model.get_layer(layer_name)
                weights = interpreter.get_tensor(quant_layer['index'])
                if quant[0] != 0.0:
                    weights = (weights - quant[1]) * quant[0]
                if "bias" in layer_part:
                    layer.set_weights([layer.get_weights()[0], weights.flatten()])
                elif "Conv2D" in layer_part:
                    # TFLite stores the output channels as the first dimension.
                    # Keras does the opposite so flip everything around
                    weights = np.array([[[weights[:, x, y, inp_c]
                                for inp_c in range(shape[3])]
                                for y in range(shape[2])]
                                for x in range(shape[1])])
                    layer.set_weights([weights, layer.get_weights()[1]])
                elif "MatMul" in layer_part:
                    layer.set_weights([weights.T, layer.get_weights()[1]])


def serialize_weights(model, save_path):
    """Serialize Keras model into flattened numpy array in correct shape for
    Pytorch in Rust"""
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
        elif "dense" in layer.name:
            A, b = layer.get_weights()
            A = A.T
            # Get the shape of last layer output to transform the FC
            # weights correctly since we don't flatten input to FC in Delphi
            inp_chans = 1
            for prev_i in range(i, 0, -1):
                layer_name = model.layers[prev_i].name
                if ("global" in layer_name):
                    inp_chans = model.layers[prev_i].output_shape[1]
                    break
                if ("conv2d" in layer_name) or ("average_pooling2d" in layer_name) or prev_i == 0:
                    inp_chans = model.layers[prev_i].output_shape[3]
                    break
            # Remap to PyTorch shape
            fc_h, fc_w = A.shape
            channel_cols = [np.hstack([A[:, [i]] for i in range(chan, fc_w, inp_chans)])
                            for chan in range(inp_chans)]
            A = np.hstack(channel_cols)
        else:
            continue
        layer_weights = np.concatenate((A.flatten(), b.flatten()))
        network_weights = np.concatenate((network_weights, layer_weights))
    np.save(os.path.join(save_path,"model.npy"), network_weights.astype(np.float64))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=int, help='<REQUIRED> Use Minionn (0) or Resnet32 (1)')
    parser.add_argument('-w', '--weights_path', required=True, type=str,
                        help='<REQUIRED> Path to model weights')
    parser.add_argument('-s', '--save_path', required=False, type=str,
                        help='Path to save (default is cwd)')
    parser.add_argument('-q', '--quantize', required=False, action="store_true",
                        help='Whether to quantize model')
    parser.add_argument('-t', '--test_acc', required=False, action="store_true",
                        help='Test accuracy of network')
    parser.add_argument('-a', '--approx', nargs='+', type=int, required=False,
                        help='Set approx layesrs')
    args = parser.parse_args()

    # Select correct model and dataset
    model_builder = resnet32_model if args.model else minionn_model
    dataset = None
    if args.test_acc:
        dataset = cifar100 if args.model else cifar10

    # Resolve paths 
    weights_path = path.abspath(args.weights_path)
    save_path = path.abspath(args.save_path) if args.save_path else os.path.curdir

    # Build model
    model = build_model(model_builder, args.approx or [])
    model.load_weights(weights_path)
    if args.test_acc:
        acc = test_accuracy(model, dataset)
        print(f"Accuracy: {acc}%\n")
    if args.quantize:
        quantize(model)
        model.save_weights(os.path.join(save_path, "model_quant.h5"))
        if args.test_acc:
            acc = test_accuracy(model, dataset)
            print(f"Quantized Accuracy: {acc}%")
    
    # Serialize weights for Rust
    serialize_weights(model, save_path)

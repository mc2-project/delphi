"""
Evaluate Keras model accuracy on generated test images
"""
import argparse
import numpy as np
import os
import random
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from importlib.util import spec_from_file_location, module_from_spec
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
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
            x = tf.keras.activations.relu(x)
            #x = tf.keras.activations.relu(x, max_value=6.0)
            print(f"ACTIVATION {ACTIVATION_NUM}: ReLU")
        ACTIVATION_NUM += 1
        return x
    get_custom_objects().update({'approx_activation': Activation(approx_activation)})
    # Load the model_builder module
    model = model_builder.build()
    # This is necessary since quantization will rebuild the network
    ACTIVATION_NUM = 0
    return model


def generate_images(image_path, num_images):
    """Sample and save a random set of num_images images"""
    # Normalize dataset
    (_, _), (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype("float32")
    x_test /= 255
    # Sample and save images 
    sample_set = random.sample(list(range(len(x_test))), num_images)
    images = [x_test[i] for i in sample_set]
    classes = [y_test[i] for i in sample_set]
    for i, (img, cls) in enumerate(zip(images, classes)):
        # Reshape image for rust
        _, _, chans = img.shape
        rust_tensor = np.array([[img[:, :, c] for c in range(chans)]])
        np.save(os.path.join(image_path, f"image_{i}.npy"), rust_tensor.flatten().astype(np.float64))
    np.save(path.join(image_path, f"classes.npy"), np.array(classes).flatten().astype(np.int64))


def test_network(model, image_path):
    """Gets inference results from given network"""
    # Load image classes
    classes = np.load(path.join(image_path, "classes.npy"))
    # Run inference on all images and track plaintext predictions
    correct = []
    for i in range(len(classes)):
        # Load image and reshape to proper shape
        image = np.load(path.join(image_path, f"image_{i}.npy")).reshape(3, 32, 32)
        image = np.array([[image[:, i, j] for j in range(32)] for i in range(32)]).reshape(32, 32, 3)
        prediction = np.argmax(model.predict(np.expand_dims(image, axis=0))) 
        correct += [1] if prediction == classes[i] else [0]
    # Save prediction results    
    np.save(path.join(image_path, "plaintext.npy"), np.array(correct))
    return 100 * (sum(correct) / len(classes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=int, help='<REQUIRED> Use Minionn (0) or Resnet32 (1)')
    parser.add_argument('-w', '--weights_path', required=True, type=str,
                        help='<REQUIRED> Path to model weights')
    parser.add_argument('-a', '--approx', nargs='+', type=int, required=False,
                        help='Set approx layesrs')
    parser.add_argument('-i', '--image_path', required=False, type=str,
                        help='Path to place images')
    parser.add_argument('-g', '--generate', required=False, type=int,
                        help='How many images to generate (default 0)')
    args = parser.parse_args()

    # Load the correct model and dataset
    dataset = cifar100 if args.model else cifar10
    if args.model:
        model_path = path.abspath("../../../../python/resnet/resnet32_model.py")
    else:
        model_path = path.abspath("../../../../python/minionn/minionn_model.py")
    
    spec = spec_from_file_location(path.basename(model_path), model_path)
    model_builder = module_from_spec(spec)
    sys.modules[path.basename(model_path)] = model_builder
    spec.loader.exec_module(model_builder)

    # Resolve paths 
    weights_path = path.abspath(args.weights_path)
    image_path = path.abspath(args.image_path) if args.image_path else os.path.curdir
    os.makedirs(image_path, exist_ok=True)

    # Build model
    model = build_model(model_builder, args.approx or [])
    model.load_weights(weights_path)

    # Sample images
    if args.generate:
        generate_images(image_path, args.generate)
    
    print(f"Accuracy: {test_network(model, image_path)}%")

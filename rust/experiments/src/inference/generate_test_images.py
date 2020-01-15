"""
Evaluate Keras model accuracy on generated test images
"""
import argparse
import numpy as np
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from tensorflow.keras.datasets import cifar10
from os import path

def generate_image(image_path):
    """Sample and save a random set of num_images images"""
    # Normalize dataset
    (_, _), (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype("float32")
    x_test /= 255
    # Sample and save images 
    idx = random.choice(range(len(x_test)))
    image = x_test[idx]
    classification = y_test[idx]

    _, _, chans = image.shape
    rust_tensor = np.array([[image[:, :, c] for c in range(chans)]])
    np.save(os.path.join(image_path, f"image.npy"), rust_tensor.flatten().astype(np.float64))
    np.save(path.join(image_path, f"class.npy"), np.array(classification).flatten().astype(np.int64))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', required=False, type=str,
                        help='Path to place images (default cwd)')
    args = parser.parse_args()

    # Resolve paths 
    image_path = path.abspath(args.image_path) if args.image_path else os.path.curdir
    os.makedirs(image_path, exist_ok=True)

    # Sample image
    generate_image(image_path) 

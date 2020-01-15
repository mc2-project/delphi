#!/usr/bin/env python
import argparse
import numpy as np
import os
import sys
import pickle
import errno
import itertools
import random
from os import path

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as K

import ray
from ray.tune import grid_search, run, sample_from
from ray.tune import Trainable
from ray.tune.schedulers import PopulationBasedTraining

import minionn_model
from minionn_model import Activation

num_classes = 10


def sample_blocks(num_layers, num_approx):
    """Generate approx block permutations by sampling w/o replacement. Leave the
    first and last blocks as ReLU"""
    perms = []
    for _ in range(1000):
        perms.append(sorted(random.sample(list(range(0,num_layers)), num_approx)))
    # Remove duplicates
    perms.sort()
    return [p for p,_ in itertools.groupby(perms) if len(p) == num_approx]


def performance(accuracy, approx_activations, total_activations):
    return accuracy * (1 + approx_activations / total_activations)


class RatioCallback(Callback):
    def __init__(self, CURRENT_EPOCH, swap_period, total, RATIO):
        self.cur = CURRENT_EPOCH
        self.total = total
        self.swap_period = swap_period
        self.ratio = RATIO

    def on_epoch_begin(self, epoch, logs=None):
        print("Current epoch = ", epoch)
        print("Total = ", self.total)
        if epoch > self.total:
            K.set_value(self.ratio, np.array(0))
            print("Ratio = ", np.array(0))
        else:
            ratio = np.array((self.total-epoch)/self.swap_period) if self.swap_period \
		    else np.array(0)
            print("Ratio = ", ratio)
            K.set_value(self.ratio, ratio)
        K.set_value(self.cur, epoch)


class Cifar10Model(Trainable):
    def _pop_activation_count(self):
        """Calculate the exact number of activations for calculating
        performance"""
        activation_map = []
        for layer in self.model.layers:
            if "activation" in layer.name:
                shape = self.model.get_layer(layer.name).input_shape
                activation_map.append(shape[1]*shape[2]*shape[3])
        self.total_activations = sum(activation_map)
        self.approx_activations = sum([activation_map[i] for i in self.approx])

    def _read_data(self):
        import tensorflow as tf
        tf.keras.backend.set_image_data_format('channels_last')
        from tensorflow.keras.datasets import cifar10

        # The data, split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # Convert class vectors to binary class matrices.
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        x_train = x_train.astype("float32")
        x_train /= 255
        x_test = x_test.astype("float32")
        x_test /= 255

        # Split the training set into 20% validation set
        x_val = x_train[:10000]
        y_val = y_train[:10000]
        x_train = x_train[10000:]
        y_train = y_train[10000:]

        return (x_train, y_train), (x_test, y_test), (x_val, y_val)


    def _build_model(self):
        import tensorflow as tf
        LAYER_NUM = 0
        self.cur = K.variable(0)
        self.ratio = K.variable(1)
        
        def approx_activation(x):
            nonlocal LAYER_NUM
            relu = tf.keras.activations.relu(x, max_value=6.0)
            if LAYER_NUM in self.approx:
                approx = .1992 + .5002*x + .1997*x**2
                x = (1-self.ratio)*approx + self.ratio*relu
                print(f"layer {LAYER_NUM}: poly")
            else:
                x = relu
                print(f"layer {LAYER_NUM}: relu")
            LAYER_NUM += 1
            return x
        
        get_custom_objects().update({'approx_activation': Activation(approx_activation)})

        model = minionn_model.build()

        # Have optimizer clip gradients
        # TODO: Experiment with l2 clipping
        opt = SGD(lr=self.lr, momentum=self.mom, clipvalue=2.0)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=opt,
            metrics=["accuracy"])

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.compat.v1.global_variables_initializer())
        tf.compat.v1.keras.backend.set_session(sess)
        self.model = model

    def _setup(self, config):
        # TODO: Normalize paths
        self.train_data, self.test_data, self.val_data = self._read_data()
        self.reset = False
        # State passed in the config
        self.lr = config["lr"]
        self.mom = config["mom"]
        self.approx = config["approx"]
        self.baseline = config["baseline"]
        self.restore_path = config["restore_path"]
        self.swap_period = config["swap_period"] if self.restore_path else 0

        # State which needs to be checkpointed
        self.epoch = 0
        self.accuracy = 0
        self.performance = 0
        # Where to store model checkpoints
        self.path = os.path.join(self.logdir, "model_checkpoints")
        try:
            os.makedirs(self.path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        # self.total refers to the epoch when activation swapping should
        # be finished. A separate variable needs to be kept for restoring
        # models
        self.total = self.swap_period

        self._build_model()
        self._pop_activation_count()

        # Initially restore from the relu model if specified
        if self.restore_path:
            self._restore(self.restore_path)

    def _train(self):
        x_train, y_train = self.train_data
        x_test, y_test = self.test_data
        x_val, y_val = self.val_data

        # Apply a few small augmentations to input images
        aug_gen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False,
        )
        aug_gen.fit(x_train)
        gen = aug_gen.flow(
            x_train, y_train, batch_size=self.config["batch_size"])

        ratio = RatioCallback(self.cur, self.swap_period, self.total, self.ratio)

        self.model.fit_generator(
            generator=gen,
            steps_per_epoch=40000 // self.config["batch_size"],
            epochs=self.epoch + self.config["epochs"],
            validation_data=None,
            callbacks=[ratio],
            initial_epoch=self.epoch)

        self.epoch += self.config["epochs"]

        # Calculate validation loss/accuracy for PBT
        loss, accuracy = self.model.evaluate(x_val, y_val, verbose=0)
        _, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
       
        cur_perf = performance(accuracy,
                               self.approx_activations,
                               self.total_activations)

        # Only checkpoint if past swap phase
        if self.epoch > self.total and cur_perf > self.performance:
            self.accuracy = test_accuracy
            self.performance = cur_perf
            self._save(self.path)
        
        return {
            "accuracy": accuracy,
            "learning_rate": self.lr,
            "loss": loss,
            "accuracy_test": test_accuracy,
            "momentum": self.mom,
            "approx": self.approx,
            "total": self.total,
            "path": self.path,
            "epoch": self.epoch,
            "ratio": max(0, (self.total-self.epoch)/self.swap_period) 
                     if self.swap_period else 0.0,
            "performance": cur_perf,
            }


    def _save(self, checkpoint_dir):
        file_path = checkpoint_dir + "/model"
        # State to pickle
        to_save = {
                'epoch': self.epoch,
                'total': self.total,
                # Variable to make sure ratio remains same if transferring
                'transfer': self.baseline,
                'approx': self.approx,
                'accuracy': self.accuracy,
                'performance': self.performance,
                }
        with open(file_path + "_state", "wb") as f:
            f.write(pickle.dumps(to_save))
        # Save weights
        self.model.save_weights(file_path, save_format='h5')
        return file_path
    

    def _restore(self, path):
        # Restore pickled state
        with open(path + "_state", "rb") as f:
            state = pickle.loads(f.read())
        self.epoch = state['epoch']

        # If we want to do approximation swapping off of the 
        # restored model reset the total period
        if state['transfer']:
            self.total = self.epoch + self.swap_period
        else:
            self.total = state['total']
        # If the approximation layers have changed on restoration
        # we need to rebuild the model
        if self.reset:
            self._build_model()
            self._pop_activation_count()
            self.reset = False
        self.model.load_weights(path)


    def reset_config(self, new_config):
        # State passed in the config. Other params shouldn't change
        self.lr = new_config["lr"]
        self.mom = new_config["mom"]
        # If the approximation layers are changed trigger a model reset
        if self.approx != new_config["approx"]:
            self.reset = True
        self.approx = new_config["approx"]
        return True


if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--test_name', required=True, type=str,
                        help='Filename of experiment data')
    parser.add_argument('-d', '--dir', required=True, type=str,
                        help='Directory for experiment data')
    parser.add_argument('-a', '--approx', required=True, type=int,
                        help='Number of approx (< s)')
    parser.add_argument('-b', '--baseline', action='store_true',
                        help='Whether these are baseline models')
    parser.add_argument('-r', '--restore', required=False, type=str,
                        help='Model to restore from')

    args = parser.parse_args()
    restore_path = path.abspath(args.restore)

    train_spec = {
        "resources_per_trial": {
            "cpu": 2,
            "gpu": 0
        },
        "stop": {
            "training_iteration": 30,
        },
        "config": {
            "epochs": 10,
            "batch_size": 128,
            "mom": 0.9,
            "lr": sample_from(lambda _: random.choice([.1, .01, .001, .0001])),
            "approx": sample_from(lambda _: random.choice(sample_blocks(7, args.approx))),
            "swap_period": 100,
            "baseline": args.baseline,
            "restore_path": restore_path,
        },
        "num_samples": 10
    }

    ray.init()

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        reward_attr="performance",
        perturbation_interval=10,
        hyperparam_mutations={
            "lr": lambda: random.uniform(.0001, 1),
            "mom": lambda: random.uniform(.5, 1),
        })

    run(Cifar10Model,
        name=args.test_name,
        local_dir=args.dir,
        scheduler=pbt,
        resume=False,
        reuse_actors=True,
        **train_spec)

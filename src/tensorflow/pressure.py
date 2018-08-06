import math
import tensorflow as tf
import numpy as np

from .pressure_model_fn import pressure_model_fn
from .fast_predict import FastPredict

class Pressure():
    def __init__(self):
        params = {}

        # ----- Input -----
        # Grid parameters
        params["p_ll"] = (0, 0)
        params["p_ur"] = (15, 15)
        params["xRes"] = 0.25
        params["yRes"] = 0.25
        params["h"]    = 0.5
        params["N_x"]  = round((params["p_ur"][0]-params["p_ll"][0] + params["xRes"]) / params["xRes"])
        params["N_y"]  = round((params["p_ur"][1]-params["p_ll"][1] + params["yRes"]) / params["yRes"])

        # Attributes that the model uses
        params["body_channels"] = ["mass", "vx", "vy", "omega"]
        params["contact_channels"] = ["nx", "ny"]
        params["feature_channels"] = params["body_channels"] + params["contact_channels"]
        params["N_c"] = len(params["body_channels"]) + len(params["contact_channels"])


        # ----- Model Parameters -----
        # Training parameters
        params["eta"]   = 0.001
        params["initializer_stddev"] = 0.1

        # A weight multiplied with the loss from nodes where the label is zero
        params["use_zero_weight"] = True
        params["zero_weight"] = 0.25

        # A weight multiplied with the l2-regularization loss
        params["lmbda"] = 0.01


        # ----- Layer Parameters -----
        f = 16
        k = 4

        # Input layer

        # Initial convolution
        params["input_conv_filters"] = f
        params["input_conv_kernel"]  = k

        # Pooling layer
        params["n_poolings"]  = 2
        params["pool_size"]   = [2 for _ in range(params["n_poolings"])]
        params["pool_stride"] = [2 for _ in range(params["n_poolings"])]

        # First round of convolution layers
        params["n_convolutions_1"] = 2
        params["conv_1_filters"] = [f for _ in range(params["n_convolutions_1"])]
        params["conv_1_kernel"]  = [k for _ in range(params["n_convolutions_1"])]

        # Second round of convolution layers
        params["n_convolutions_2"] = 2
        params["conv_2_filters"] = [f/2, f/4]
        params["conv_2_kernel"]  = [1 for _ in range(params["n_convolutions_2"])]


        # ----- Model config -----
        # Path to model storage
        params["model_path"] = "./src/tensorflow/models/pressure/"

        # Tells the model whether or not to use GPUs, and only keep 1 checkpoint
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(
            device_count = {"GPU": 0},   # Switch from GPU to CPU
            gpu_options = gpu_options,
        )
        runconfig = tf.estimator.RunConfig(
            keep_checkpoint_max = 1,
            session_config = config
        )

        self.params = params
        self.estimator = tf.estimator.Estimator(
            model_fn = pressure_model_fn,
            model_dir = params["model_path"],
            params = params,
            config = runconfig
        )

        # We create a FastPredict object, which is used for making predictions
        # without having to reload the model each time
        # First we create an input fn taking a generator as input
        def pred_input_fn(generator):

            def _inner_input_fn():
                dataset = tf.data.Dataset().from_generator(
                    generator,
                    output_types=(tf.float32),
                ).batch(1)
                iterator = dataset.make_one_shot_iterator()
                features = iterator.get_next()
                return {'x': features}

            return _inner_input_fn

        self.FastPredict = FastPredict(self.estimator, pred_input_fn)


    def train(self, features, labels, train_params):
        # We create the input function
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {"x": features},
            y = labels,
            batch_size = train_params["batch_size"],
            num_epochs = train_params["num_epochs"],
            shuffle = True
        )

        # We train
        self.estimator.train(
            input_fn=train_input_fn
        )


    def evaluate(self, features, labels):
        # We create the input function
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {"x": features},
            y = labels,
            num_epochs = 1,
            shuffle = False
        )

        # We do the evaluation
        eval_results = self.estimator.evaluate(input_fn=eval_input_fn)

        return eval_results


    def predict(self, features):
        # We do the predictions
        pred_results = self.FastPredict.predict(features)

        # For some reason the result is some unsubscriptable iterator object
        # There should only be a single pred_dict
        for pred_dict in pred_results:
            ni = pred_dict["ni"]
            ti = pred_dict["ti"]

        return (ni, ti)

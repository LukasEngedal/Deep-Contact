import math
import tensorflow as tf
import numpy as np

from .peak_model_fn import peak_model_fn
from .fast_predict import FastPredict

class Peak():
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
        N = params["N_x"] * params["N_y"]

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
        # Input layer
        params["input_forward"] = True


        # Convolution layers
        params["convolution_layers"]  = 4
        params["filters"]             = [8, 16, 32, 64]
        params["kernel_size"]         = [4 for _ in range(params["convolution_layers"])]

        params["pool_size"]    = [2 for _ in range(params["convolution_layers"])]
        params["pool_stride"]  = [2 for _ in range(params["convolution_layers"])]

        # Decides whether to forward convolution output to the dense layers directly as well
        # Set to [False, False, ..., True] to only use last convolution output
        params["conv_forward"] = [True for _ in range(params["convolution_layers"])]


        # Dense layers
        params["dense_layers"] = 2
        params["dense_units"] = [6*N, 4*N]

        # Dropout actually happens before the dense layer
        params["dense_dropout"] = [True for _ in range(params["dense_layers"])]
        params["dense_dropout_rate"] = [0.4 for _ in range(params["dense_layers"])]


        # Output layer
        params["output_units"] = 2 * N
        params["output_grids"] = 2



        # ----- Model config -----
        # Path to model storage
        params["model_path"] = "./src/tensorflow/models/peak/"

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
            model_fn = peak_model_fn,
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

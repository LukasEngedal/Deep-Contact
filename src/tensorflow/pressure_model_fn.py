import numpy as np
import tensorflow as tf


# Custom convolution layer including batch normalization
def conv_layer(inputs, filters, kernel, initializer, regularizer, training):
    # Convolution Layer
    conv_output = tf.layers.conv2d(
        inputs      = inputs,
        filters     = filters,
        kernel_size = kernel,
        padding     = "same",
        activation  = tf.nn.relu,
        kernel_initializer = initializer,
        kernel_regularizer = regularizer,
    )

    # Batch normalization
    conv_norm = tf.layers.batch_normalization(
        inputs = conv_output,
        training = training
    )

    return conv_norm



# Our main cnn model function
def pressure_model_fn(features, labels, mode, params):
    """Model function for Pressure CNN Model"""
    training = mode == tf.estimator.ModeKeys.TRAIN

    kernel_regularizer = tf.contrib.layers.l2_regularizer(params["lmbda"])
    kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=params["initializer_stddev"])

    # Input Layer
    inputs = tf.reshape(features["x"], [-1, params["N_x"], params["N_y"], params["N_c"]])


    # Initial convolution of input
    conv_0 = conv_layer(
        inputs      = inputs,
        filters     = params["input_conv_filters"],
        kernel      = params["input_conv_kernel"],
        initializer = kernel_initializer,
        regularizer = kernel_regularizer,
        training    = training,
    )


    # Pooling layer
    branch_input = [conv_0]

    for i in range(params["n_poolings"]):
        pool_output = tf.layers.average_pooling2d(
            inputs    = branch_input[-1],
            pool_size = params["pool_size"][i],
            strides   = params["pool_stride"][i],
            padding   = "same"
        )
        branch_input.append(pool_output)


    # First round of convolution layers
    branch_output = []
    for b in range(params["n_poolings"]+1):
        conv_input = branch_input[b]

        for i in range(params["n_convolutions_1"]):
            conv_output = conv_layer(
                inputs      = conv_input,
                filters     = params["conv_1_filters"][i],
                kernel      = params["conv_1_kernel"][i],
                initializer = kernel_initializer,
                regularizer = kernel_regularizer,
                training    = training,
            )
            conv_input = conv_output

        branch_output.append(conv_output)


    # Upscaling and summing
    branch_sum = tf.zeros_like(branch_output[0])

    for b in range(params["n_poolings"]+1):
        branch_sum += tf.image.resize_images(
            images = branch_output[b],
            size   = [params["N_x"], params["N_y"]],
            method = tf.image.ResizeMethod.BILINEAR,
        )


    # Second round of convolution layers
    conv_input = branch_sum
    for i in range(params["n_convolutions_2"]):
        conv_output = conv_layer(
            inputs      = conv_input,
            filters     = params["conv_2_filters"][i],
            kernel      = params["conv_2_kernel"][i],
            initializer = kernel_initializer,
            regularizer = kernel_regularizer,
            training    = training,
        )
        conv_input = conv_output


    # Final output convolution layer
    output_output = tf.layers.conv2d(
        inputs      = conv_output,
        filters     = 2,
        kernel_size = 1,
        padding     = "same",
        activation  = None,
        kernel_initializer = kernel_initializer,
        kernel_regularizer = kernel_regularizer,
    )

    # We extract the two label attributes, flatten them and concat them
    ni, ti = tf.split(output_output, 2, axis=3)
    ni_flat = tf.layers.flatten(ni)
    ti_flat = tf.layers.flatten(ti)
    predictions = tf.concat([ni_flat, ti_flat], axis=1)


    # Prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions_dict = {
            "ni": ni_flat,
            "ti": ti_flat,
        }

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions_dict)


    # Multiply prediction grid nodes where the label is 0 with a special weight
    if params["use_zero_weight"]:
        flags = tf.to_float(tf.not_equal(labels, 0.0))
        weights = flags * (1-params["zero_weight"]) + params["zero_weight"]
        predictions = predictions * flags


    # Calculate Loss (for both TRAIN and EVAL modes)
    average_loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)
    regularization_loss = tf.losses.get_regularization_loss()

    loss = average_loss + regularization_loss


    # Add some more metrics
    absolute_error = tf.metrics.mean_absolute_error(labels=labels, predictions=predictions)
    rms_error = tf.metrics.root_mean_squared_error(labels=labels, predictions=predictions)
    tf.summary.scalar("MSE", average_loss)
    tf.summary.scalar("MAE", absolute_error[1])
    tf.summary.scalar("RMSE", rms_error[1])
    tf.summary.scalar("reg_loss", regularization_loss)


    # Configure the Training Op (for TRAIN mode)
    if training:
        learning_rate = params["eta"]

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "MAE": absolute_error,
        "RMSE": rms_error
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=average_loss, eval_metric_ops=eval_metric_ops)

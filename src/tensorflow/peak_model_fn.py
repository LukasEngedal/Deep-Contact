import numpy as np
import tensorflow as tf

# Custom dense layer including batch normalization
def dense_layer(inputs, units, initializer, regularizer, training):
    # Dense Layer
    dense_data = tf.layers.dense(
        inputs = inputs,
        units = units,
        activation = tf.nn.relu,
        kernel_initializer = initializer,
        kernel_regularizer = regularizer,
    )

    # Batch normalization
    dense_norm = tf.layers.batch_normalization(
        inputs = dense_data,
        training = training
    )

    return dense_norm



def peak_model_fn(features, labels, mode, params):
    """Model function for Peak CNN Model"""
    training = mode == tf.estimator.ModeKeys.TRAIN

    kernel_regularizer = tf.contrib.layers.l2_regularizer(params["lmbda"])
    kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=params["initializer_stddev"])

    # Input Layer
    input_data = tf.reshape(features["x"], [-1, params["N_x"], params["N_y"], params["N_c"]])

    # Forward raw input to dense layers
    if params["input_forward"]:
        dense_sum = input_data
    else:
        dense_sum = tf.zeros_like(input_data)


    # Convolution and max pooling layers
    conv_input = input_data
    for i in range(params["convolution_layers"]):
        # Convolution Layer
        conv_data = tf.layers.conv2d(
            inputs = conv_input,
            filters = params["filters"][i],
            kernel_size = params["kernel_size"][i],
            padding = "same",
            activation = tf.nn.relu,
            kernel_initializer = kernel_initializer,
            kernel_regularizer = kernel_regularizer,
        )

        # Batch normalization
        conv_norm = tf.layers.batch_normalization(
            inputs = conv_data,
            training = training
        )

        # Average Pooling Layer
        pool_data = tf.layers.average_pooling2d(
            inputs = conv_norm,
            pool_size = params["pool_size"][i],
            strides = params["pool_stride"][i],
            padding = "same",
        )
        conv_input = pool_data

        # Reduce channels in conv data, upscale and add to dense layer input sum
        if params["conv_forward"][i]:
            conv_small = tf.layers.conv2d(
                inputs = conv_norm,
                filters = params["N_c"],
                kernel_size = 1,
                padding = "same",
                activation = tf.nn.relu,
                kernel_initializer = kernel_initializer,
                kernel_regularizer = kernel_regularizer,
            )

            conv_up = tf.image.resize_images(
                images = conv_small,
                size   = [params["N_x"], params["N_y"]],
                method = tf.image.ResizeMethod.BILINEAR,
            )

            dense_sum += conv_up


    # We split and flatten input for dense layers
    grids = tf.split(dense_sum, params["N_c"], axis=3)
    grids_flat = [tf.layers.flatten(g) for g in grids]
    dense_input = tf.concat(grids_flat, axis=1)


    # Dense layers
    for i in range(params["dense_layers"]):
        # Dropout
        if params["dense_dropout"][i]:
            dense_dropout = tf.layers.dropout(
                inputs = dense_input,
                rate = params["dense_dropout_rate"][i],
                training = training
            )
        else:
            dense_dropout = dense_input

        dense_data = dense_layer(
            inputs = dense_dropout,
            units = params["dense_units"][i],
            initializer = kernel_initializer,
            regularizer = kernel_regularizer,
            training = training
        )
        dense_input = dense_data


    # Output layer
    predictions = tf.layers.dense(
        inputs = dense_data,
        units = params["output_units"],
        kernel_initializer = kernel_initializer,
        kernel_regularizer = kernel_regularizer,
        activation = None
    )


    # Prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        ni, ti = tf.split(predictions, params["output_grids"], axis=1)

        predictions_dict = {
            "ni": ni,
            "ti": ti
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

import keras.layers
import numpy as np
import random
import string
import tensorflow as tf
from .common import random_string

def extract_input_weight_bias(node, weights):
    W = None
    bias = None
    input_name = None
    for node_input in node.input:
        if node_input.endswith('weight'):
            W = weights[node_input]
        elif node_input.endswith('bias'):
            bias = weights[node_input]
        else:
            input_name = node_input
    return input_name, W, bias, bias is not None


def convert_conv(node, params, layers, weights, node_name):
    """
    Convert convolution layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting convolution ...')

    input_name, W, bias, has_bias = extract_input_weight_bias(node, weights)

    if len(W.shape) == 5:  # 3D conv
        raise NotImplementedError('Not implemented')

    elif len(W.shape) == 4:  # 2D conv
        if params['pads'][0] > 0 or params['pads'][1] > 0:
            padding_name = node_name + '_pad'
            padding_layer = keras.layers.ZeroPadding2D(
                padding=(params['pads'][0], params['pads'][1]),
                name=padding_name
            )
            layers[padding_name] = padding_layer(layers[input_name])
            input_name = padding_name

        W = W.transpose(2, 3, 1, 0)
        height, width, channels_per_group, out_channels = W.shape
        n_groups = params['group']
        in_channels = channels_per_group * n_groups

        if n_groups == in_channels and n_groups != 1:
            W = W.transpose(0, 1, 3, 2)
            if has_bias:
                weights = [W, bias]
            else:
                weights = [W]

            conv = keras.layers.DepthwiseConv2D(
                kernel_size=(height, width),
                strides=(params['strides'][0], params['strides'][1]),
                padding='valid',
                use_bias=has_bias,
                activation=None,
                depth_multiplier=1,
                weights=weights,
                dilation_rate=params['dilations'][0],
                bias_initializer='zeros', kernel_initializer='zeros',
                name=node_name
            )
            layers[node_name] = conv(layers[input_name])

        elif n_groups != 1:
            # Example from https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
            def target_layer(x, groups=params['group'], stride_y=params['strides'][0], stride_x=params['strides'][1]):
                x = tf.transpose(x, [0, 2, 3, 1])

                def convolve_lambda(i, k):
                    return tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding='VALID')

                input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
                weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=W.transpose(0, 1, 2, 3))
                output_groups = [convolve_lambda(i, k) for i, k in zip(input_groups, weight_groups)]

                layer = tf.concat(axis=3, values=output_groups)

                layer = tf.transpose(layer, [0, 3, 1, 2])
                return layer

            lambda_layer = keras.layers.Lambda(target_layer)
            layers[node_name] = lambda_layer(layers[input_name])

        else:
            if has_bias:
                weights = [W, bias]
            else:
                weights = [W]

            conv = keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=(height, width),
                strides=(params['strides'][0], params['strides'][1]),
                padding='valid',
                weights=weights,
                use_bias=has_bias,
                activation=None,
                dilation_rate=params['dilations'][0],
                bias_initializer='zeros', kernel_initializer='zeros',
                name=node_name
            )
            layers[node_name] = conv(layers[input_name])

    else:  # 1D conv
        raise NotImplementedError('Not implemented')


def convert_convtranspose(node, params, layers, weights, node_name):
    """
    Convert transposed convolution layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting transposed convolution ...')

    input_name, W, bias, has_bias = extract_input_weight_bias(node, weights)

    if len(W.shape) == 4:
        W = W.transpose(2, 3, 1, 0)
        height, width, n_filters, channels = W.shape

        if has_bias:
            weights = [W, bias]
        else:
            weights = [W]
                
        n_groups = params['group']
        if n_groups > 1:
            raise AssertionError('Cannot convert conv1d with groups != 1')

        if params['dilations'][0] > 1:
            raise AssertionError('Cannot convert conv1d with dilation_rate != 1')

        conv = keras.layers.Conv2DTranspose(
            filters=n_filters,
            kernel_size=(height, width),
            strides=(params['strides'][0], params['strides'][1]),
            padding='valid',
            output_padding=0,
            weights=weights,
            use_bias=has_bias,
            activation=None,
            dilation_rate=params['dilations'][0],
            bias_initializer='zeros', kernel_initializer='zeros',
            name=node_name
        )

        layers[node_name] = conv(layers[input_name])

        # Magic ad-hoc.
        # See the Keras issue: https://github.com/keras-team/keras/issues/6777
        layers[node_name].set_shape(layers[node_name]._keras_shape)

        pads = params['pads']
        if pads[0] > 0:
            assert(len(pads) == 2 or (pads[2] == pads[0] and pads[3] == pads[1]))

            crop = keras.layers.Cropping2D(
                pads[:2],
                name=node_name + '_crop'
            )
            layers[node_name] = crop(layers[node_name])
    else:
        raise AssertionError('Layer is not supported for now')
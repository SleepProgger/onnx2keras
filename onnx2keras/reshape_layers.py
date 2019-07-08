import keras.layers
import keras.backend as K
import numpy as np
import logging
from .utils import is_numpy, ensure_tf_type


def convert_transpose(node, params, layers, node_name):
    """
    Convert transpose.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:transpose')
    input_name = node.input[0]

    if params['perm'][0] != 0:
        logger.warning('Can\'t permute batch dimension. Result may be wrong.')
        if is_numpy(layers[input_name]):
            logger.warning('Transposing numpy array.')
            layers[node_name] = np.transpose(layers[input_name], axes=params['perm'])
        else:
            raise NotImplementedError('Can\'t modify this type of data')
    else:
        permute = keras.layers.Permute(params['perm'][1:], name=node_name)
        layers[node_name] = permute(layers[input_name])


def convert_shape(node, params, layers, node_name):
    """
    Convert shape.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:shape')
    input_0 = ensure_tf_type(layers[node.input[0]], layers[list(layers)[0]])

    logger.debug('Actual result:')
    logger.debug(np.array(input_0._keras_shape))

    layers[node_name] = np.array(input_0._keras_shape)


def convert_gather(node, params, layers, node_name):
    """
    Convert gather.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:gather')

    if is_numpy(layers[node.input[0]]) and is_numpy(layers[node.input[1]]):
        logger.debug('Gather from numpy array')

        if params['axis'] == 0:
            layers[node_name] = np.array(layers[node.input[0]][layers[node.input[1]]])
        elif params['axis'] == 1:
            layers[node_name] = np.array(layers[:, node.input[0]][layers[node.input[1]]])
        elif params['axis'] == 2:
            layers[node_name] = np.array(layers[:, :, node.input[0]][layers[node.input[1]]])
        elif params['axis'] == 3:
            layers[node_name] = np.array(layers[:, :, :, node.input[0]][layers[node.input[1]]])
        else:
            raise AttributeError('Can\'t gather by axis more than 3.')
    else:
        # TODO: impl using K.gather... should be simply
        raise AttributeError('Can\'t gather from tf tensor.')


def convert_concat(node, params, layers, node_name):
    """
    Convert concat.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:concat')

    if is_numpy(layers[node.input[0]]) and is_numpy(layers[node.input[1]]):
        # TODO impll. for n
        logger.debug('Concat 2 numpy arrays.')
        layers[node_name] = np.concatenate([layers[node.input[0]], layers[node.input[1]]], axis=params['axis'])
    else:
        logger.debug('Concat  tf tensors.')
        inputs = [ensure_tf_type(layers[x], layers[list(layers)[0]]) for x in node.input]
        axis =params["axis"]

        def target_layer(x, axis=axis):
            import keras.backend as K
            return K.concatenate(x, axis=axis)

        out_shape = list(K.int_shape(inputs[0]))
        # TODO: doesn't support dynamic values in shape[axis]
        out_shape[axis] = int(sum([K.int_shape(x)[axis] for x in inputs]))
        lambda_layer = keras.layers.Lambda(target_layer, output_shape=out_shape, name=node_name)
        layers[node_name] = lambda_layer(inputs)


def convert_reshape(node, params, layers, node_name):
    """
    Convert reshape.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:reshape')

    input_0 = layers[node.input[0]]
    input_1 = layers[node.input[1]]

    if is_numpy(input_1):
        logger.debug('The second argument is numpy array.')
        if is_numpy(input_0):
            logger.debug('The first argument is numpy array. Apply np.reshape.')
            layers[node_name] = np.reshape(input_0, input_1)
        else:
            input_0 = ensure_tf_type(layers[node.input[0]], layers[list(layers)[0]])
            logger.debug('The first argument is Keras/tf layer. Apply keras.Reshape.')
            reshape = keras.layers.Reshape(input_1[1:], name=node_name)
            layers[node_name] = reshape(input_0)
    else:
        raise AttributeError('Can\'t reshape dynamic size.')


def convert_unsqueeze(node, params, layers, node_name):
    """
    Convert unsqueeze.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:unsqueeze')

    if len(node.input) != 1:
        raise AttributeError('Number of inputs is not equal 1 for unsqueeze layer')

    if len(params['axes']) != 1:
        raise AttributeError('Number of axes is not equal 1. Cannot unsqueeze')

    if is_numpy(layers[node.input[0]]):
        logger.debug('Work with numpy types.')
        # TODO: support multiple axis
        layers[node_name] = np.expand_dims(layers[node.input[0]], params['axes'][0])
    else:
        axis = tuple(sorted(params['axes'], reverse=True))

        def target_layer(x, axis=axis):
            import keras
            for ax in axis:
                x = keras.backend.expand_dims(x, ax)
            return x

        # According to keras docs All input dimensions need to be fixed....
        # TODO: test and if true -> change to expand_dims in loop or K.reshape ?
        # TODO: Reshape ignores first axis (BS), is this ok here ?
        #lambda_layer = keras.layers.Lambda(target_layer, name=node_name)
        # layer = lambda_layer(layers[node.input[0]])

        new_shape = list(K.int_shape(node.input[0]))
        for i in axis: # TODO: verify if this is correct
            new_shape.insert(i, 1)
        layer = keras.layers.Reshape(new_shape[1:])

        layers[node_name] = layer


def convert_flatten(node, params, layers, node_name):
    """
    Convert flatten.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:flatten')

    if len(node.input) != 1:
        raise AttributeError('Number of inputs is not equal 1 for flatten layer')

    logger.debug('Convert inputs to Keras/TF layers if needed.')
    input_0 = ensure_tf_type(layers[node.input[0]], layers[list(layers)[0]])

    reshape = keras.layers.Reshape([-1], name=node_name)
    layers[node_name] = reshape(input_0)

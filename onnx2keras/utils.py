import numpy as np
import keras
import keras.backend as K

def is_numpy(obj):
    """
    Check of the type is instance of numpy array
    :param obj: object to check
    :return: True if the object is numpy-type array.
    """
    return isinstance(obj, (np.ndarray, np.generic))


def ensure_numpy_type(obj):
    """
    Raise exception if it's not a numpy
    :param obj: object to check
    :return: numpy object
    """
    if is_numpy(obj):
        return obj
    else:
        raise AttributeError('Not a numpy type.')


def get_same_lambda_shape(inp):
    return K.int_shape(inp)[1:]


def get_reduce_lambda_shape(inp, axis, keepdims=False):
    shape = list(K.int_shape(inp))
    if keepdims:
        return K.int_shape(inp)[1:]
    axis = [axis] if isinstance(int) else axis
    for ax in axis:
        shape[ax] = 1
    return tuple(shape[1:])


_count = 0 # TODO: dirty
def ensure_tf_type(obj, fake_input_layer=None):
    """
    Convert to Keras Constant if needed
    :param obj: numpy / tf type
    :param fake_input_layer: fake input layer to add constant
    :return: tf type
    """
    if is_numpy(obj):
        if obj.dtype == np.int64:
            obj = np.int32(obj)

        def target_layer(_, inp=obj, dtype=obj.dtype.name):
            import numpy as np
            import keras.backend as K
            if not isinstance(inp, (np.ndarray, np.generic)):
                inp = np.array(inp, dtype=dtype)
            return K.constant(inp, dtype=inp.dtype, shape=inp.shape)

        global _count
        # TODO: i am not sure aboutthe output_shape...
        # This way it has to always contain a dimension for the batches or be a scalar
        # ... but it seems to work
        lambda_layer = keras.layers.Lambda(target_layer, output_shape=obj.shape[1:], name="constant_%i" % _count)
        _count += 1
        return lambda_layer(fake_input_layer)
    else:
        return obj


def check_torch_keras_error(model, k_model, input_np, epsilon=1e-5):
    """
    Check difference between Torch and Keras models
    :param model: torch model
    :param k_model: keras model
    :param input_np: input data
    :param epsilon: allowed difference
    :return: actual difference
    """
    from torch.autograd import Variable
    import torch

    input_var = Variable(torch.FloatTensor(input_np))
    pytorch_output = model(input_var).data.numpy()
    keras_output = k_model.predict(input_np)

    error = np.max(pytorch_output - keras_output)

    assert error < epsilon
    return error
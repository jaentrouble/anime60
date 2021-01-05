import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import nn

BN_MOMENTUM = 0.9
NORM = 'batch'

def nc(name, postfix):
    """nc (name check)
    If name is None, return None
    Else, append name and postfix
    """
    if name is None:
        return None
    else:
        return '_'.join([name,postfix])


"""
Simple wrapper around keras layers for convenience
"""


def norm_layer(inputs, name):
    norm_layers = {
        'layer' : layers.LayerNormalization,
        'batch' : layers.BatchNormalization,
        'none' : layers.Activation,
    }
    norm_kwargs = {
        'layer' : {'name':name},
        'batch' : {'momentum':BN_MOMENTUM,'name':name},
        'none' : {'activation':'linear','name':name}
    }
    return norm_layers[NORM](**norm_kwargs[NORM])(inputs)

def conv2d_layer(inputs, *args, **kwargs):
    """
    Simple wrapper around conv2d layer for convenience
    """
    return layers.Conv2D(*args,**kwargs)(inputs)

def relu_layer(inputs, *args, **kwargs):
    """
    Simple wrapper around relu layer for convenience
    """
    return layers.ReLU(*args,**kwargs)(inputs)

def add_layer(inputs, **kwargs):
    """
    Simple wrapper around add layer for convenience
    """
    return layers.Add(**kwargs)(inputs)

def upsampling_layer(inputs, *args, **kwargs):
    """
    Simple wrapper around upsampling2d layer for convenience
    """
    return layers.UpSampling2D(*args,**kwargs)(inputs)
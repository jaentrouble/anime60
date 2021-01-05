import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .functional_layers import *

"""
Voxel interpreting layer using only Functional API

Every functions should:
    1. 'inputs' argument as the first argument,
    2. return outputs tensor
So that every functions should be used in this format:
    output = function(inputs, *args, **kwargs)

"""

GAMMA_FLOW = 0.1
GAMMA_MASK = 0.03

def voxel_interp(
    inputs,
    interpolate_ratios,
    gamma_flow=GAMMA_FLOW,
    gamma_mask=GAMMA_MASK,
    name=None,
):
    if not isinstance(inputs,list):
        raise ValueError('A VoxelInterp layer should be called '
                            'on a list of inputs')
    if not len(inputs) == 2:
        raise ValueError('A VoxelInterp layer gets two inputs:'
                            'Frame0 & Frame1 concat, Encoded_image\n'
                            f'But got {len(inputs)} inputs')
    
    
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import nn
from .functional_layers import *


"""
HRNet using Functional API only.

Every functions should:
    1. 'inputs' argument as the first argument,
    2. return outputs tensor
So that every functions should be used in this format:
    output = function(inputs, *args, **kwargs)
"""


def conv3x3(inputs, filters, stride=1, name=None):
    """A 3x3 Conv2D layer with 'same' padding"""
    return layers.Conv2D(filters=filters, kernel_size=3, strides=stride,
                        padding='same',name=name)(inputs)

def basic_block(inputs, filters, stride=1, name=None):
    r"""Smallest building block of HR-Net

    It consists of two Conv2D layers, each followed by a Batch Normalization
    layer. A residual is added just before the last ReLU layer.
    Striding is only done in the first Conv2D layer.

        Input - Conv2D(Strided) - BN - ReLU - Conv2D - BN  (+) - ReLU
            \                                              /
              - Residual (Conv2D + BN if needed to match size)

    Arguments
    ---------
    inputs : keras.Input or Tensor
        Input tensor
    filters : int
        Number of filters of Conv2D layers
    stride : int, optional
        Stride in the first Conv2D layer (default 1)
    name : str, optional
        Name of this block. (default None)

    Output
    ------
    Output shape:
        (N,H,W,C) Tensor where C = filters
        If strided, H, W will be downsampled.
    """
    if stride != 1 or inputs.shape[-1] != filters:
        residual = conv2d_layer(
            inputs,
            filters,
            1,
            strides=stride,
            padding='same',
            name=nc(name,'res_downsample')
        )
        residual = norm_layer(residual, name=nc(name,'res_norm'))
    else:
        residual = inputs
    x = conv3x3(inputs, filters, stride, name=nc(name,'conv0'))
    x = norm_layer(x, name=nc(name, 'norm0'))
    x = relu_layer(x, name=nc(name, 'relu0'))

    x = conv3x3(x, filters, name=nc(name,'conv1'))
    x = norm_layer(x, name=nc(name, 'norm1'))

    x = add_layer([x,residual], name=nc(name, 'res_add'))
    outputs = relu_layer(x, name=nc(name, 'relu1'))

    return outputs

def high_resolution_branch(
    inputs, filters, blocks, stride=1, name=None,
):
    r"""A branch of HR-Net Module

    It consists of BasicBlocks. If specified, it will stride once.

        Input - block - block - ... - block

    If stide:

        Input
            \
             block(stride) - block - ... - block

    Arguments
    ---------
    inputs : keras.Input or Tensor
        Input tensor
    filters : int
        Number of filters of Blocks
    blocks : int
        number of blocks, must be greater than 0
    stride : int, optional
        Stride in the first block (default 1)
    name : str, optional
        Name of this branch. (default None)

    Output
    ------
    Output shape:
        (N,H,W,C) Tensor where C = filters
        If downsampled, H, W will decrease
    """
    x = basic_block(inputs, filters, stride, name=nc(name,'block0'))
    for i in range(1, blocks):
        x = basic_block(x, filters, name=nc(name,f'block{i}'))
    return x

def high_resolution_fusion(
    inputs, filters:list, name=None
):
    r"""A fusion layer at the end of a HR-Net Module

        #j inputs          #i outputs

        Tensor   -----    Tensor
        .         \ /       .
        .       ---*---     .
        .         / \       .
        Tensor   -----    Tensor
        
    Input & Output numbers of layers may be different

    Arguments
    ---------
    inputs : keras.Input or Tensor
        Input tensor
    filters : list
        Number of filters(C) per outputs. len(filters) == output Tensor #
    name : str, optional
        Name of this fusion layer. (default None)

    Output
    ------
    Tuple of #i Tensors:
        Each Tensor shaped (N,H,W,C)
    """
    if not isinstance(inputs,list):
        raise ValueError('A HRFusion layer should be called '
                            'on a list of inputs')
    num_inputs = len(inputs)
    num_outputs = len(filters)

    outputs = []
    for i in range(num_outputs):
        for j in range(num_inputs):
            if j > i:
                xp = upsampling_layer(
                    inputs[j],
                    size=2**(j-i),
                    interpolation='bilinear',
                    dtype=tf.float32,
                    name=nc(name,f'upsample{i}_{j}'),
                )
                xp = conv2d_layer(
                    xp,
                    filters[i],
                    1,
                    padding='same',
                    name=nc(name,f'conv{i}_{j}'),
                )
                xp = norm_layer(
                    xp,
                    name=nc(name,f'norm{i}_{j}'),
                )
            elif j == i:
                xp = conv2d_layer(
                    inputs[j],
                    filters[i],
                    1,
                    padding='same',
                    name=nc(name,f'conv{i}_{j}'),
                )
                xp = norm_layer(
                    xp,
                    name=nc(name,f'norm{i}_{j}'),
                )
            elif j < i:
                xp = inputs[j]
                for k in range(i-j):
                    xp = conv3x3(
                        xp,
                        filters[i],
                        stride=2,
                        name=nc(name,f'downsample{i}_{j}_{k}'),
                    )
                    xp = norm_layer(
                        xp,
                        name=nc(name,f'down_norm{i}_{j}_{k}'),
                    )
                    if k < (i-j-1):
                        # Skip relu on the last part
                        xp = relu_layer(
                            xp,
                            name=nc(name,f'down_relu{i}_{j}_{k}'),
                        )
            if j==0:
                x = xp
            else:
                x = add_layer([x, xp], name=nc(name,f'add{i}_{j}'))
        out = relu_layer(x, name=nc(name,f'relu{i}'))
        outputs.append(out)
    return outputs

def high_resolution_module(
    inputs, filters:list, blocks:list, name=None,
):
    r"""A Fusion - Branch Module

    It consists of a fusion module and several branches

        Input(1) \                      -Branch(1)
                  \                    |
        Input(2) -\\                    -Branch(2)
        .           --  Fusion layer --|
        .          /                    .
        .         /                     .
        Input(j) /                      .
                                       |
                                        -Branch(i)

    Arguments
    ---------
    inputs : keras.Input or Tensor
        Input tensor
    filters : list
        Number of filters(C) per branches. len(filters) == output Tensor #
    blocks : list
        Number of blocks per each branch.
    name : str, optional
        Name of this branch. (default None)

    Output
    ------
    Tuple of #i Tensors:
        Each Tensor shaped (N,H,W,C)
    """
    num_branches = len(filters)
    if num_branches != len(blocks):
        err_msg = (f'NUM_BRANCHES {num_branches} !='
                    f'NUM_BLOCKS {len(blocks)}')
        raise ValueError(err_msg)
    
    x = high_resolution_fusion(
        inputs,
        filters,
        name=nc(name, 'fusion')
    )
    outputs = []

    for i in range(num_branches):
        outputs.append(
            high_resolution_branch(
                x[i],
                filters[i],
                blocks[i],
                name=nc(name,f'branch{i}')
            )
        )
    return outputs

if __name__ == '__main__':
    inputs = keras.Input((320,320,3))
    x = high_resolution_module(
        [inputs],
        filters=[8],
        blocks=[3],
        name='module_0'
    )
    x = high_resolution_module(
        x,
        filters=[8,16],
        blocks=[3,3],
        name='module_1'
    )
    x = high_resolution_module(
        x,
        filters=[8,16,32],
        blocks=[3,3,3],
        name='module_2'
    )
    x = high_resolution_fusion(
        x,
        filters=[8],
        name='fusion_0'
    )
    outputs = x[0]
    test_model = keras.Model(inputs=inputs, outputs=outputs)
    test_model.summary()
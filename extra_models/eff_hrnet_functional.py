import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .hrnet_functional import *

"""
Efficient HRNet using Functional API only.

Every functions should:
    1. 'inputs' argument as the first argument,
    2. return outputs tensor
So that every functions should be used in this format:
    output = function(inputs, *args, **kwargs)
"""


def branches(
    inputs:list,
    filters:list,
    blocks:list,
    name=None,
):
    r"""branches
    Branch part of Efficient HRNet
    
    Arguments
    ---------
    inputs : list of 4 Tensors
        Outputs of backbone model.
        Needs to be a list of 4 Tensors
    filters : list of 4 integers
        Number of filters (Wb in paper) per branch.
        Needs to be 4 integers.
    blocks : list of 3 integers
        Number of per each stage. (Refer to the paper)
        Note: Here, it refers to Basic block number, 
                which has 2 conv layers
    name : str, optional
        Name of this network. (default None)

    Output
    ------
    Output Shape:
        (N,H/2,W/2,Wb1)
    """
    assert len(inputs) == 4, 'Inputs must be a list of 4 Tensors'
    assert len(filters) == 4, 'Filters should be a list of 4 integers'
    assert len(blocks) == 3, 'Blocks should be a list of 3 integers'

    branch_outputs=[]
    # branch 0
    x0 = inputs[0]
    for i, bn in enumerate(blocks):
        x0 = high_resolution_branch(
            x0,
            filters[0],
            bn,
            name=nc(name,f'br0_stage{i}')
        )
    branch_outputs.append(x0)

    x1 = inputs[1]
    for i, bn in enumerate(blocks):
        x1 = high_resolution_branch(
            x1,
            filters[1],
            bn,
            name=nc(name,f'br1_stage{i}')
        )
    branch_outputs.append(x1)

    x2 = inputs[2]
    for i, bn in enumerate(blocks[1:]):
        x2 = high_resolution_branch(
            x2,
            filters[2],
            bn,
            name=nc(name,f'br2_stage{i+1}')
        )
    branch_outputs.append(x2)

    x3 = inputs[3]
    x3 = high_resolution_branch(
        x3,
        filters[3],
        blocks[2],
        name=nc(name,f'br3_stage2')
    )
    branch_outputs.append(x3)

    fused_output = high_resolution_fusion(
        branch_outputs,
        [filters[0]],
        name=nc(name,f'fusion')
    )
    x = upsampling_layer(
        fused_output[0],
        size=2,
        interpolation='bilinear',
        dtype=tf.float32,
        name=nc(name,f'upsampling')
    )
    x = basic_block(
        x,
        filters[0],
        name=nc(name,'upsample_block0')
    )
    x = basic_block(
        x,
        filters[0],
        name=nc(name,'upsample_block1')
    )

    outputs = x
    return outputs

def efficient_hrnet_b0(
    inputs,
    filters:list,
    blocks:list,
):
    """EfficientHRNet_B0
    Uses Efficientnet_B0 as a backbone model

    Arguments
    ---------
    inputs : keras.Input or Tensor
        Input tensor
        (N,H,W,C) expected
    filters : list of 4 integers
        Number of filters (Wb in paper) per branch.
        Needs to be 4 integers.
    blocks : list of 3 integers
        Number of per each stage. (Refer to the paper)
        Note: Here, it refers to Basic block number, 
                which has 2 conv layers

    Output
    ------
    Output Shape:
        (N,H/2,W/2,Wb1)
    """
    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_tensor=inputs,
    )
    backbone_outputs = [
        backbone.get_layer('block2b_add').output,
        backbone.get_layer('block3b_add').output,
        backbone.get_layer('block5c_add').output,
        backbone.get_layer('block7a_project_bn').output,
    ]
    outputs = branches(
        backbone_outputs,
        filters,
        blocks,
        name='branches',
    )
    return outputs

def efficient_hrnet_mv3_small(
    inputs,
    filters:list,
    blocks:list,
    alpha=1.0,
):
    """EfficientHRNet_MV3_Small
    Uses MobileNetV3Large as a backbone model

    Arguments
    ---------
    inputs : keras.Input or Tensor
        Input tensor
        (N,H,W,C) expected
    filters : list of 4 integers
        Number of filters (Wb in paper) per branch.
        Needs to be 4 integers.
    blocks : list of 3 integers
        Number of per each stage. (Refer to the paper)
        Note: Here, it refers to Basic block number, 
                which has 2 conv layers
    alpha : float
        MobileNetV3Small's width parameter

    Output
    ------
    Output Shape:
        (N,H/2,W/2,Wb1)
    """
    backbone = keras.applications.MobileNetV3Small(
        include_top=False,
        weights=None,
        input_tensor=inputs,
        alpha=alpha,
    )
    backbone_outputs = [
        backbone.get_layer('expanded_conv/project/BatchNorm').output,
        backbone.get_layer('expanded_conv_2/Add').output,
        backbone.get_layer('expanded_conv_7/Add').output,
        backbone.get_layer('expanded_conv_10/Add').output,
    ]
    outputs = branches(
        backbone_outputs,
        filters,
        blocks,
        name='branches',
    )
    return outputs


if __name__ == '__main__':
    inputs = keras.Input((320,320,3))
    outputs = efficient_hrnet_b0(
        inputs,
        filters=[12,22,44,86],
        blocks=[2,2,4],
    )
    test_model = keras.Model(inputs=inputs, outputs=outputs,
                            name='effhrnet')
    test_model.summary()
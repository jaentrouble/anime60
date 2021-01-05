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
    r"""Voxel interpreter

    Creates voxel flow from any inputs and returns synthesized frames.
    It takes an encoded tensor of any number of channels and
    two original frames and returns interpolated frames.

    Example : [0.4, 0.8]

                 0.4             0.4         0.2
    Frame 0|--------------------------------------|Frame 1
                          |                |
                          |                |
                        Inter1            Inter2
    
    Arguments
    ---------
    Inputs : list of Tensors
        [Frame0 & Frame1 concat, Encoded_image]
        Frames should be concatenated i.e. (N,H,W,2*3)
        Watch out not to concatenate edge maps here. They are unnecessary.
        Encoded_image's shape does not need to be the same as frames.
        It will be resized to frames' size
    interpolate_ratios : list
        list of floats in [0.0, 1.0], where the interpolated frames should be
    gamma_motion : float
        total variation regularization weight for voxel flow
    gamma_mask : float
        total variation regularization weight for temporal component
        

    Returns
    -------
    outputs: tf.Tensor
        (N,H,W,3*F) where 3 = RGB, F = interpolated frames
        images will be concatenated along the last axis
        i.e. R0,B0,G0, R1,B1,G1 ...
    losses : tuple of tf.Tensor
        Additional regularization loss
    """

    if not isinstance(inputs,list):
        raise ValueError('A VoxelInterp layer should be called '
                            'on a list of inputs')
    if not len(inputs) == 2:
        raise ValueError('A VoxelInterp layer gets two inputs:'
                            'Frame0 & Frame1 concat, Encoded_image\n'
                            f'But got {len(inputs)} inputs')
    
    frame0 = inputs[0][...,0:3]
    frame1 = inputs[0][...,3:6]
    encoded_image = inputs[1]
    encoded_image_finite = tf.where(
        tf.math.is_finite(encoded_image),
        encoded_image,
        0.0,
    )

    # Keep layer temporarily to add flow/mask total variaton loss
    conv_flow_mask = layers.Conv2D(
        filters=3,
        kernel_size=3,
        padding='same',
        activation='tanh',
        dtype='float32',
        kernel_initializer='zeros',
        name=nc(name,'conv_flow_mask')
    )
    net = conv_flow_mask(encoded_image_finite)

    frame_shape = tf.shape(frame0,name=nc(name,'frame_shape'))
    batch_size, height, width = frame_shape[0:3]
    total_pixels =tf.reduce_prod(frame_shape,name=nc(name,'total_pixels'))

    net = tf.image.resize(net, (height,width), name=nc(name,'resize'))

    _, hh, ww = tf.meshgrid(
        tf.range(batch_size, dtype=tf.float32, name=nc(name,'bb')),
        tf.linspace(tf.cast(-1,tf.float32), 1, height,name=nc(name,'hh')),
        tf.linspace(tf.cast(-1,tf.float32), 1, width,name=nc(name,'ww')),
        indexing='ij',
        name='meshgrid',
    )

    flow = net[...,0:2] * 2 # Range: (-2,2)
    mask = net[...,2:3] # Range: (-1,1)

    flow_reg_loss = tf.math.divide_no_nan(
        gamma_flow*tf.reduce_sum(tf.image.total_variation(
            flow, name=nc(name,'flow_reg_loss_tv'))),
        tf.cast(
            total_pixels,
            tf.float32,
            name=nc(name,'flow_reg_loss_cast')),
        name=nc(name,'flow_reg_loss_div'),
    )

    mask_reg_loss = tf.math.divide_no_nan(
        gamma_mask*tf.reduce_sum(tf.image.total_variation(
            mask, name=nc(name,'mask_reg_loss_tv'))),
        tf.cast(
            total_pixels,
            tf.float32,
            name=nc(name,'mask_reg_loss_cast')),
        name=nc(name,'flow_reg_loss_div')
    )

    output_frames = []
    for i, alpha in enumerate(interpolate_ratios):
        coor_h_0 = hh + flow[...,0] * alpha
        coor_w_0 = ww + flow[...,1] * alpha

        coor_h_1 = hh - flow[...,0] * (1-alpha)
        coor_w_1 = ww - flow[...,1] * (1-alpha)

        output_0 = bilinear_interp(
            frame0, 
            coor_h_0,
            coor_w_0,
            name=nc(name,'bilinear0')    
        )
        output_1 = bilinear_interp(
            frame1,
            coor_h_1,
            coor_w_1,
            name=nc(name,'bilinear1')
        )

        norm_mask = (1-alpha) * (1+mask) # Normalize to (0.0, 1.0)
        output = norm_mask*output_0 + (1-norm_mask)*output_1
        output_frames.append(output)

    outputs = tf.concat(output_frames, axis=-1, name=nc(name,'concat'))
    return outputs, (flow_reg_loss,mask_reg_loss)

def bilinear_interp(image, new_hh, new_ww, name=None):
    """Perform bilinear sampling on im given x, y coordinates

    This function implements the differentiable sampling mechanism with
    bilinear kernel. Introduced in https://arxiv.org/abs/1506.02025, equation
    (5).
    
    x,y are tensors specfying normalized coorindates [-1,1] to sample from im.
    (-1,-1) means (0,0) coordinate in im. (1,1) means the most bottom right pixel.

    Arguments
    ---------
        image: Tensor of size [N, H, W, C] 
        new_hh: Tensor of size [N, H, W]
        new_ww: Tensor of size [N, H, W]

    Returns
    -------
        Tensor of size [batch_size, height, width, depth]
    """
    height, width = tf.shape(image, name=nc(name,'image_shape'))[1:3]
    max_h = height-1
    max_w = width -1

    height_f = tf.cast(height, tf.float32, name=nc(name,'height_f'))
    width_f = tf.cast(width, tf.float32, name=nc(name,'width_f'))

    hh = (new_hh+1) * (height_f-1) / 2
    ww = (new_ww+1) * (width_f-1) / 2

    h0 = tf.cast(tf.floor(hh), tf.int32)
    h1 = h0 + 1
    w0 = tf.cast(tf.floor(ww), tf.int32)
    w1 = w0 + 1

    h0 = tf.clip_by_value(h0, 0, max_h,name=nc(name,'clip_h0'))
    h1 = tf.clip_by_value(h1, 0, max_h,name=nc(name,'clip_h1'))
    w0 = tf.clip_by_value(w0, 0, max_w,name=nc(name,'clip_w0'))
    w1 = tf.clip_by_value(w1, 0, max_w,name=nc(name,'clip_w1'))

    idx_ru = tf.stack([h0, w0], axis=-1,name=nc(name,'stack_ru'))
    idx_rd = tf.stack([h1, w0], axis=-1,name=nc(name,'stack_rd'))
    idx_lu = tf.stack([h0, w1], axis=-1,name=nc(name,'stack_lu'))
    idx_ld = tf.stack([h1, w1], axis=-1,name=nc(name,'stack_ld'))

    ru = tf.gather_nd(image, idx_ru, batch_dims=1,name=nc(name,'gather_ru'))
    rd = tf.gather_nd(image, idx_rd, batch_dims=1,name=nc(name,'gather_rd'))
    lu = tf.gather_nd(image, idx_lu, batch_dims=1,name=nc(name,'gather_lu'))
    ld = tf.gather_nd(image, idx_ld, batch_dims=1,name=nc(name,'gather_ld'))

    h1_f = tf.cast(h1, tf.float32, name=nc(name,'h1_f'))
    w1_f = tf.cast(w1, tf.float32, name=nc(name,'w2_f'))

    w_ru = tf.expand_dims((h1_f-hh) * (w1_f-ww),-1)
    w_rd = tf.expand_dims((1-(h1_f-hh)) * (w1_f-ww),-1)
    w_lu = tf.expand_dims((h1_f-hh) * (1-(w1_f-ww)),-1)
    w_ld = tf.expand_dims((1-(h1_f-hh)) * (1-(w1_f-ww)),-1)

    output = w_ru*ru + w_rd*rd + w_lu*lu + w_ld*ld
    return output


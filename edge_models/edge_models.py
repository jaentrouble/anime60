import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from . import custom_layers_edge as clayers

# Get inputs and return outputs


def hr_5_3_8(inputs):
    x = [inputs]
    x = clayers.HighResolutionModule(
        filters=[8],
        blocks=[3],
        name='HR_0'
    )(x)
    x = clayers.HighResolutionModule(
        filters=[8,16],
        blocks=[3,3],
        name='HR_1'
    )(x)
    x = clayers.HighResolutionModule(
        filters=[8,16,32],
        blocks=[3,3,3],
        name='HR_2'
    )(x)
    x = clayers.HighResolutionModule(
        filters=[8,16,32,64],
        blocks=[3,3,3,3],
        name='HR_3'
    )(x)
    x = clayers.HighResolutionModule(
        filters=[8,16,32,64],
        blocks=[3,3,3,3],
        name='HR_4'
    )(x)
    x = clayers.HighResolutionFusion(
        filters=[8],
        name='Fusion_0'
    )(x)
    outputs = layers.Activation('linear', dtype='float32')(x[0])
    return outputs

def hr_3_2_16(inputs):
    x = [inputs]
    x = clayers.HighResolutionModule(
        filters=[16],
        blocks=[2],
        name='HR_0'
    )(x)
    x = clayers.HighResolutionModule(
        filters=[16,32],
        blocks=[2,2],
        name='HR_1'
    )(x)
    x = clayers.HighResolutionModule(
        filters=[16,32,64],
        blocks=[2,2,2],
        name='HR_2'
    )(x)
    x = clayers.HighResolutionFusion(
        filters=[32],
        name='Fusion_0'
    )(x)
    outputs = layers.Activation('linear', dtype='float32')(x[0])
    return outputs

###############################################################################
# EfficientHRNet returns half the size of input, so need to upscale
# Use UpSampling2D, not Deconv layer to prevent checkerboard effect
def upscale_block(inputs, filters):
    up_layers = keras.Sequential([
        layers.UpSampling2D(
            size=2,
            interpolation='bilinear',
            dtype=tf.float32,
        ),
        clayers.BasicBlock(filters)
    ], name='upscale_block')
    return up_layers(inputs)

###############################################################################

def ehrb0_112_12(inputs):
    half_sized = clayers.EfficientHRNet_B0(
        filters=[12,22,44,86],
        blocks =[2,2,4],
        name='EffHRNetB0'
    )(inputs)
    upscaled = upscale_block(half_sized, 12)
    outputs = layers.Activation('linear', dtype='float32')(upscaled)
    return outputs

def ehrb0_123_21(inputs):
    half_sized = clayers.EfficientHRNet_B0(
        filters=[21,42,83,166],
        blocks =[2,4,6],
        name='EffHRNetB0'
    )(inputs)
    upscaled = upscale_block(half_sized, 21)
    outputs = layers.Activation('linear', dtype='float32')(upscaled)
    return outputs

def ehrb0_143_32(inputs):
    half_sized = clayers.EfficientHRNet_B0(
        filters=[32,64,128,256],
        blocks =[2,8,6],
        name='EffHRNetB0'
    )(inputs)
    upscaled = upscale_block(half_sized, 32)
    outputs = layers.Activation('linear', dtype='float32')(upscaled)
    return outputs

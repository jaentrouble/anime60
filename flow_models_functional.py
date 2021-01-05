import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import custom_layers as clayers

# Get inputs and return outputs

###############################################################################
# EfficientHRNet returns half the size of input, so need to upscale
# Use UpSampling2D, not Deconv layer to prevent checkerboard effect
def upscale_block(inputs, filters, name=None):
    upsampled = clayers.upsampling_layer(
        inputs,
        size=2,
        interpolation='bilinear',
        dtype=tf.float32,
        name=clayers.nc(name, 'upsample')
    )
    outputs = clayers.basic_block(
        upsampled,
        filters,
        name=clayers.nc(name,'basic_block')
    )
    return outputs

###############################################################################

def ehrb0_143_32(inputs):
    half_sized = clayers.efficient_hrnet_b0(
        inputs,
        filters=[32,64,128,256],
        blocks =[2,8,6],
    )
    upscaled = upscale_block(half_sized, 32, name='upscale_block')
    outputs = layers.Activation('linear', dtype='float32')(upscaled)
    return outputs

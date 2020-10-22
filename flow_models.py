import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import custom_layers as clayers

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

def hr_3_2_24(inputs):
    x = [inputs]
    x = clayers.HighResolutionModule(
        filters=[24],
        blocks=[2],
        name='HR_0'
    )(x)
    x = clayers.HighResolutionModule(
        filters=[24,48],
        blocks=[2,2],
        name='HR_1'
    )(x)
    x = clayers.HighResolutionModule(
        filters=[24,48,96],
        blocks=[2,2,2],
        name='HR_2'
    )(x)
    x = clayers.HighResolutionFusion(
        filters=[32],
        name='Fusion_0'
    )(x)
    outputs = layers.Activation('linear', dtype='float32')(x[0])
    return outputs

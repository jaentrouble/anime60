import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from .extra_models.hrnet import *
from .extra_models.deep_voxel import *
from .extra_models.eff_hrnet import *

class MaxPoolWithArgmax2D(layers.Layer):
    def __init__(
            self, 
            pool_size=(2,2), 
            strides=(2,2), 
            padding='SAME', 
            **kwargs,
            ):
        """
        Read tf.nn.max_pool_with_argmax
        return output, mask
        """
        super().__init__(**kwargs)
        self.ksize = pool_size
        self.strides = strides
        self.padding = padding

    def call(self, inputs):
        outputs, mask = tf.nn.max_pool_with_argmax(
            inputs,
            ksize=self.ksize,
            strides=self.strides,
            padding=self.padding,
        )
        return outputs, mask

    def get_config(self):
        config = super().get_config()
        config['pool_size'] = self.ksize
        config['strides'] = self.strides
        config['padding'] = self.padding
        return config

class Max_Unpool2D(layers.Layer):
    """
    mask is from MaxPoolWithArgmax layer
    """
    def __init__(self, unpool_size=(2,2), name=None):
        super().__init__(name=name)
        self.ksize=unpool_size

    def call(self, inputs, mask):
        input_shape = tf.shape(inputs)
        output_shape = [
            input_shape[0],
            input_shape[1] * self.ksize[0],
            input_shape[2] * self.ksize[1],
            input_shape[3],
        ]
        total_flat_input_size = tf.reduce_prod(input_shape)
        flat_output_shape = [
            output_shape[0],
            output_shape[1]*output_shape[2]*output_shape[3],
        ]
        flatten_input = tf.reshape(inputs, [total_flat_input_size])
        batch_range = tf.range(input_shape[0], dtype=mask.dtype)
        b = tf.reshape(batch_range, [input_shape[0],1,1,1])
        b = tf.ones_like(inputs, dtype=mask.dtype) * b
        b = tf.reshape(b, [total_flat_input_size, 1])
        idx = tf.reshape(mask, [total_flat_input_size, 1])
        idx = tf.concat([b, idx], 1)

        outputs = tf.scatter_nd(idx, flatten_input, shape=flat_output_shape)
        outputs = tf.reshape(outputs, output_shape)
        set_input_shape = inputs.get_shape()
        set_output_shape = []
        set_output_shape.append(set_input_shape[0])
        if set_input_shape[1] != None:
            set_output_shape.append(set_input_shape[1]*self.ksize[0])
        else:
            set_output_shape.append(set_input_shape[1])
        if set_input_shape[2] != None:
            set_output_shape.append(set_input_shape[2]*self.ksize[1])
        else:
            set_output_shape.append(set_input_shape[2])
        set_output_shape.append(set_input_shape[3])
        outputs.set_shape(set_output_shape)
        return outputs        

    def get_config(self):
        config = super().get_config()
        config['unpool_size'] = self.ksize
        return config

if __name__ == '__main__':
    a = tf.random.uniform([4,6,6,1])
    inputs, mask = MaxPoolWithArgmax2D()(a)
    print('a:\n{}'.format(a[0].numpy().reshape((6,6))))
    print('mask:\n{}'.format(mask[0].numpy().reshape((3,3))))
    outputs = Max_Unpool2D()(inputs, mask)
    print(outputs.shape)
    print('a:\n{}'.format(a[0].numpy().reshape((6,6))))
    print('outputs:\n{}'.format(outputs[0].numpy().reshape((6,6))))
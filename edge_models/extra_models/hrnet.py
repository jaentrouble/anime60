import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import nn

BN_MOMENTUM = 0.999

def conv3x3(filters, stride=1):
    """A 3x3 Conv2D layer with 'same' padding"""
    return layers.Conv2D(filters=filters, kernel_size=3, strides=stride,
                        padding='same')

class BasicBlock(layers.Layer):
    r"""Smallest building block of HR-Net

    It consists of two Conv2D layers, each followed by a Batch Normalization
    layer. A residual is added just before the last ReLU layer.
    Striding is only done in the first Conv2D layer.

        Input - Conv2D(Strided) - BN - ReLU - Conv2D - BN  (+) - ReLU
            \                                              /
              - Residual (Conv2D + BN if needed to match size)

    Output
    ------
    Output shape:
        (N,H,W,C) Tensor where C = filters
        If strided, H, W will be downsampled.
    """
    def __init__(
        self,
        filters,
        stride=1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        filters : int
            Number of filters of Conv2D layers
        stride : int, optional
            Stride in the first Conv2D layer (default 1)
        """
        super().__init__(**kwargs)
        self.stride = stride
        self.filters = filters

    def build(self, input_shape):
        self.conv1 = conv3x3(self.filters, self.stride)
        self.bn1 = layers.BatchNormalization(momentum=BN_MOMENTUM)
        self.conv2 = conv3x3(self.filters)
        self.bn2 = layers.BatchNormalization(momentum=BN_MOMENTUM)
        if self.stride != 1 or \
            input_shape[-1] != self.filters:
            self.downsample = keras.Sequential([
                layers.Conv2D(self.filters, 1, strides=self.stride,
                    padding='same'),
                layers.BatchNormalization(momentum=BN_MOMENTUM)
            ])
        else:
            # Do nothing
            self.downsample = layers.Activation('linear')

    def call(self, inputs):
        residual = self.downsample(inputs)
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = layers.ReLU()(x)

        x = self.conv2(x)
        x = self.bn2(x)
        
        x = tf.add(x, residual)
        out = layers.ReLU()(x)

        return out

    def get_config(self):
        config = super().get_config()
        config['filters'] = self.filters
        config['stride'] = self.stride
        return config

class HighResolutionBranch(layers.Layer):
    r"""A branch of HR-Net Module

    It consists of BasicBlocks. If specified, it will stride once.

        Input - block - block - ... - block

    If stide:

        Input
            \
             block(stride) - block - ... - block

    Output
    ------
    Output shape:
        (N,H,W,C) Tensor where C = filters
        If downsampled, H, W will decrease
    """
    def __init__(
        self,
        filters,
        blocks,
        stride=1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        filters : int
            Number of filters of Blocks
        blocks : int
            number of blocks
        stride : int, optional
            Stride in the first block (default 1)
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.num_blocks = blocks
        self.stride = stride

    def build(self, input_shape):
        self.block_layers = []
        first_block = BasicBlock(self.filters, self.stride)
        self.block_layers.append(first_block)
        for _ in range(1, self.num_blocks):
            self.block_layers.append(
                BasicBlock(self.filters)
            )

    def call(self, inputs):
        x = inputs
        for block in self.block_layers:
            x = block(x)
        return x

    def get_config(self):
        config = super().get_config()
        config['filters'] = self.filters
        config['blocks'] = self.num_blocks
        config['stride'] = self.stride
        return config

class HighResolutionFusion(layers.Layer):
    r"""A fusion layer at the end of a HR-Net Module

        #j inputs          #i outputs

        Tensor   -----    Tensor
        .         \ /       .
        .       ---*---     .
        .         / \       .
        Tensor   -----    Tensor
        
    Input & Output numbers of layers may be different

    Output
    ------
    Tuple of #i Tensors:
        Each Tensor shaped (N,H,W,C)
    """
    def __init__(
        self,
        filters:list,
        **kwargs,
    ):
        """
        Parameters
        ----------
        filters : list
            Number of filters(C) per outputs. len(filters) == output Tensor #
        """
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        if not isinstance(input_shape,list):
            raise ValueError('A HRFusion layer should be called '
                             'on a list of inputs')
        self.num_inputs = len(input_shape)
        self.num_outputs = len(self.filters)
        self.fuse_layers = []
        for i in range(self.num_outputs):
            fuse_layer = []
            for j in range(self.num_inputs):
                if j > i:
                    fuse_layer.append(keras.Sequential([
                        # layers.Conv2DTranspose(
                        #     self.filters[i],
                        #     2**(j-i),
                        #     strides=2**(j-i),
                        # ),
                        # layers.BatchNormalization(momentum=BN_MOMENTUM),
                        layers.UpSampling2D(
                            size=2**(j-i),
                            interpolation='bilinear',
                            dtype=tf.float32,
                        ),
                        layers.Conv2D(
                            self.filters[i],
                            1,
                            padding='same',
                        ),
                        layers.BatchNormalization(momentum=BN_MOMENTUM),
                    ]))
                elif j == i:
                    fuse_layer.append(keras.Sequential([
                        layers.Conv2D(
                            self.filters[i],
                            1,
                            padding='same'
                        ),
                        layers.BatchNormalization(momentum=BN_MOMENTUM)
                    ]))
                elif j < i:
                    downsampling=[]
                    for k in range(i-j-1):
                        downsampling.append(keras.Sequential([
                            conv3x3(self.filters[i],stride=2),
                            layers.BatchNormalization(momentum=BN_MOMENTUM),
                            layers.ReLU(),
                        ]))
                    downsampling.append(keras.Sequential([
                        conv3x3(self.filters[i],stride=2),
                        layers.BatchNormalization(momentum=BN_MOMENTUM),
                    ]))
                    fuse_layer.append(keras.Sequential(downsampling))
            self.fuse_layers.append(fuse_layer)

    def call(self, inputs):
        outputs = []
        for i in range(self.num_outputs):
            # First input
            x = self.fuse_layers[i][0](inputs[0])
            for j in range(1, self.num_inputs):
                x = layers.add([x, self.fuse_layers[i][j](inputs[j])])
            out = layers.ReLU()(x)
            outputs.append(out)

        return outputs

    def get_config(self):
        config = super().get_config()
        config['filters'] = self.filters
        return config

class HighResolutionModule(layers.Layer):
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
    
    Output
    ------
    Tuple of #i Tensors:
        Each Tensor shaped (N,H,W,C)
    """
    def __init__(
        self,
        filters:list,
        blocks:list,
        **kwargs,
    ):
        """
        Parameters
        ----------
        filters : list
            Number of filters(C) per branches. len(filters) == output Tensor #
        blocks : list
            Number of blocks per each branch.
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.num_branches = len(filters)
        self.blocks = blocks
        if self.num_branches != len(self.blocks):
            err_msg = (f'NUM_BRANCHES {self.num_branches} !='
                        f'NUM_BLOCKS {len(self.blocks)}')
            raise ValueError()

    def build(self, input_shape):
        if not isinstance(input_shape,list):
            raise ValueError('A HRModule layer should be called '
                             'on a list of inputs')
        self.fusion_layer = HighResolutionFusion(self.filters)
        self.branches = []
        for i in range(self.num_branches):
            self.branches.append(HighResolutionBranch(
                filters=self.filters[i],
                blocks=self.blocks[i],
            ))
    
    def call(self, inputs):
        x = self.fusion_layer(inputs)
        outputs = []
        for i in range(self.num_branches):
            outputs.append(self.branches[i](x[i]))
        return outputs

    def get_config(self):
        config = super().get_config()
        config['filters'] = self.filters
        config['blocks'] = self.blocks
        return config

if __name__ == '__main__':
    from datetime import datetime
    import numpy as np

    
    sample_input = [keras.Input((200,200,3))]
    x = HighResolutionModule([8],[2])(sample_input)
    x = HighResolutionModule([8,16],[2,2])(x)
    x = HighResolutionModule([8,16,32],[2,2,2])(x)
    x = HighResolutionFusion([1])(x)
    flat = layers.Flatten()(x[0])
    out = layers.Dense(1)(flat)
    model = keras.Model(inputs=sample_input, outputs=out)
    model.summary()
    model.compile(
        loss='mse',
        metrics=['mse']
    )
    sample_x = np.ones((32,200,200,3))
    sample_y = np.ones((32))

    logdir = 'logs/test/' + datetime.now().strftime('%m%d-%H%M%S')
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    model.fit(
        sample_x,
        sample_y,
        callbacks=[tensorboard_callback]
    )
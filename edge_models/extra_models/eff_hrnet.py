import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .hrnet import *

# NOTE: EfficientHRNet reduces its output size into half
#       I.e. (N, H/2, W/2, C)

class EfficientHRNet(layers.Layer):
    r"""EfficientHRNet
    EfficientHRNet Model template
    Override build and define self.backbone
    Don't forget to call super().build(input_shape)
    
    Output
    ------
    Output Shape:
        (N,H/2,W/2,Wb1)
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
        filters : list of 4 integers
            Number of filters (Wb in paper) per branch.
            Needs to be 4 integers.
        blocks : list of 3 integers
            Number of per each stage. (Refer to the paper)
            Note: Here, it refers to Basic block number, 
                  which has 2 conv layers
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.blocks = blocks
        self.backbone = None
        assert len(filters) == 4, 'Filters should be a list of 4 integers'
        assert len(blocks) == 3, 'Blocks should be a list of 3 integers'

    def build(self, input_shape):
        assert self.backbone, 'self.backbone should be defined'
        self.branches = []
        self.branch1_layers = []
        for i,bn in enumerate(self.blocks):
            self.branch1_layers.append(
                HighResolutionBranch(
                    self.filters[0],
                    bn,
                    name=f'branch1_stage{i+1}'
                )
            )
        self.branches.append(keras.Sequential(self.branch1_layers))
        self.branch2_layers = []
        for i,bn in enumerate(self.blocks):
            self.branch2_layers.append(
                HighResolutionBranch(
                    self.filters[1],
                    bn,
                    name=f'branch2_stage{i+1}'
                )
            )
        self.branches.append(keras.Sequential(self.branch2_layers))
        self.branch3_layers = []
        for i,bn in enumerate(self.blocks[1:]):
            self.branch3_layers.append(
                HighResolutionBranch(
                    self.filters[2],
                    bn,
                    name=f'branch3_stage{i+2}'
                )
            )
        self.branches.append(keras.Sequential(self.branch3_layers))
        self.branches.append(HighResolutionBranch(
            self.filters[3],
            self.blocks[2],
            name=f'branch4_stage3'
        ))
        self.fusion_layer = HighResolutionFusion(
            [self.filters[0]],
            name='fusion_layer'
        )
        self.deconv_block = keras.Sequential([
            layers.UpSampling2D(
                size=2,
                interpolation='bilinear',
                dtype=tf.float32,
            ),
            BasicBlock(self.filters[0]),
            BasicBlock(self.filters[0]),
        ], name='deconv_block')

    def call(self, inputs):
        backbone_features = self.backbone(inputs)
        branch_outputs = [
            br(ft) for br,ft in zip(self.branches, backbone_features)
        ]
        fused_output = self.fusion_layer(branch_outputs)
        deconv_output = self.deconv_block(fused_output[0])
        return deconv_output

    def get_config(self):
        config = super().get_config()
        config['filters'] = self.filters
        config['blocks'] = self.blocks
        return config

class EfficientHRNet_B0(EfficientHRNet):
    """EfficientHRNet_B0
    Uses Efficientnet_B0 as a backbone model
    """
    def build(self, input_shape):
        effnet=keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=input_shape[1:],
        )
        self.backbone = keras.Model(
            inputs = effnet.input,
            outputs=[
                effnet.get_layer('block2b_add').output,
                effnet.get_layer('block3b_add').output,
                effnet.get_layer('block5c_add').output,
                effnet.get_layer('block7a_project_bn').output,
            ]
        )
        return super().build(input_shape)

class EfficientHRNet_MV3_Small(EfficientHRNet):
    """EfficientHRNet_MV3_Small
    Uses MobileNetV3Large as a backbone model
    """
    def __init__(
        self,
        filters:list,
        blocks:list,
        alpha=1.0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        filters : list of 4 integers
            Number of filters (Wb in paper) per branch.
            Needs to be 4 integers.
        blocks : list of 3 integers
            Number of per each stage. (Refer to the paper)
            Note: Here, it refers to Basic block number, 
                  which has 2 conv layers
        alpha : float
            MobileNetV3Small's width parameter
        """
        super().__init__(filters,blocks,**kwargs)
        self._alpha = alpha

    def build(self, input_shape):
        mobnet = keras.applications.MobileNetV3Small(
            input_shape=input_shape[1:],
            weights=None,
            include_top=False,
            alpha=self._alpha
        )
        self.backbone = keras.Model(
            inputs = mobnet.input,
            outputs=[
                mobnet.get_layer('expanded_conv/project/BatchNorm').output,
                mobnet.get_layer('expanded_conv_2/Add').output,
                mobnet.get_layer('expanded_conv_7/Add').output,
                mobnet.get_layer('expanded_conv_10/Add').output,
            ]
        )
        return super().build(input_shape)

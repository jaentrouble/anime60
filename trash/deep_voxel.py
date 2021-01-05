import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

GAMMA_FLOW = 0.1
GAMMA_MASK = 0.03

class VoxelInterp(layers.Layer):
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
    
    Input
    -----
    Inputs : list of Tensors
        [Frame0 & Frame1 concat, Encoded_image]
        Frames should be concatenated i.e. (N,H,W,2*C)
        Encoded_image's shape does not need to be the same as frames.
        It will be resized to frames' size

    Output
    ------
    Output shape:
        (N,H,W,C*F) where C = filters, F = interpolated frames
        images will be concatenated along the last axis
        i.e. R0,B0,G0, R1,B1,G1 ...
    """
    def __init__(
        self,
        interpolate_ratios,
        gamma_flow=GAMMA_FLOW,
        gamma_mask=GAMMA_MASK,
        **kwargs,
    ):
        """
        Arguments
        ---------
        interpolate_ratios : list
            list of floats in [0.0, 1.0], where the interpolated frames should be
        gamma_motion : float
            total variation regularization weight for voxel flow
        gamma_mask : float
            total variation regularization weight for temporal component
        """
        super().__init__(**kwargs)
        self.interpolate_ratios = interpolate_ratios
        self.frame_n = len(self.interpolate_ratios)
        self.gamma_flow = gamma_flow
        self.gamma_mask = gamma_mask

    def build(self, input_shape):
        if not isinstance(input_shape,list):
            raise ValueError('A VoxelInterp layer should be called '
                             'on a list of inputs')
        if not len(input_shape) == 2:
            raise ValueError('A VoxelInterp layer gets two inputs:'
                             'Frame0 & Frame1 concat, Encoded_image\n'
                             f'But got {len(input_shape)} inputs')
        
        self.conv = layers.Conv2D(3, 3, padding='same',
                                  activation='tanh',dtype='float32',
                                  kernel_initializer='zeros')

        self.step_counter = tf.Variable(0,trainable=False,dtype=tf.int64)
        

    def call(self, inputs):
        self.step_counter.assign_add(1)
        frame0 = inputs[0][...,0:3]
        frame1 = inputs[0][...,3:6]
        encoded_image = inputs[1]
        encoded_image = tf.where(
            tf.math.is_finite(encoded_image),
            encoded_image,
            0.0,
        )

        net = self.conv(encoded_image)
        
        batch_size = tf.shape(frame0)[0]
        height = tf.shape(frame0)[1]
        width = tf.shape(frame0)[2]
        total_pixels = tf.reduce_prod(tf.shape(frame0))

        net = tf.image.resize(net, (height,width), name='upscale')

        _, hh, ww = tf.meshgrid(
            tf.range(batch_size, dtype=tf.float32),
            tf.linspace(tf.cast(-1,tf.float32), 1, height),
            tf.linspace(tf.cast(-1,tf.float32), 1, width),
            indexing='ij',
        )

        output_frames = []
        flow = net[...,0:2] * 2 # Range: (-2,2)
        mask = net[...,2:3] # Range: (-1,1)
        tf.summary.histogram('flow',flow, step=self.step_counter)
        flow_reg_loss = tf.math.divide_no_nan(
            self.gamma_flow*tf.reduce_sum(tf.image.total_variation(flow)),
            tf.cast(total_pixels,tf.float32)
        )
        self.add_loss(flow_reg_loss)
        mask_reg_loss = tf.math.divide_no_nan(
            self.gamma_mask*tf.reduce_sum(tf.image.total_variation(mask)),
            tf.cast(total_pixels,tf.float32)
        )
        self.add_loss(mask_reg_loss)

        for i in range(self.frame_n):
            tf.summary.histogram(f'mask_{i}',mask, step=self.step_counter)

            alpha = self.interpolate_ratios[i]

            coor_h_0 = hh + flow[...,0] * alpha
            coor_w_0 = ww + flow[...,1] * alpha

            coor_h_1 = hh - flow[...,0] * (1-alpha)
            coor_w_1 = ww - flow[...,1] * (1-alpha)

            output_0 = self.bilinear_interp(frame0, coor_h_0, coor_w_0)
            output_1 = self.bilinear_interp(frame1, coor_h_1, coor_w_1)

            norm_mask = (1-alpha) * (1+mask) # Normalize to (0.0, 1.0)
            output = norm_mask*output_0 + (1-norm_mask)*output_1
            output_frames.append(output)

        stacked = tf.concat(output_frames, axis=-1)
        return stacked

    @tf.function
    def bilinear_interp(self, image, new_hh, new_ww):
        """Perform bilinear sampling on im given x, y coordinates
  
        This function implements the differentiable sampling mechanism with
        bilinear kernel. Introduced in https://arxiv.org/abs/1506.02025, equation
        (5).
        
        x,y are tensors specfying normalized coorindates [-1,1] to sample from im.
        (-1,-1) means (0,0) coordinate in im. (1,1) means the most bottom right pixel.

        Args
        ----
            image: Tensor of size [N, H, W, C] 
            new_hh: Tensor of size [N, H, W]
            new_ww: Tensor of size [N, H, W]

        Returns
        -------
            Tensor of size [batch_size, height, width, depth]
        """

        height, width = tf.shape(image)[1], tf.shape(image)[2]
        max_h = height-1
        max_w = width -1

        height_f = tf.cast(height, tf.float32)
        width_f = tf.cast(width, tf.float32)

        hh = (new_hh+1) * (height_f-1) / 2
        ww = (new_ww+1) * (width_f-1) / 2

        h0 = tf.cast(tf.floor(hh), tf.int32)
        h1 = h0 + 1
        w0 = tf.cast(tf.floor(ww), tf.int32)
        w1 = w0 + 1

        h0 = tf.clip_by_value(h0, 0, max_h)
        h1 = tf.clip_by_value(h1, 0, max_h)
        w0 = tf.clip_by_value(w0, 0, max_w)
        w1 = tf.clip_by_value(w1, 0, max_w)

        idx_ru = tf.stack([h0, w0], axis=-1)
        idx_rd = tf.stack([h1, w0], axis=-1)
        idx_lu = tf.stack([h0, w1], axis=-1)
        idx_ld = tf.stack([h1, w1], axis=-1)

        ru = tf.gather_nd(image, idx_ru, batch_dims=1)
        rd = tf.gather_nd(image, idx_rd, batch_dims=1)
        lu = tf.gather_nd(image, idx_lu, batch_dims=1)
        ld = tf.gather_nd(image, idx_ld, batch_dims=1)

        h1_f = tf.cast(h1, tf.float32)
        w1_f = tf.cast(w1, tf.float32)

        w_ru = tf.expand_dims((h1_f-hh) * (w1_f-ww),-1)
        w_rd = tf.expand_dims((1-(h1_f-hh)) * (w1_f-ww),-1)
        w_lu = tf.expand_dims((h1_f-hh) * (1-(w1_f-ww)),-1)
        w_ld = tf.expand_dims((1-(h1_f-hh)) * (1-(w1_f-ww)),-1)

        output = w_ru*ru + w_rd*rd + w_lu*lu + w_ld*ld
        return output


    def get_config(self):
        config = super().get_config()
        config['interpolate_ratios'] = self.interpolate_ratios
        config['gamma_motion'] = self.gamma_motion
        config['gamma_mask'] = self.gamma_mask
        return config
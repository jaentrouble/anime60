import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
import time
from custom_tqdm import TqdmNotebookCallback
from tqdm.keras import TqdmCallback
import albumentations as A
import random
import io
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
from extra_models.deep_voxel_functional import voxel_interp
import os
from pathlib import Path
import cv2
from model_trainer_half import anime_model
from tools.stitch_tf import *
from edge_models.model_trainer_edge import EdgeModel


NET_LAYER_NAME = 'voxel_interp_conv_flow_mask'
GAMMA_CYCLIC = 1.0
GAMMA_LIN = 0.1

DELTA_MAX = 60
class AnimeModelCyclic(keras.Model):
    """AnimeModelCyclic
    Fine tuning model using cyclic consistency loss and motion linearity loss.
    It acts like original anime_model when evaluating (i.e. using call method)
    """
    def __init__(
        self,
        anime_model,
        edge_model,
        patch_shape,
        overlap,
    ):
        """
        Arguments
        ---------
        anime_model: keras.Model
            The model to fine train.
            It may use different backbone models, but should use
            voxel_interp layer.
            If learning from a pre-trained model (which is recommended),
            load weights BEFORE handing over to this model.
        edge_model: keras.Model
            The model to make edge maps
            Like anime_model, the weight needs to be loaded BEFORE handing over
        patch_shape: tuple of ints
            Input shape of edge model.
            Will divide into patches of this size to put into edge_model
            Format : (Height, Width)
        overlap: int
            Overlap value when dividing into patches
        """
        super().__init__()
        self.anime_model = anime_model
        # A model that outputs interpolated frame and a net(flow & mask)
        self.interp_flow_model = keras.Model(
            inputs=self.anime_model.input,
            outputs=[
                self.anime_model.output,
                self.anime_model.get_layer(
                    name=NET_LAYER_NAME
                ).output
            ]
        )

        self.edge_model = edge_model
        self.patch_shape = patch_shape
        self.overlap = overlap

        self.total_loss_tracker = keras.metrics.Mean(name='L')
        self.cyclic_loss_tracker = keras.metrics.Mean(name='Lc')
        self.recon_loss_tracker = keras.metrics.Mean(name='Lr')
        self.lin_loss_tracker = keras.metrics.Mean(name='Lm')

        self.mae = keras.losses.MeanAbsoluteError()
        self.mse = keras.losses.MeanSquaredError()
    
    def call(self, inputs, training=False):
        return self.anime_model(inputs, training=training)
    
    def train_step(self, data):
        f_02, f_1 = data
        f_01 = tf.concat([f_02[...,:4],f_1],axis=-1, name='f_01_concat')
        f_12 = tf.concat([f_1,f_02[...,4:]],axis=-1, name='f_12_concat')
        with tf.GradientTape() as tape:
            inter_10, net_10 = self.interp_flow_model(f_02, training=True)
            recon_loss = self.mae(f_1[...,:3], inter_10)

            inter_05 = self.anime_model(f_01, training=True)
            inter_15 = self.anime_model(f_12, training=True)
            
            with tape.stop_recording():
                inter_05_p = frame_to_patch_on_batch(
                    inter_05, 
                    self.patch_shape,
                    self.overlap,
                )
                inter_15_p = frame_to_patch_on_batch(
                    inter_15,
                    self.patch_shape,
                    self.overlap,
                )

                edge_05_p = self.edge_model(inter_05_p,training=False)
                edge_05_p = edge_05_p[...,tf.newaxis]
                edge_15_p = self.edge_model(inter_15_p,training=False)
                edge_15_p = edge_15_p[...,tf.newaxis]

                edge_05 = patch_to_frame_on_batch(
                    edge_05_p,
                    self.patch_shape,
                    tf.shape(inter_05)[1:3],
                    self.overlap,
                )
                edge_15 = patch_to_frame_on_batch(
                    edge_15_p,
                    self.patch_shape,
                    tf.shape(inter_15)[1:3],
                    self.overlap,
                )
            f_0515 = tf.concat([
                inter_05,
                edge_05,
                inter_15,
                edge_15,
            ],axis=-1)

            inter_10pp, net_10pp = self.interp_flow_model(f_0515)
            cyclic_loss = self.mae(f_1[...,:3], inter_10pp)
            lin_loss = self.mse(net_10[...,:2],net_10pp[...,:2]*2)

            total_loss = recon_loss + \
                         GAMMA_CYCLIC * cyclic_loss + \
                         GAMMA_LIN * lin_loss

        trainable_vars = self.anime_model.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.total_loss_tracker.update_state(total_loss)
        self.cyclic_loss_tracker.update_state(cyclic_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.lin_loss_tracker.update_state(lin_loss)

        return {
            'L' : self.total_loss_tracker.result(),
            'Lc' : self.cyclic_loss_tracker.result(),
            'Lr' : self.recon_loss_tracker.result(),
            'Lm' : self.lin_loss_tracker.result()
        }
    
    def test_step(self, data):
        x, y = data
        y_pred = self.anime_model(x, training=False)
        recon_loss = self.mae(y, y_pred)
        self.recon_loss_tracker.update_state(recon_loss)
        return {'Lr' : self.recon_loss_tracker.result()}

    @property
    def metrics(self):
        return[
            self.total_loss_tracker,
            self.cyclic_loss_tracker,
            self.recon_loss_tracker,
            self.lin_loss_tracker,
        ]



class AugGenerator():
    """An iterable generator that makes data

        Reads 3 serial frames

        0   1   2
        X1  Y1  X2

    NOTE:: This makes data to use for cyclic training.
    Therefore, Y has 4 channels.

    return
    ------
    X : np.array, dtype= np.float32 Normalized to [0.0,1.0]
        shape : (HEIGHT, WIDTH, 8)
        * Additional one channel for edge maps
    Y : np.array, dtype= np.float32 Normalized to [0.0,1.0]
        shape : (HEIGHT, WIDTH, 4)

    """
    def __init__(self, vid_dir, edge_dir, vid_names, frame_size):
        """ 
        arguments
        ---------
        vid_dir : string or Path object
            A directory which has raw video files
        edge_dir : string or Path object
            A directory which has edge video files
        vid_names : list of strings
            List of video names. All raw video names and edge video names
            are expected to have the same name for corresponding counterpart.
        frame_size : tuple (WIDTH, HEIGHT)
            Desired output frame size
            ex) (1280,720) for 720p
        """
        self.vid_dir = Path(vid_dir)
        self.edge_dir = Path(edge_dir)
        self.vid_names = vid_names
        self.vid_n = len(self.vid_names)
        self.frame_size = frame_size
        self.aug = A.Compose([
            # A.OneOf([
            #     A.RandomGamma((40,200),p=1),
            #     A.RandomBrightness(limit=0.5, p=1),
            #     A.RandomContrast(limit=0.5,p=1),
            #     A.RGBShift(40,40,40,p=1),
            #     A.ChannelShuffle(p=1),
            # ], p=0.8),
            # A.InvertImg(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Resize(frame_size[1], frame_size[0]),
        ],
        additional_targets={
            'Y0' : 'image',
            'X1' : 'image',
            'X0e' : 'image',
            'Y0e' : 'image',
            'X1e' : 'image',
        },
        )
        self.frame_idx = 0
        self.cap_raw = None
        self.cap_edge = None

    def __iter__(self):
        return self
    
    def __call__(self, *args):
        return self

    def __next__(self):
        if self.frame_idx+6 >= 500 or (self.cap_raw is None)\
            or (self.cap_edge is None):
            self.reset_cap()

        # Throw away some frames so data will not use
        # same frames set everytime.
        for i in range(random.randrange(0,30)):
            if self.cap_raw.isOpened() and self.cap_edge.isOpened():
                ret_raw, _ = self.cap_raw.read()
                ret_edge, _ = self.cap_edge.read()
                self.frame_idx += 1
                if not (ret_raw and ret_edge):
                    self.reset_cap()
                    return self.__next__()
            else:
                self.reset_cap()
                return self.__next__()
        
        sampled_frames_raw = []
        sampled_frames_edge = []
        for i in range(3):
            if self.cap_raw.isOpened() and self.cap_edge.isOpened():
                ret_raw, frame_raw = self.cap_raw.read()
                ret_edge, frame_edge = self.cap_edge.read()
                self.frame_idx += 1
                if ret_raw and ret_edge:
                    sampled_frames_raw.append(frame_raw)
                    sampled_frames_edge.append(frame_edge)
                else:
                    self.reset_cap()
                    return self.__next__()
            else:
                self.reset_cap()
                return self.__next__()
        
        height, width = sampled_frames_raw[0].shape[:2]
        
        # frame_size : (width, height) ex) (1280, 720)
        rotate = False
        move = False
        if height>self.frame_size[0] and \
            width>self.frame_size[1] and \
            random.random()<0.3:

            crop_height = self.frame_size[0]
            crop_width = self.frame_size[1]
            rotate = True
        elif random.random()<0.5 :
            max_ratio = min(width/self.frame_size[0],height/self.frame_size[1])
            ratio = 1 + random.random()*(max_ratio-1)
            crop_height = int(self.frame_size[1]*ratio)
            crop_width = int(self.frame_size[0]*ratio)
        else:
            # Manually make movement (shift window)
            move = True
            crop_width, crop_height = self.frame_size
        crop_min = (random.randint(0, height-crop_height),
                    random.randint(0, width-crop_width))
        crop_max = (crop_min[0]+crop_height,crop_min[1]+crop_width)

        if rotate:
            cropped_frames_raw = [f[crop_min[0]:crop_max[0],
                                    crop_min[1]:crop_max[1]].swapaxes(0,1)\
                                    for f in sampled_frames_raw]
            cropped_frames_edge = [f[crop_min[0]:crop_max[0],
                                    crop_min[1]:crop_max[1]].swapaxes(0,1)\
                                    for f in sampled_frames_edge]
        elif move:
            # direction (-)
            if crop_min[0] > (height-crop_max[0]):
                delta_h_max = int((max(-2*DELTA_MAX,-crop_min[0]))/2)
                delta_h = random.randint(delta_h_max,0)
            # direction (+)
            else:
                delta_h_max = int((min(2*DELTA_MAX,height-crop_max[0]))/2)
                delta_h = random.randint(0,delta_h_max)
            # direction (-)
            if crop_min[1] > (width-crop_max[1]):
                delta_w_max = int((max(-2*DELTA_MAX,-crop_min[1]))/2)
                delta_w = random.randint(delta_w_max,0)
            # direction (+)
            else:
                delta_w_max = int((min(2*DELTA_MAX,width-crop_max[1]))/2)
                delta_w = random.randint(0,delta_w_max)
            cropped_frames_raw = [
                f[crop_min[0]+(d*delta_h):crop_max[0]+(d*delta_h),
                  crop_min[1]+(d*delta_w):crop_max[1]+(d*delta_w)]\
                for d,f in enumerate(sampled_frames_raw)
            ]
            cropped_frames_edge = [
                f[crop_min[0]+(d*delta_h):crop_max[0]+(d*delta_h),
                  crop_min[1]+(d*delta_w):crop_max[1]+(d*delta_w)]\
                for d,f in enumerate(sampled_frames_edge)
            ]
        else:
            cropped_frames_raw = [f[crop_min[0]:crop_max[0],
                                crop_min[1]:crop_max[1]]\
                                for f in sampled_frames_raw]
            cropped_frames_edge = [f[crop_min[0]:crop_max[0],
                                crop_min[1]:crop_max[1]]\
                                for f in sampled_frames_edge]

        x0, y0, x1 = cropped_frames_raw
        # Need y0_e for cyclic training
        x0_e, y0_e, x1_e = cropped_frames_edge

        transformed = self.aug(
            image=x0,
            Y0=y0,
            X1=x1,
            X0e=x0_e,
            Y0e=y0_e,
            X1e=x1_e,
        )
        x0 = transformed['image']
        y0 = transformed['Y0']
        x1 = transformed['X1']
        # Select only one channel for edge mask
        # Leave the last axis to concatenate
        x0_e = transformed['X0e'][...,0:1]
        y0_e = transformed['Y0e'][...,0:1]
        x1_e = transformed['X1e'][...,0:1]



        x_concat = np.concatenate([x0,x0_e,x1,x1_e],axis=-1).astype(np.float32)\
                    /255.0
        y_concat = np.concatenate([y0,y0_e],axis=-1).astype(np.float32)/255.0

        return x_concat, y_concat

    def reset_cap(self):
        if not(self.cap_raw is None):
            self.cap_raw.release()
        if not(self.cap_edge is None):
            self.cap_edge.release()

        target_vid = self.vid_names[random.randrange(0,self.vid_n)]
        self.cap_raw = cv2.VideoCapture(str(self.vid_dir/target_vid))
        self.cap_edge = cv2.VideoCapture(str(self.edge_dir/target_vid))
        self.frame_idx = 0

class ValGenerator(AugGenerator):
    """Same as AugGenerator, but without augmentation.

    NOTE:: Validation set does not add edge maps to Y, because
    when evaluating, edge map is not used.

    return
    ------
    X : np.array, dtype= np.float32 Normalized to [0.0,1.0]
        shape : (HEIGHT, WIDTH, 8)
        * Additional one channel for edge maps
    Y : np.array, dtype= np.float32 Normalized to [0.0,1.0]
        shape : (HEIGHT, WIDTH, 3)

    """
    def __init__(self, vid_dir, edge_dir, vid_names, frame_size):
        """ 
        arguments
        ---------
        vid_dir : string or Path object
            A directory which has raw video files
        edge_dir : string or Path object
            A directory which has edge video files
        vid_names : list of strings
            List of video names. All raw video names and edge video names
            are expected to have the same name for corresponding counterpart.
        frame_size : tuple (WIDTH, HEIGHT)
            Desired output frame size
            ex) (1280,720) for 720p
        """
        super().__init__(vid_dir, edge_dir, vid_names, frame_size)
        self.aug = A.Compose([
            A.Resize(frame_size[1], frame_size[0]),
        ],
        additional_targets={
            'Y0' : 'image',
            'X1' : 'image',
            'X0e' : 'image',
            'Y0e' : 'image',
            'X1e' : 'image',
        },
        )
    
    def __next__(self):
        x, y = super().__next__()
        return x, y[...,:3]

def create_train_dataset(
        vid_dir,
        edge_dir,
        vid_names, 
        frame_size, 
        batch_size, 
        val_data=False,
        parallel=8,
    ):
    """
    image_size : tuple
        (WIDTH, HEIGHT)
    """
    autotune = tf.data.experimental.AUTOTUNE
    num_vids = len(vid_names)
    if val_data:
        generator = ValGenerator
        y_channel = 3
    else:
        generator = AugGenerator
        y_channel = 4
    
    parallel = min(parallel, num_vids)
    indices = np.arange(parallel+1)*(num_vids//parallel)
    indices[-1] = num_vids

    dummy_ds = tf.data.Dataset.range(parallel)
    dataset = dummy_ds.interleave(
        lambda x: tf.data.Dataset.from_generator(
            lambda x: generator(
                vid_dir,
                edge_dir,
                vid_names[indices[x]:indices[x+1]],
                frame_size,
            ),
            output_signature=(
                tf.TensorSpec(shape=[frame_size[1],frame_size[0],8],
                              dtype=tf.float32),
                tf.TensorSpec(shape=[frame_size[1],frame_size[0],y_channel],
                              dtype=tf.float32),
            ),
            args=(x,)
        ),
        cycle_length=parallel,
        block_length=1,
        num_parallel_calls=parallel,
    )
    if val_data:
        dataset = dataset.shuffle(30, reshuffle_each_iteration=False)
    else:
        dataset = dataset.shuffle(500, reshuffle_each_iteration=False)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(autotune)
    dataset = dataset.repeat()

    return dataset


class ValFigCallback(keras.callbacks.Callback):
    def __init__(self, val_ds, logdir):
        super().__init__()
        self.val_ds = val_ds
        self.filewriter = tf.summary.create_file_writer(logdir+'/val_image')

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def val_result_fig(self):
        samples = self.val_ds.take(4).as_numpy_iterator()
        fig = plt.figure(figsize=(40,40))
        for i in range(4):
            sample = next(samples)
            sample_x = sample[0]
            sample_y = sample[1]
            predict = self.model(sample_x, training=False).numpy()

            ax = fig.add_subplot(8,3,6*i+1)
            x0e = sample_x[0][...,3]
            ax.imshow(x0e,cmap='binary')

            ax = fig.add_subplot(8,3,6*i+2)
            p0 = predict[0][...,2::-1]
            ax.imshow(p0)
            
            ax = fig.add_subplot(8,3,6*i+3)
            x1e = sample_x[0][...,7]
            ax.imshow(x1e,cmap='binary')

            ax = fig.add_subplot(8,3,6*i+4)
            x0 = sample_x[0][...,2::-1]
            ax.imshow(x0)

            ax = fig.add_subplot(8,3,6*i+5)
            y0 = sample_y[0][...,2::-1]
            ax.imshow(y0)
            
            # ax = fig.add_subplot(8,4,8*i+7)
            # y1 = sample_y[0][...,5:2:-1]
            # ax.imshow(y1)

            ax = fig.add_subplot(8,3,6*i+6)
            x1 = sample_x[0][...,6:3:-1]
            ax.imshow(x1)
        return fig

    def on_epoch_end(self, epoch, logs=None):
        image = self.plot_to_image(self.val_result_fig())
        with self.filewriter.as_default():
            tf.summary.image('val prediction', image, step=epoch)

def run_training(
        model_f, 
        lr_f, 
        name, 
        epochs, 
        batch_size, 
        steps_per_epoch,
        vid_dir,
        edge_dir,
        train_vid_names,
        val_vid_names,
        frame_size,
        flow_map_size,
        interpolate_ratios,
        patch_size,
        overlap,
        edge_model_f,
        mixed_float = True,
        notebook = True,
        profile = False,
        edge_model_path=None,
        amodel_path = None,
        load_model_path = None,
    ):
    """
    patch_size, frame_size and flow_map_size are all
        (WIDTH, HEIGHT) format
    """
    if ((edge_model_path is None) or (amodel_path is None))\
        and (load_model_path is None):
        raise ValueError('Need a path to load model')
    if mixed_float:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
    
    st = time.time()

    a_model = anime_model(
        model_f,
        interpolate_ratios,
        flow_map_size
    )
    e_model = EdgeModel([patch_size[1],patch_size[0],3], edge_model_f)

    if amodel_path is not None:
        a_model.load_weights(amodel_path).expect_partial()
        print('*'*50)
        print(f'Anime model loaded from : {amodel_path}')
        print('*'*50)

    if edge_model_path is not None:
        e_model.load_weights(edge_model_path).expect_partial()
        print('*'*50)
        print(f'Edge model loaded from : {edge_model_path}')
        print('*'*50)

    c_model = AnimeModelCyclic(
        a_model,
        e_model,
        (patch_size[1],patch_size[0]),
        overlap,
    )
    if load_model_path is not None:
        c_model.load_weights(load_model_path)
        print('*'*50)
        print(f'Cyclic model loaded from : {load_model_path}')
        print('*'*50)
    c_model.compile(optimizer='adam')


    logdir = 'logs/fit/' + name
    if profile:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=logdir,
            histogram_freq=1,
            profile_batch='3,5',
            update_freq='epoch'
        )
    else :
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=logdir,
            histogram_freq=1,
            profile_batch=0,
            update_freq='epoch'
        )

    lr_callback = keras.callbacks.LearningRateScheduler(lr_f, verbose=1)

    savedir = 'savedmodels/' + name + '/{epoch}'
    save_callback = keras.callbacks.ModelCheckpoint(
        savedir,
        save_weights_only=True,
        verbose=1
    )

    if notebook:
        tqdm_callback = TqdmNotebookCallback(metrics=['loss'],
                                            leave_inner=False)
    else:
        tqdm_callback = TqdmCallback()

    train_ds = create_train_dataset(vid_dir,edge_dir, train_vid_names,
                                    frame_size,batch_size, parallel=6)
    val_ds = create_train_dataset(vid_dir,edge_dir,val_vid_names,
                                  frame_size,batch_size,
                                  val_data=True, parallel=4)

    image_callback = ValFigCallback(val_ds, logdir)

    c_model.fit(
        x=train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[
            tensorboard_callback,
            lr_callback,
            save_callback,
            tqdm_callback,
            image_callback,
        ],
        verbose=0,
        validation_data=val_ds,
        validation_steps=50,
    )

    delta = time.time()-st
    hours, remain = divmod(delta, 3600)
    minutes, seconds = divmod(remain, 60)
    print(f'Took {hours:.0f} hours {minutes:.0f} minutes {seconds:.2f} seconds')


if __name__ == '__main__':
    from flow_models_functional import *
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    test_model = anime_model(
        ehrb0_143_32,
        [0.5],
        (512,288),
    )
    test_model.compile(
        run_eagerly=True
    )
    import numpy as np
    test_model(np.random.random((7,540,960,8)))
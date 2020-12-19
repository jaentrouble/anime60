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
from extra_models.deep_voxel import VoxelInterp
import os
from pathlib import Path
import cv2

DELTA_MAX = 60

class AnimeModel(keras.Model):
    def __init__(self, model_function, interpolate_ratios,
                       flow_map_size,):
        """Gets 2 frames and returns interpolated frames

        Args
        ----
        inputs : keras.Input
            Expects [0,1] range normalized frames
            shape : (N,H,W,2*C) where two frames are concatenated

        model_function : function that takes keras.Input and returns
        encoded image

        interpolate_ratios: list
            ex) [0.5] would make a single frame of 1/2 position.
            ex) [0.4,0.8] would make two frames at 2/5, 4/5 position.

        flow_map_size: tuple of ints
            format: (H,W)
            Will resize any shape of input to flow_map_size
            Flow map will then be resized again to the original size

        Output
        ------
        outputs : (N,H,W,C*F) where F is number of interpolated frames.
            Concatenated as R0,B0,G0, R1,B1,G1 ...
            Returns [0,1] range normalized frames

        """
        super().__init__()
        
        self.model_function = model_function
        self.interpolate_ratios = interpolate_ratios
        self.flow_map_size = flow_map_size
        
        inputs = keras.Input((flow_map_size[0],flow_map_size[1],6))
        encoded = model_function(inputs)
        self.encoder= keras.Model(inputs=inputs, outputs=encoded, name='encoder')
        self.encoder.summary()
        self.interpolator = VoxelInterp(interpolate_ratios, dtype=tf.float32)
        
    def call(self, inputs, training=None):
        inputs = tf.cast(inputs, tf.float32)
        resized_inputs = tf.image.resize(inputs, self.flow_map_size, 
                                         name='downscale')
        encoded = self.encoder(resized_inputs)
        interpolated = self.interpolator([inputs, encoded], training=training)
        return interpolated
    
    def get_config(self):
        config = super().get_config()
        config['model_function'] = self.model_function
        config['interpolate_ratios'] = self.interpolate_ratios

class AugGenerator():
    """An iterable generator that makes data

        Reads 3 serial frames

        0   1   2
        X1  Y1  X2

    return
    ------
    X : np.array, dtype= np.float32 Normalized to [0.0,1.0]
        shape : (HEIGHT, WIDTH, 6)
    Y : np.array, dtype= np.float32 Normalized to [0.0,1.0]
        shape : (HEIGHT, WIDTH, 3)

    """
    def __init__(self, vid_paths, frame_size):
        """ 
        arguments
        ---------
        vid_paths : list of strings or Path objects
            Each video should have more than 500 frames.

        frame_size : tuple (WIDTH, HEIGHT)
            Desired output frame size
            ex) (1280,720) for 720p
        """
        self.vid_paths = vid_paths
        self.vid_n = len(self.vid_paths)
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
            # 'Y1' : 'image',
            'X1' : 'image',
        },
        )
        self.aug_noise = A.Compose([
            A.GaussNoise((0.0, 0.0),p=0.0)
        ],
        additional_targets={
            'X1' : 'image',
        },
        )
        self.frame_idx = 0
        self.cap = None

    def __iter__(self):
        return self
    
    def __call__(self, *args):
        return self

    def __next__(self):
        if self.frame_idx+6 >= 500 or (self.cap is None):
            self.reset_cap()

        # Throw away some frames so data will not use
        # same frames set everytime.
        for i in range(random.randrange(0,30)):
            if self.cap.isOpened():
                ret, _ = self.cap.read()
                self.frame_idx += 1
                if not ret:
                    self.reset_cap()
                    return self.__next__()
            else:
                self.reset_cap()
                return self.__next__()
        
        sampled_frames = []
        for i in range(3):
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                self.frame_idx += 1
                if ret:
                    sampled_frames.append(frame)
                else:
                    self.reset_cap()
                    return self.__next__()
            else:
                self.reset_cap()
                return self.__next__()
        
        height, width = sampled_frames[0].shape[:2]
        
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
            cropped_frames = [f[crop_min[0]:crop_max[0],
                                crop_min[1]:crop_max[1]].swapaxes(0,1)\
                                for f in sampled_frames]
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
            cropped_frames = [
                f[crop_min[0]+(d*delta_h):crop_max[0]+(d*delta_h),
                  crop_min[1]+(d*delta_w):crop_max[1]+(d*delta_w)]\
                for d,f in enumerate(sampled_frames)
            ]
        else:
            cropped_frames = [f[crop_min[0]:crop_max[0],
                                crop_min[1]:crop_max[1]]\
                                for f in sampled_frames]
        x0, y0, x1 = cropped_frames

        transformed = self.aug(
            image=x0,
            Y0=y0,
            # Y1=y1,
            X1=x1,
        )
        x0 = transformed['image']
        x1 = transformed['X1']
        y0 = transformed['Y0']
        # y1 = transformed['Y1']

        noised = self.aug_noise(
            image=x0,
            X1 =x1
        )

        x0 = noised['image']
        x1 = noised['X1']

        x_concat = np.concatenate([x0,x1],axis=-1).astype(np.float32)/255.0
        y_concat = y0.astype(np.float32)/255.0

        return x_concat, y_concat

    def reset_cap(self):
        if not(self.cap is None):
            self.cap.release()
        vid_idx = random.randrange(0,self.vid_n)
        self.cap = cv2.VideoCapture(self.vid_paths[vid_idx])
        self.frame_idx = 0

class ValGenerator(AugGenerator):
    """Same as AugGenerator, but without augmentation.
    """
    def __init__(self, vid_paths, frame_size):
        """ 
        arguments
        ---------
        vid_paths : list of strings or Path objects
            Each video should have more than 500 frames.

        frame_size : tuple (WIDTH, HEIGHT)
            Desired output frame size
            ex) (1280,720) for 720p
        """
        super().__init__(vid_paths, frame_size)
        self.aug = A.Compose([
            A.Resize(frame_size[1], frame_size[0]),
        ],
        additional_targets={
            'Y0' : 'image',
            # 'Y1' : 'image',
            'X1' : 'image',
        },
        )
        self.aug_noise = A.Compose([
            A.GaussNoise(0.0,p=0)
        ],
        additional_targets={
            'X1' : 'image',
        },
        )

def create_train_dataset(
        vid_paths, 
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
    num_vids = len(vid_paths)
    if val_data:
        generator = ValGenerator
    else:
        generator = AugGenerator
    
    parallel = min(parallel, num_vids)
    indices = np.arange(parallel+1)*(num_vids//parallel)
    indices[-1] = num_vids

    dummy_ds = tf.data.Dataset.range(parallel)
    dataset = dummy_ds.interleave(
        lambda x: tf.data.Dataset.from_generator(
            lambda x: generator(
                vid_paths[indices[x]:indices[x+1]],
                frame_size,
            ),
            output_signature=(
                tf.TensorSpec(shape=[frame_size[1],frame_size[0],6],
                              dtype=tf.float32),
                tf.TensorSpec(shape=[frame_size[1],frame_size[0],3],
                              dtype=tf.float32),
            ),
            args=(x,)
        ),
        cycle_length=parallel,
        block_length=1,
        num_parallel_calls=parallel,
    )
    if val_data:
        dataset = dataset.shuffle(50, reshuffle_each_iteration=False)
    else:
        dataset = dataset.shuffle(1700, reshuffle_each_iteration=False)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(autotune)
    dataset = dataset.repeat()

    return dataset


def get_model(frame_size, model_f, interpolate_ratios, flow_map_size):
    """
    simple wrapper around a model (To use keras.Model functional api)
    
    Arguments
    ---------
    frame_size : tuple of two ints
        format (W, H)
    flow_map_size: tuple of two ints
        format (W, H)
    """
    flow_map_size_hw = (flow_map_size[1], flow_map_size[0])
    inputs = keras.Input((frame_size[1],frame_size[0],6))
    raw_model = AnimeModel(model_f, interpolate_ratios, flow_map_size_hw)
    outputs = raw_model(inputs)
    mymodel = keras.Model(inputs=inputs, outputs=outputs)
    mymodel.summary()
    return mymodel

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
            x0 = sample_x[0][...,2::-1]
            ax.imshow(x0)

            ax = fig.add_subplot(8,3,6*i+2)
            p0 = predict[0][...,2::-1]
            ax.imshow(p0)
            
            # ax = fig.add_subplot(8,4,8*i+3)
            # p1 = predict[0][...,5:2:-1]
            # ax.imshow(p1)

            ax = fig.add_subplot(8,3,6*i+3)
            x1 = sample_x[0][...,5:2:-1]
            ax.imshow(x1)

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
            x1 = sample_x[0][...,5:2:-1]
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
        train_vid_paths,
        val_vid_paths,
        test_vid_paths,
        frame_size,
        flow_map_size,
        interpolate_ratios,
        mixed_float = True,
        notebook = True,
        profile = False,
        load_model_path = None,
    ):
    """
    frame_size and flow_map_size are both
        (WIDTH, HEIGHT) format
    """
    if mixed_float:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
    
    st = time.time()

    mymodel = AnimeModel(model_f, interpolate_ratios, flow_map_size)

    if load_model_path:
        mymodel.load_weights(load_model_path)
        print('*'*50)
        print(f'Loaded from : {load_model_path}')
        print('*'*50)
    loss = keras.losses.MeanAbsoluteError()
    mymodel.compile(
        optimizer='adam',
        loss=loss,
    )

    logdir = 'logs/fit/' + name
    if profile:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=logdir,
            histogram_freq=1,
            profile_batch='3,5',
            update_freq=steps_per_epoch
        )
    else :
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=logdir,
            histogram_freq=1,
            profile_batch=0,
            update_freq=steps_per_epoch
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

    train_ds = create_train_dataset(train_vid_paths,frame_size,batch_size, 
                                    parallel=6)
    val_ds = create_train_dataset(val_vid_paths,frame_size,batch_size,
                                    val_data=True, parallel=6)

    image_callback = ValFigCallback(val_ds, logdir)

    mymodel.fit(
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
        validation_steps=10,
    )

    delta = time.time()-st
    hours, remain = divmod(delta, 3600)
    minutes, seconds = divmod(remain, 60)
    print(f'Took {hours:.0f} hours {minutes:.0f} minutes {seconds:.2f} seconds')

    # test_ds = create_train_dataset(test_vid_paths,frame_size,batch_size,True)
    # mymodel.evaluate(test_ds, steps=600)

if __name__ == '__main__':
    from flow_models import *
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

    inputs = keras.Input((1280,720,6))

    lr_f = lambda x : 1.0
    lr_callback = keras.callbacks.LearningRateScheduler(lr_f, verbose=1)

    logdir = 'logs/fit/test'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=1,
        profile_batch=0,
        update_freq='epoch'
    )

    model = AnimeModel(inputs, hr_5_3_0, [0.4, 0.8])
    model.compile(optimizer='adam',loss='mse')
    sample_x = np.random.random((10,1280,720,6))
    sample_y = np.random.random((10,1280,720,6))
    model.fit(x=sample_x, y=sample_y,batch_size=1 ,epochs=10,
    callbacks=[lr_callback,tensorboard_callback])
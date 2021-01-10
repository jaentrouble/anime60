import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision, layers
import time
from custom_tqdm import TqdmNotebookCallback
from tqdm.keras import TqdmCallback
import albumentations as A
import random
import io
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import os
from pathlib import Path
import cv2


class EdgeModel(keras.Model):
    def __init__(self, input_shape, model_function,):
        """Gets an image and returns a edge map

        For numerical stability, softmax layer is only applied
        when training = False.
        When defining loss, set 'from logits = True'

        Expects input images to be normalized in the range of [0.0,1.0]

        Args
        ----
        input_shape : tuple of ints
            Expected input shape. Input images should have the same shape.
            (H, W, C) 

        model_function : function that takes keras.Input and returns a
            feature map

        Output
        ------
        outputs : Squeezed edge map
            (H, W)

        """
        super().__init__()
        
        self.model_function = model_function
        self.image_shape = input_shape
        
        inputs = keras.Input(input_shape)
        encoded = model_function(inputs)

        one_channel = layers.Conv2D(1, 3, padding='same', activation='linear',
                                          dtype='float32')(encoded)
        squeezed = tf.squeeze(one_channel, axis=-1)

        self.logits= keras.Model(inputs=inputs, outputs=squeezed, name='encoder')
        self.logits.summary()
        
    def call(self, inputs, training=None):
        inputs = tf.cast(inputs, tf.float32)
        if training:
            return self.logits(inputs, training=training)
        return tf.math.sigmoid(self.logits(inputs, training=training))
    
    def get_config(self):
        config = super().get_config()
        config['model_function'] = self.model_function
        config['input_shape'] = self.image_shape

class AugGenerator():
    """An iterable generator that makes data

    return
    ------
    X : np.array, dtype= np.float32 Normalized to [0.0,1.0]
        An input raw image
        shape : (HEIGHT, WIDTH, 3)
    Y : np.array, dtype= np.float32 Normalized to [0.0,1.0]
        Ground truth edge map
        shape : (HEIGHT, WIDTH)

    """
    def __init__(self, image_directory, edge_directory, image_size):
        """ 
        All images and corresponding edge maps are expected to have
        the same file name.

        arguments
        ---------
        image_directory : str or Path object
            Path to directory which has raw images
        
        edge_directory : str or Path object
            Path to directory which has edge maps
            All edge images are expected to be .png files

        image_size : tuple (WIDTH, HEIGHT)
            Desired output image size
            ex) (1280,720)
        """
        self.image_dir = Path(image_directory)
        self.edge_dir = Path(edge_directory)
        self.image_size = image_size
        self.images = []
        self.edge_maps = []
        for img_path in self.image_dir.iterdir():
            img_name = img_path.stem
            edge_path = self.edge_dir/f'{img_name}.png'
            self.images.append(
                cv2.cvtColor(cv2.imread(
                    str(img_path), cv2.IMREAD_COLOR
                ), cv2.COLOR_BGR2RGB)
            )
            self.edge_maps.append(
                cv2.imread(
                    str(edge_path), cv2.IMREAD_GRAYSCALE
                )
            )
        self.image_n = len(self.images)
        self.aug = A.Compose([
            A.RandomRotate90(p=1),
            A.RandomCrop(self.image_size[1],self.image_size[0],p=1),
            A.OneOf([
                A.RandomGamma((40,200),p=1),
                A.RandomBrightness(limit=0.5, p=1),
                A.RandomContrast(limit=0.5,p=1),
                A.RGBShift(40,40,40,p=1),
                A.ChannelShuffle(p=1),
                A.GaussNoise(p=1)
            ], p=0.8),
            A.InvertImg(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            # A.Resize(self.image_size[1],self.image_size[0]),
        ],)

    def __iter__(self):
        return self
    
    def __call__(self, *args):
        return self

    def __next__(self):
        idx = random.randrange(0,self.image_n)
        distorted = self.aug(
            image = self.images[idx],
            mask=self.edge_maps[idx],
        )
        X = distorted['image'].astype(np.float32)/255
        Y = np.round(distorted['mask'].astype(np.float32)/255)
        return X, Y


class ValGenerator(AugGenerator):
    """Same as AugGenerator, but without augmentation.
    """
    def __init__(self, image_directory, edge_directory, image_size):
        """ 
        All images and corresponding edge maps are expected to have
        the same file name.

        arguments
        ---------
        image_directory : str or Path object
            Path to directory which has raw images
        
        edge_directory : str or Path object
            Path to directory which has edge maps
            All edge images are expected to be .png files

        image_size : tuple (WIDTH, HEIGHT)
            Desired output image size
            ex) (1280,720)
        """
        super().__init__(image_directory, edge_directory, image_size)
        self.aug = A.Compose([
            A.RandomCrop(self.image_size[1],self.image_size[0],p=1),
        ],)

def create_train_dataset(
        data_dir, 
        image_size, 
        batch_size, 
        val_data=False,
        parallel=8,
    ):
    """
    image_size : tuple
        (WIDTH, HEIGHT)
    """
    autotune = tf.data.experimental.AUTOTUNE
    if val_data:
        generator = ValGenerator
    else:
        generator = AugGenerator
    
    dummy_ds = tf.data.Dataset.range(parallel)
    dataset = dummy_ds.interleave(
        lambda x: tf.data.Dataset.from_generator(
            lambda x: generator(
                Path(data_dir)/'images',
                Path(data_dir)/'edges',
                image_size,
            ),
            output_signature=(
                tf.TensorSpec(shape=[image_size[1],image_size[0],3],
                              dtype=tf.float32),
                tf.TensorSpec(shape=[image_size[1],image_size[0]],
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
        samples = self.val_ds.take(8).as_numpy_iterator()
        fig = plt.figure(figsize=(10,20))
        for i in range(8):
            sample = next(samples)
            sample_x = sample[0]
            sample_y = sample[1]
            predict = self.model(sample_x, training=False).numpy()

            ax = fig.add_subplot(8,3,3*i+1)
            img = sample_x[0]
            ax.imshow(img)

            ax = fig.add_subplot(8,3,3*i+2)
            pred = predict[0]
            ax.imshow(pred, cmap='binary')
            
            ax = fig.add_subplot(8,3,3*i+3)
            gt = sample_y[0]
            ax.imshow(gt, cmap='binary')
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
        train_data_dir,
        val_data_dir,
        image_size,
        mixed_float = True,
        notebook = True,
        profile = False,
        load_model_path = None,
    ):
    """
    image_size:
        (WIDTH, HEIGHT) format
    """
    if mixed_float:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
    
    st = time.time()

    mymodel = EdgeModel([image_size[1],image_size[0],3], model_f)

    if load_model_path:
        mymodel.load_weights(load_model_path)
        print('*'*50)
        print(f'Loaded from : {load_model_path}')
        print('*'*50)
    loss = keras.losses.BinaryCrossentropy(from_logits=True)
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

    train_ds = create_train_dataset(train_data_dir, image_size, batch_size, 
                                    parallel=6)
    val_ds = create_train_dataset(val_data_dir, image_size, batch_size, 
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
        validation_steps=30,
    )

    delta = time.time()-st
    hours, remain = divmod(delta, 3600)
    minutes, seconds = divmod(remain, 60)
    print(f'Took {hours:.0f} hours {minutes:.0f} minutes {seconds:.2f} seconds')


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
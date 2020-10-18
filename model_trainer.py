import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import time
from custom_tqdm import TqdmNotebookCallback
from tqdm.keras import TqdmCallback
import albumentations as A
import random
import io
import matplotlib.pyplot as plt
from functools import partial
import numpy as np

class AdiposeModel(keras.Model):
    def __init__(self, inputs, model_function):
        """
        Because of numerical stability, softmax layer should be
        taken out, and use it only when not training.
        Args
            inputs : keras.Input
            model_function : function that takes keras.Input and returns
            output tensor of logits
        """
        super().__init__()
        outputs = model_function(inputs)
        self.logits = keras.Model(inputs=inputs, outputs=outputs)
        self.logits.summary()
        
    def call(self, inputs, training=None):
        casted = tf.cast(inputs, tf.float32) / 255.0
        if training:
            return self.logits(inputs, training=training)
        return tf.math.sigmoid(self.logits(inputs, training=training))

class AugGenerator():
    """An iterable generator that makes data

    NOTE: 
        Every img is reshaped to img_size
    NOTE: 
        The position value is like pygame. (width, height),
        which does not match with common image order (height,width)

        Image input is expected to be the shape of (height, width),
        i.e. the transformation to match two is handled in here automatically
    NOTE: 
        THE OUTPUT IMAGE WILL BE (WIDTH, HEIGHT)
        It is because pygame has shape (width, height)
    return
    ------
    X : np.array, dtype= np.uint8
        shape : (WIDTH, HEIGHT, 3)
    Y : np.array, dtype= np.float32
    """
    def __init__(self, img, data, img_size):
        """ 
        arguments
        ---------
        img : list
            list of images, in the original size (height, width, 3)
        data : list of dict
            Each dict has :
                'image' : index of the image. The index should match with img
                'mask' : [xx, yy]
                        IMPORTANT : (WIDTH, HEIGHT)
                'box' : [[xmin, ymin], [xmax,ymax]]
                'size' : the size of the image that the data was created with
                        IMPORTANT : (WIDTH, HEIGHT)
        img_size : tuple
            Desired output image size
            The axes will be swapped to match pygame.
            IMPORTANT : (WIDTH, HEIGHT)
        """
        self.image = img
        self.data = data
        self.n = len(data)
        self.output_size = img_size
        self.aug = A.Compose([
            A.OneOf([
                A.RandomGamma((40,200),p=1),
                A.RandomBrightness(limit=0.5, p=1),
                A.RandomContrast(limit=0.5,p=1),
                A.RGBShift(40,40,40,p=1),
                A.Downscale(scale_min=0.25,scale_max=0.5,p=1),
                A.ChannelShuffle(p=1),
            ], p=0.8),
            A.InvertImg(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1),
            A.Resize(img_size[0], img_size[1]),
        ],
        )
        for datum in data:
            datum['mask_min'] = np.min(datum['mask'], axis=1)
            datum['mask_max'] = np.max(datum['mask'], axis=1) + 1

    def __iter__(self):
        return self
    
    def __call__(self, *args):
        return self

    def __next__(self):
        idx = random.randrange(0,self.n)
        datum = self.data[idx]
        image = self.image[datum['image']]
        x_min, y_min = datum['mask_min']
        x_max, y_max = datum['mask_max']

        crop_min = (max(0, x_min-random.randrange(5,30)),
                    max(0, y_min-random.randrange(5,30)))
        crop_max = (min(datum['size'][0],x_max+random.randrange(5,30)),
                    min(datum['size'][1],y_max+random.randrange(5,30)))
        new_mask = np.zeros(np.subtract(crop_max, crop_min), dtype=np.float32)
        xx, yy = np.array(datum['mask'],dtype=np.int32)
        m_xx = xx - crop_min[0]
        m_yy = yy - crop_min[1]
        new_mask[m_xx,m_yy] = 1

        if np.any(np.not_equal(image.shape[:2], np.flip(datum['size']))):
            row_ratio = image.shape[0] / datum['size'][1]
            col_ratio = image.shape[1] / datum['size'][0]
            cx_min = int(col_ratio*crop_min[0])
            cy_min = int(row_ratio*crop_min[1])
            
            cx_max = int(col_ratio*crop_max[0])
            cy_max = int(row_ratio*crop_max[1])

            cropped_image = np.swapaxes(image[cy_min:cy_max,cx_min:cx_max],0,1)
        else:
            cropped_image = np.swapaxes(
                image[crop_min[1]:crop_max[1],crop_min[0]:crop_max[0]],
                0, 
                1,
            )
        
        distorted = self.aug(
            image=cropped_image,
            mask =new_mask
        )

        return distorted['image'], distorted['mask']

class ValGenerator(AugGenerator):
    """Same as AugGenerator, but without augmentation.
    """
    def __init__(self, img, data, img_size):
        """ 
        arguments
        ---------
        img : list
            list of images, in the original size (height, width, 3)
        data : list of dict
            Each dict has :
                'image' : index of the image. The index should match with img
                'mask' : [xx, yy]
                        IMPORTANT : (WIDTH, HEIGHT)
                'box' : [[xmin, ymin], [xmax,ymax]]
                'size' : the size of the image that the data was created with
                        IMPORTANT : (WIDTH, HEIGHT)
        img_size : tuple
            Desired output image size
            The axes will be swapped to match pygame.
            IMPORTANT : (WIDTH, HEIGHT)
        """
        super().__init__(img, data, img_size)
        self.aug = A.Resize(img_size[0], img_size[1])

def create_train_dataset(img, data, img_size, batch_size, val_data=False):
    autotune = tf.data.experimental.AUTOTUNE
    if val_data:
        generator = ValGenerator(img, data, img_size)
    else:
        generator = AugGenerator(img, data, img_size)
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.uint8, tf.float32),
        output_shapes=(
            tf.TensorShape([img_size[0],img_size[1],3]), 
            tf.TensorShape(img_size)
        ),
    )
    dataset = dataset.shuffle(min(len(data),1000))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(autotune)
    dataset = dataset.repeat()

    return dataset


def get_model(model_f):
    """
    To get model only and load weights.
    """
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_policy(policy)
    inputs = keras.Input((200,200,3))
    test_model = AdiposeModel(inputs, model_f)
    test_model.compile(
        optimizer='adam',
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.BinaryAccuracy(threshold=0.1),
        ]
    )
    return test_model

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
        sample = self.val_ds.take(1).as_numpy_iterator()
        sample = next(sample)
        sample_x = sample[0]
        sample_y = sample[1]
        predict = self.model(sample_x, training=False).numpy()
        fig = plt.figure()
        for i in range(3):
            ax = fig.add_subplot(3,3,3*i+1)
            img = sample_x[i]
            ax.imshow(img)
            ax = fig.add_subplot(3,3,3*i+2)
            true_mask = sample_y[i]
            ax.imshow(true_mask, cmap='binary')
            ax = fig.add_subplot(3,3,3*i+3)
            p = predict[i]
            ax.imshow(p, cmap='binary')
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
        train_data,
        val_data,
        img,
        img_size,
        mixed_float = True,
        notebook = True,
    ):
    """
    val_data : (X_val, Y_val) tuple
    """
    if mixed_float:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
    
    st = time.time()

    inputs = keras.Input((200,200,3))
    mymodel = AdiposeModel(inputs, model_f)
    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    mymodel.compile(
        optimizer='adam',
        loss=loss,
        metrics=[
            keras.metrics.BinaryAccuracy(threshold=0.5),
        ]
    )

    logdir = 'logs/fit/' + name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=1,
        profile_batch='3,5',
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
        tqdm_callback = TqdmNotebookCallback(metrics=['loss', 'binary_accuracy'],
                                            leave_inner=False)
    else:
        tqdm_callback = TqdmCallback()

    train_ds = create_train_dataset(img, train_data, img_size,batch_size)
    val_ds = create_train_dataset(img, val_data, img_size,batch_size,True)

    image_callback = ValFigCallback(val_ds, logdir)

    mymodel.fit(
        x=train_ds,
        epochs=epochs,
        steps_per_epoch=len(train_data)//batch_size,
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


    print('Took {} seconds'.format(time.time()-st))

    mymodel.evaluate(val_ds, steps=1000)

if __name__ == '__main__':
    import os
    import imageio as io
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import draw
    import cv2
    from pathlib import Path

    data_dir = Path('data')
    data_groups = next(os.walk(data_dir))[1]
    img = []
    data = []
    img_name_dict = {}
    img_idx = 0
    for dg in data_groups[:]:
        img_dir = data_dir/dg/'done'
        img_names = os.listdir(img_dir)
        for name in img_names:
            img_path = str(img_dir/name)
            img.append(io.imread(img_path))
            img_name_dict[img_path] = img_idx
            img_idx += 1

        json_dir = data_dir/dg/'save'
        json_names = os.listdir(json_dir)
        dg_data = []
        for name in json_names[:]:
            with open(str(json_dir/name),'r') as j:
                dg_data.extend(json.load(j))
        for dg_datum in dg_data :
            long_img_name = str(img_dir/dg_datum['image'])
            dg_datum['image'] = img_name_dict[long_img_name]
        data.extend(dg_data)

    # fig = plt.figure()
    # d_idx = random.randrange(0,len(data)-5)
    # for i, d in enumerate(data[d_idx:d_idx+5]):
    #     image = img[d['image']].copy()
    #     image = cv2.resize(image, (1200,900), interpolation=cv2.INTER_LINEAR)
    #     mask = d['mask']
    #     m_idx = random.randrange(0,len(mask[0]))
    #     pos = (mask[0][m_idx], mask[1][m_idx])
    #     boxmin = d['box'][0]
    #     boxmax = d['box'][1]
    #     rr, cc = draw.disk((pos[1],pos[0]),5)
    #     image[rr, cc] = [0,255,0]
    #     rr, cc = draw.rectangle_perimeter((boxmin[1],boxmin[0]),(boxmax[1],boxmax[0]))
    #     image[rr,cc] = [255,0,0]
    #     image[mask[1],mask[0]] = [100,100,100]
    #     ax = fig.add_subplot(5,1,i+1)
    #     ax.imshow(image)
    # plt.show()

    # gen = AugGenerator(img, data, (400,400))
    # s = next(gen)

    ds = create_train_dataset(img, data, (200,200),1, False)
    sample = ds.take(5).as_numpy_iterator()
    fig = plt.figure()
    for i, s in enumerate(sample):
        ax = fig.add_subplot(5,2,2*i+1)
        img = s[0][0].swapaxes(0,1)
        ax.imshow(img)
        ax = fig.add_subplot(5,2,2*i+2)
        mask = s[1][0].swapaxes(0,1)
        ax.imshow(mask)
    plt.show()
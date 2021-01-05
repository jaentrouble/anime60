import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inputs= keras.Input((100,100,3))
x = layers.Conv2D(3,3,padding='same')(inputs)
b, h, w = tf.shape(x)[0:3]
model = keras.Model(inputs=inputs, outputs=h)
model.summary()
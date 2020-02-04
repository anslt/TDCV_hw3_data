import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


## configeration of first convolution layer
conv_1=tf.keras.layers.Conv2D(filters=16, 
                       kernel_size=(8,8),
                       strides=(1,1),
                       padding='valid',
                       data_format='channels_last',
                       # dilation rate
                       activation='relu',
                       use_bias=True,
                       kernel_initializer='glorot_normal',
                       bias_initializer='Zeros',
                       dtype=tf.float32,
                       input_shape=(64,64,3),
                       )#kernel_regularizer=tf.keras.regularizers.l2(10e-6))
## configeration of first convolution layer


## configeration of first max pooling layer
mp_1=tf.keras.layers.MaxPooling2D(pool_size=(2, 2), 
                            strides=None, 
                            padding='valid', 
                            data_format=None,
                            dtype=tf.float32)
## configeration of first max pooling layer


## configeration of second convolution layer
conv_2=tf.keras.layers.Conv2D(filters=7, 
                       kernel_size=(5,5),
                       strides=(1,1),
                       padding='valid',
                       data_format='channels_last',
                       # dilation rate
                       activation='relu',
                       use_bias=True,
                       kernel_initializer='glorot_normal',
                       bias_initializer='Zeros',
                       )#kernel_regularizer=tf.keras.regularizers.l2(10e-6),
                       dtype=tf.float32)
## configeration of second convolution layer



## configeration of second max pooling layer
mp_2=tf.keras.layers.MaxPooling2D(pool_size=(2, 2), 
                            strides=None, 
                            padding='valid', 
                            data_format=None,
                            dtype=tf.float32)
## configeration of second max pooling layer


## flattening
flatten=tf.keras.layers.Flatten()
## flattening


## configeration of first full connected layer
fc_1=tf.keras.layers.Dense(256, 
                      activation='relu', 
                      use_bias=True,
                      kernel_initializer='glorot_normal',
                      bias_initializer='Zeros',
                      )#kernel_regularizer=tf.keras.regularizers.l2(10e-6))
## configeration of first full connected layer


## configeration of second full connected layer
fc_2=tf.keras.layers.Dense(16, 
                      activation='relu', 
                      use_bias=True,
                      kernel_initializer='glorot_normal',
                      bias_initializer='Zeros',
                      )#kernel_regularizer=tf.keras.regularizers.l2(10e-6))
## configeration of second full connected layer




## defining the loss
def custom_loss():
    def full_loss(y_true=0,y_pred=fc_2): 
        #default m
        m = 0.01
        batch_size=tf.shape(y_pred)[0]
        #||f(x_a)-f(x+)|| and ||f(x_a)-f(x-)||
        diff_pos = tf.subtract(y_pred[0: batch_size: 3], y_pred[1: batch_size: 3])
        diff_neg = tf.subtract(y_pred[0: batch_size: 3], y_pred[2: batch_size: 3])

        #print(diff_pos.eval().shape)

        #||f(x_a)-f(x+)||^2 and ||f(x_a)-f(x-)||^2
        square_pos = tf.square(diff_pos)
        square_neg = tf.square(diff_neg)

        #print(square_pos.eval().shape)

        pair_loss = tf.reduce_sum(square_pos)
        tf.summary.scalar('loss_pairs', pair_loss)
        
        norm_2_pos=tf.reduce_sum(square_pos, axis=1)
        norm_2_neg=tf.reduce_sum(square_neg, axis=1)

        #print(norm_2_pos.eval().shape)

        triplet_loss = tf.reduce_sum(tf.maximum(0.0, tf.subtract(1.0, tf.divide(norm_2_neg, tf.add(norm_2_pos, m)))))
        tf.summary.scalar('triplet_loss', triplet_loss)
        
        full_loss = tf.add(pair_loss, triplet_loss)
        tf.summary.scalar('full_loss',full_loss )
        return full_loss
    return full_loss
## defining the loss





















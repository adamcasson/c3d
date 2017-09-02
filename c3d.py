# -*- coding: utf-8 -*-
"""C3D model for Keras

# Reference:

- [Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/abs/1412.0767)

Based on code from @albertomontesg
"""

import skvideo.io
import keras.backend as K
from keras.models import Sequential
from keras.utils.data_utils import get_file
from keras.layers.core import Dense, Dropout, Flatten
from sports1M_utils import preprocess_input, decode_predictions
from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D

WEIGHTS_PATH = 'https://github.com/adamcasson/c3d/releases/download/v0.1/sports1M_weights_tf.h5'

def C3D(weights='sports1M'):
    """Instantiates a C3D Kerasl model
    
    Keyword arguments:
    weights -- weights to load into model. (default is sports1M)
    
    Returns:
    A Keras model.
    
    """
    
    if weights not in {'sports1M', None}:
        raise ValueError('weights should be either be sports1M or None')
    
    if K.image_data_format() == 'channels_last':
        shape = (16,112,112,3)
    else:
        shape = (3,16,112,112)
        
    model = Sequential()
    model.add(Conv3D(64, 3, activation='relu', padding='same', name='conv1', input_shape=shape))
    model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='same', name='pool1'))
    
    model.add(Conv3D(128, 3, activation='relu', padding='same', name='conv2'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool2'))
    
    model.add(Conv3D(256, 3, activation='relu', padding='same', name='conv3a'))
    model.add(Conv3D(256, 3, activation='relu', padding='same', name='conv3b'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool3'))
    
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv4a'))
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv4b'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool4'))
    
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv5a'))
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv5b'))
    model.add(ZeroPadding3D(padding=(0,1,1)))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool5'))
    
    model.add(Flatten())
    
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(0.5))
    model.add(Dense(487, activation='softmax', name='fc8'))

    if weights == 'sports1M':
        weights_path = get_file('sports1M_weights_tf.h5',
                                WEIGHTS_PATH,
                                cache_subdir='models',
                                md5_hash='b7a93b2f9156ccbebe3ca24b41fc5402')
        
        model.load_weights(weights_path)
    
    return model
    
if __name__ == '__main__':
    model = C3D(weights='sports1M')
    
    vid_path = 'homerun.mp4'
    vid = skvideo.io.vread(vid_path)
    vid = vid[40:56]
    vid = preprocess_input(vid)
    
    preds = model.predict(vid)
    print(decode_predictions(preds))
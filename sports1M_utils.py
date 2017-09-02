# -*- coding: utf-8 -*-
"""Preprocessing tools for C3D input videos

"""

import numpy as np
from scipy.misc import imresize
from keras.utils.data_utils import get_file

C3D_MEAN_PATH = 'https://github.com/adamcasson/c3d/releases/download/v0.1/c3d_mean.npy'
SPORTS1M_CLASSES_PATH = 'https://github.com/adamcasson/c3d/releases/download/v0.1/sports1M_classes.txt'

def preprocess_input(video):
    """Resize and subtract mean from video input
    
    Keyword arguments:
    video -- video frames to preprocess. Expected shape 
        (frames, rows, columns, channels). If the input has more than 16 frames
        then only 16 evenly samples frames will be selected to process.
    
    Returns:
    A numpy array.
    
    """
    intervals = np.ceil(np.linspace(0, video.shape[0]-1, 16)).astype(int)
    frames = video[intervals]
    
    # Reshape to 128x171
    reshape_frames = np.zeros((frames.shape[0], 128, 171, frames.shape[3]))
    for i, img in enumerate(frames):
        img = imresize(img, (128,171), 'bicubic')
        reshape_frames[i,:,:,:] = img
        
    mean_path = get_file('c3d_mean.npy',
                         C3D_MEAN_PATH,
                         cache_subdir='models',
                         md5_hash='08a07d9761e76097985124d9e8b2fe34')
    
    # Subtract mean
    mean = np.load(mean_path)
    reshape_frames -= mean
    # Crop to 112x112
    reshape_frames = reshape_frames[:,8:120,30:142,:]
    # Add extra dimension for samples
    reshape_frames = np.expand_dims(reshape_frames, axis=0)
    
    return reshape_frames

def decode_predictions(preds):
    """Returns class label and confidence of top predicted answer
    
    Keyword arguments:
    preds -- numpy array of class probability
    
    Returns:
    A list of tuples.
    
    """
    class_pred = []
    for x in range(preds.shape[0]):
        class_pred.append(np.argmax(preds[x]))
    
    labels_path = get_file('sports1M_classes.txt',
                           SPORTS1M_CLASSES_PATH,
                           cache_subdir='models',
                           md5_hash='c102dd9508f3aa8e360494a8a0468ad9')
    
    with open(labels_path, 'r') as f:
        labels = [lines.strip() for lines in f]
        
    decoded = [(labels[x],preds[i,x]) for i,x in enumerate(class_pred)]
    
    return decoded
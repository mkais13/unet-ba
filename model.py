import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as keras
import losses
#custom loss functions



def DiceLoss(y_true, y_pred, smooth=0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def ms_ssimLoss(y_true, y_pred, **kwargs):
    tf_ms_ssim = tf.image.ssim_multiscale(y_true, y_pred, max_val=256, **kwargs)    
    return 1 - tf_ms_ssim


def IoULoss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return intersection/union

ALPHA = 0.8
GAMMA = 2

def FocalLoss(y_true, y_pred, alpha=ALPHA, gamma=GAMMA):    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    BCE = K.binary_crossentropy(y_true_f, y_pred_f)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    return focal_loss


def unet(loss, optimizer, topology_factor, kernel_init, pretrained_weights = None, input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(int(64 * topology_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(inputs)
    conv1 = Conv2D(int(64* topology_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(int(128* topology_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool1)
    conv2 = Conv2D(int(128* topology_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(int(256* topology_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool2)
    conv3 = Conv2D(int(256* topology_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(int(512* topology_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool3)
    conv4 = Conv2D(int(512* topology_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(int(1024* topology_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool4)
    conv5 = Conv2D(int(1024* topology_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(int(512*topology_factor), 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(int(512* topology_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge6)
    conv6 = Conv2D(int(512* topology_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv6)

    up7 = Conv2D(int(256* topology_factor), 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(int(256* topology_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge7)
    conv7 = Conv2D(int(256* topology_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv7)

    up8 = Conv2D(int(128* topology_factor), 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(int(128* topology_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge8)
    conv8 = Conv2D(int(128* topology_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv8)

    up9 = Conv2D(int(64* topology_factor), 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(int(64* topology_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge9)
    conv9 = Conv2D(int(64* topology_factor), 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    conv9 = Conv2D(int(topology_factor*2), 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

   
   
    if optimizer == "Adagrad" :
        optimizer_function = Adagrad(lr=1e-4)
    elif optimizer == "SGD":
        optimizer_function = SGD(lr=1e-4)
    else:
        optimizer_function = Adam(lr=1e-4)

    #checks if its a custom loss-function or one provided by keras

    if loss == "iou":
        loss_function = IoULoss
    elif loss == "dice":
        loss_function = losses.dice
    elif loss == "focal":
        loss_function = FocalLoss
    elif loss == "tversky":
        loss_function = losses.tversky
    elif loss == "focal_tversky":
        loss_function = losses.focal_tversky
    elif loss == "msssim":
        loss_function = ms_ssimLoss
    else:
        loss_function = "binary_crossentropy"


    model.compile(optimizer = optimizer_function, loss = loss_function, metrics = ['accuracy'])
    
    model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model


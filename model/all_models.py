
"""
This Network has connection from all level of decending path to the current level in the acending path!!!!
It interplates the feature maps to be able to caoncatenating them with the desired level feature map size.
It is a self feeding network!!! train quickly.

"""
from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, AlphaDropout, BatchNormalization , Activation, UpSampling2D, Lambda, add
from keras.optimizers import Adam, SGD
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint,CSVLogger
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf
import h5py
from losses.losses import getLoss


import warnings
warnings.filterwarnings("ignore")



K.set_image_data_format('channels_last')  # TF dimension ordering in this code

#img_rows = 1024
#img_cols = 1024
img_chls = 7
#input_shape = (img_rows,img_cols)

weight_decay=5e-5
smooth = 1.
channel_dim = 1 if K.image_data_format() == 'channels_first' else -1

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def weighted_dice_loss(weights,reps=1):
    def loss(y_true, y_pred):
        if reps>1:
            weights_r = K.repeat_elements(weights,reps,axis=K.ndim(weights)-1)
        else:
            weights_r = weights
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        weight_f = K.flatten(weights_r)
        intersection = K.sum(y_true_f * y_pred_f)
        return 1-(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f*weight_f) + smooth)
    return loss

def jaccard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    y_true_f_s = K.pow(y_true_f,2)
    y_pred_f_s = K.pow(y_pred_f,2)
    return (intersection + smooth) / (K.sum(y_true_f_s) + K.sum(y_pred_f_s) - intersection + smooth)
    
def jaccard_loss(y_true, y_pred):
    return 1-jaccard_coef(y_true, y_pred)
	
def weighted_jaccard_loss(weights,reps=1):
    def loss(y_true, y_pred):
        if reps>1:
            weights_r = K.repeat_elements(weights,reps,axis=K.ndim(weights)-1)
        else:
            weights_r = weights
        weight_f = K.flatten(weights_r)
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        y_true_f_s = K.pow(y_true_f,2)
        y_pred_f_s = K.pow(y_pred_f,2)
        return 1 - (intersection + smooth) / (K.sum(y_true_f_s) + K.sum(y_pred_f_s*weight_f) - intersection + smooth)
    return loss

def weighted_binary_crossentropy(y_true, y_pred,weights,reps=1):
    if reps>1:
        weights_r = K.repeat_elements(weights,reps,axis=K.ndim(weights)-1)
    else:
        weights_r = weights
    weight_f = K.flatten(weights_r)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    bce = K.binary_crossentropy(y_true_f, y_pred_f)
    return K.mean(bce*weight_f, axis=-1)
    
def complex_loss(weights,a=1.,b=1.): #'''*** dice_loss can be replaced with jaccard_loss***'''
    def loss(y_true, y_pred):
        cmplxLoss = a*jaccard_loss(y_true, y_pred) + b*weighted_binary_crossentropy(y_true, y_pred,weights,1)
        return cmplxLoss
    return loss
    
'''    
##################### DEFINING MAIN BLOCKS #######################################
'''      
def _conv_bn_relu(input, features=32, kernelSize=(3,3), strds=(1, 1), actv = 'relu', useBias=False, useRegulizer=False, dilatationRate=(1, 1), doBatchNorm =True):
    if useRegulizer:
        kernelRegularizer=l2(weight_decay)
    else:
        kernelRegularizer=None
    if actv=='selu':
        kernel_init='lecun_normal'
    else:
        kernel_init='glorot_uniform'
    convB1 = Conv2D(features, kernelSize, strides=strds,  padding='same', use_bias=useBias, kernel_regularizer=kernelRegularizer, kernel_initializer=kernel_init, dilation_rate=dilatationRate)(input)
    if actv!='selu' and doBatchNorm:
        convB1 = BatchNormalization(axis=channel_dim)(convB1)
    if actv!='None':
        convB1 = Activation(actv)(convB1)
    return convB1
    
def multiScaleConv (input_map,features, strds=(1, 1), actv = 'relu', useBias=False, useRegulizer=False, dilatationRate=(1, 1)):
    conv3 = _conv_bn_relu(input_map, features//4,(3,3), strds, actv, useBias, useRegulizer, (1,1))
    conv5 = _conv_bn_relu(input_map, features//4,(5,5), strds, actv, useBias, useRegulizer, (1,1))
    conv7 = _conv_bn_relu(input_map, features//4,(7,7), strds, actv, useBias, useRegulizer, (1,1))
    conv9 = _conv_bn_relu(input_map, features//4,(9,9), strds, actv, useBias, useRegulizer, (1,1))
    output_map = concatenate([conv3, conv5, conv7, conv9], axis=3)
    return output_map
    
def multiScaleConv_atrous (input_map,features, strds=(1, 1), actv = 'relu', useBias=False, useRegulizer=False, dilatationRate=(1, 1)):
    conv3 = _conv_bn_relu(input_map, features//4, (3,3), strds, actv, useBias, useRegulizer, (1,1))
    conv5 = _conv_bn_relu(input_map, features//4, (3,3), strds, actv, useBias, useRegulizer, (3,3))
    conv7 = _conv_bn_relu(input_map, features//4, (3,3), strds, actv, useBias, useRegulizer, (5,5))
    conv9 = _conv_bn_relu(input_map, features//4, (3,3), strds, actv, useBias, useRegulizer, (7,7))
    output_map = concatenate([conv3, conv5, conv7, conv9], axis=3)
    return output_map
    
def residual_conv(input, features=32, kernelSize=(3,3), strds=(1, 1), actv = 'relu', useBias=False, useRegulizer=False, dilatationRate=(1, 1)):
    if actv == 'selu':
        conv1 = _conv_bn_relu(input, features, kernelSize, strds, actv = 'None', useBias=useBias, useRegulizer=useRegulizer, dilatationRate=dilatationRate, doBatchNorm=False)
        conv2 = _conv_bn_relu(conv1, features, kernelSize, strds, actv = 'None', useBias=useBias, useRegulizer=useRegulizer, dilatationRate=dilatationRate, doBatchNorm=False)
    else:
        conv1 = _conv_bn_relu(input, features, kernelSize, strds, actv = 'None', useBias=useBias, useRegulizer=useRegulizer, dilatationRate=dilatationRate, doBatchNorm=True)
        conv2 = _conv_bn_relu(conv1, features, kernelSize, strds, actv = 'None', useBias=useBias, useRegulizer=useRegulizer, dilatationRate=dilatationRate, doBatchNorm=True)
    out = add([conv1, conv2])
    out = Activation(actv)(out)
    return out

def residual_conv_identity(input, features=32, kernelSize=(3,3), strds=(1, 1), actv = 'relu', useBias=False, useRegulizer=False, dilatationRate=(1, 1)):
    conv1 = _conv_bn_relu(input, features, kernelSize, strds, actv = 'None', useBias=useBias, useRegulizer=useRegulizer, dilatationRate=dilatationRate, doBatchNorm=True)
    conv2 = _conv_bn_relu(conv1, features, kernelSize, strds, actv = 'None', useBias=useBias, useRegulizer=useRegulizer, dilatationRate=dilatationRate, doBatchNorm=True)
    out = add([input, conv2])
    out = Activation(actv)(out)
    return out
    
def _deconv_bn_relu(input, features=32, kernelSize=(3,3), strds=(1, 1), actv = 'relu', useBias=False, useRegulizer=False, dilatationRate=(1, 1), doBatchNorm=True):
    if useRegulizer:
        kernelRegularizer=l2(weight_decay)
    else:
        kernelRegularizer=None
    if actv=='selu':
        kernel_init='lecun_normal'
    else:
        kernel_init='glorot_uniform'
    convB1 = Conv2DTranspose(features, kernelSize, strides=strds,  padding='same', use_bias=useBias, kernel_regularizer=kernelRegularizer, kernel_initializer=kernel_init, dilation_rate=dilatationRate)(input)
    if actv!='selu' and doBatchNorm:
        convB1 = BatchNormalization(axis=channel_dim)(convB1)
    if actv!='None':
        convB1 = Activation(actv)(convB1)
    return convB1
    
def multiScaleDeconv (input_map,features, strds=(1, 1), actv = 'relu', useBias=False, useRegulizer=False, dilatationRate=(1, 1)):
    conv3 = _deconv_bn_relu(input_map, features//4,(3,3), strds, actv, useBias, useRegulizer, (1,1))
    conv5 = _deconv_bn_relu(input_map, features//4,(5,5), strds, actv, useBias, useRegulizer, (1,1))
    conv7 = _deconv_bn_relu(input_map, features//4,(7,7), strds, actv, useBias, useRegulizer, (1,1))
    conv9 = _deconv_bn_relu(input_map, features//4,(9,9), strds, actv, useBias, useRegulizer, (1,1))
    output_map = concatenate([conv3, conv5, conv7, conv9], axis=3)
    return output_map
    
def multiScaleDeconv_atrous (input_map,features, strds=(1, 1), actv = 'relu', useBias=False, useRegulizer=False, dilatationRate=(1, 1)):
    conv3 = _deconv_bn_relu(input_map, features//4, (3,3), strds, actv, useBias, useRegulizer, (1,1))
    conv5 = _deconv_bn_relu(input_map, features//4, (3,3), strds, actv, useBias, useRegulizer, (3,3))
    conv7 = _deconv_bn_relu(input_map, features//4, (3,3), strds, actv, useBias, useRegulizer, (5,5))
    conv9 = _deconv_bn_relu(input_map, features//4, (3,3), strds, actv, useBias, useRegulizer, (7,7))
    output_map = concatenate([conv3, conv5, conv7, conv9], axis=3)
    return output_map
    
def residual_deconv(input, features=32, kernelSize=(3,3), strds=(1, 1), actv = 'relu', useBias=False, useRegulizer=False, dilatationRate=(1, 1)):
    if actv == 'selu':
        conv1 = _deconv_bn_relu(input, features, kernelSize, strds, actv = 'None', useBias=useBias, useRegulizer=useRegulizer, dilatationRate=dilatationRate, doBatchNorm=False)
        conv2 = _deconv_bn_relu(conv1, features, kernelSize, strds, actv = 'None',  useBias=useBias, useRegulizer=useRegulizer, dilatationRate=dilatationRate, doBatchNorm=False)
    else:
        conv1 = _deconv_bn_relu(input, features, kernelSize, strds, actv = 'None', useBias=useBias, useRegulizer=useRegulizer, dilatationRate=dilatationRate, doBatchNorm=True)
        conv2 = _deconv_bn_relu(conv1, features, kernelSize, strds, actv = 'None', useBias=useBias, useRegulizer=useRegulizer, dilatationRate=dilatationRate, doBatchNorm=True)
    out = add([conv1, conv2])
    out = Activation(actv)(out)
    return out

def residual_deconv_identity(input, features=32, kernelSize=(3,3), strds=(1, 1), actv = 'relu', useBias=False, useRegulizer=False, dilatationRate=(1, 1)):
    conv1 = _deconv_bn_relu(input, features, kernelSize, strds, actv = 'None', useBias=useBias, useRegulizer=useRegulizer, dilatationRate=dilatationRate, doBatchNorm=True)
    conv2 = _conv_bn_relu(conv1, features, kernelSize, strds, actv = 'None', useBias=useBias, useRegulizer=useRegulizer, dilatationRate=dilatationRate, doBatchNorm=True)
    out = add([input, conv2])
    out = Activation(actv)(out)
    return out

'''    
##################### DEFINING NETWORKS #######################################
'''    
def get_spagettiNet_singleHead_multiscale_residual_deep(input_shape,cellLoss):
# ALL DECONV IN DECODING PATH
    inputs = Input(input_shape+(img_chls,), name='main_input') # size: 1024

    conv1 = _conv_bn_relu(inputs, 64)# size: 256
    conv1 = _conv_bn_relu(conv1, 64)
    conv1 = multiScaleConv(conv1, 128)
    conv1 = _conv_bn_relu(conv1, 32)
    conv1 = concatenate([conv1, inputs], axis=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)#_conv_bn_relu(conv1, 32, (2, 2), strds=(2, 2), useRegulizer=False)#

    conv2 = residual_conv(pool1, 64, (3,3)) # size: 128
    conv2 = multiScaleConv(conv2, 256) 
    conv2 = residual_conv(conv2, 64, (3,3))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)#_conv_bn_relu(conv2, 64, (2, 2), strds=(2, 2), useRegulizer=False)#

    conv3 = residual_conv(pool2, 128, (3,3)) # size: 64
    conv3 = multiScaleConv(conv3, 512)
    conv3 = residual_conv(conv3, 128, (3,3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)#_conv_bn_relu(conv3, 128, (2, 2), strds=(2, 2), useRegulizer=False)#

    conv4 = residual_conv(pool3,256, (3,3))# size: 32
    conv4 = residual_conv_identity(conv4,256, (3,3))
    conv4 = residual_conv_identity(conv4,256, (3,3))
    conv4 = residual_conv_identity(conv4,256, (3,3))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)#_conv_bn_relu(conv4, 256, (2, 2), strds=(2, 2), useRegulizer=False)#
    
    conv5 = residual_conv(pool4, 512, (3,3))# size: 16
    conv5 = residual_conv_identity(conv5, 512, (3,3))# size: 16
    conv5 = residual_conv_identity(conv5, 512, (3,3))# size: 16
    conv5 = residual_deconv_identity(conv5, 512, (3,3))
    conv5 = residual_deconv_identity(conv5, 512, (3,3))
    conv5 = residual_deconv_identity(conv5, 512, (3,3))
    ### Add dropout here!!!###
    
    # Gland segmentation Head
    toConcat60 = _deconv_bn_relu(conv5, 256, (2, 2), strds=(2, 2))
    toConcat61 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//8,input_shape[1]//8),method=tf.image.ResizeMethod.BILINEAR))(conv1)
    toConcat62 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//8,input_shape[1]//8),method=tf.image.ResizeMethod.BILINEAR))(conv2)
    toConcat63 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//8,input_shape[1]//8),method=tf.image.ResizeMethod.BILINEAR))(conv3)
    up6 = concatenate([toConcat60, toConcat61, toConcat62, toConcat63, conv4], axis=3)
    conv6 = residual_deconv(up6, 256, (3,3))
    conv6 = residual_deconv_identity(conv6, 256, (3,3))
    conv6 = residual_deconv_identity(conv6, 256, (3,3))
    conv6 = residual_deconv_identity(conv6, 256, (3,3))
    
    toConcat70 = _deconv_bn_relu(conv6, 128, (2, 2), strds=(2, 2))
    toConcat71 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//4,input_shape[1]//4),method=tf.image.ResizeMethod.BILINEAR))(conv1)
    toConcat72 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//4,input_shape[1]//4),method=tf.image.ResizeMethod.BILINEAR))(conv2)
    toConcat73 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//4,input_shape[1]//4),method=tf.image.ResizeMethod.BILINEAR))(conv4)
    up7 = concatenate([toConcat70, toConcat71, toConcat72, toConcat73, conv3], axis=3)
    conv7 = residual_deconv(up7, 128, (3,3))
    conv7 = multiScaleDeconv(conv7, 512)
    conv7 = residual_deconv(conv7, 128, (3,3))

    toConcat80 = _deconv_bn_relu(conv7, 64, (2, 2), strds=(2, 2))
    toConcat81 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//2,input_shape[1]//2),method=tf.image.ResizeMethod.BILINEAR))(conv1)
    toConcat82 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//2,input_shape[1]//2),method=tf.image.ResizeMethod.BILINEAR))(conv3)
    toConcat83 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//2,input_shape[1]//2),method=tf.image.ResizeMethod.BILINEAR))(conv4)
    up8 = concatenate([toConcat80, toConcat81, toConcat82, toConcat83, conv2], axis=3)
    conv8 = residual_deconv(up8, 64, (3,3))
    conv8 = multiScaleDeconv(conv8, 256)
    conv8 = residual_deconv(conv8, 64, (3,3))

    toConcat90 = _deconv_bn_relu(conv8, 32, (2, 2), strds=(2, 2))
    toConcat91 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0],input_shape[1]),method=tf.image.ResizeMethod.BILINEAR))(conv2)
    toConcat92 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0],input_shape[1]),method=tf.image.ResizeMethod.BILINEAR))(conv3)
    toConcat93 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0],input_shape[1]),method=tf.image.ResizeMethod.BILINEAR))(conv4)
    up9 = concatenate([toConcat90, toConcat91, toConcat92, toConcat93, conv1], axis=3)
    conv9 = _deconv_bn_relu(up9, 32, (3,3))
    conv9 = multiScaleDeconv(conv9, 128)
    conv9 = _deconv_bn_relu(conv9, 64, (3,3))
    conv9 = _deconv_bn_relu(conv9, 64, (3,3))
    
    cell_output = Conv2D(1, (1, 1), activation='sigmoid', name='cell_output')(conv9)

    model = Model(inputs=inputs, outputs=cell_output)
   # model.compile(optimizer=Adam(lr=3e-3, decay=1e-4, amsgrad=True),
    #          loss=cellLoss, metrics=['mean_absolute_error'])

    return model

def get_spagettiNet_singleHead_multiscale_residual(input_shape,cellLoss): # NAVID>>>> USE THIS
# ALL DECONV IN DECODING PATH
    inputs = Input(input_shape+(img_chls,), name='main_input') # size: 1024

    conv1 = _conv_bn_relu(inputs, 64)# size: 256
    conv1 = _conv_bn_relu(conv1, 64)
    conv1 = multiScaleConv(conv1, 128)
    conv1 = _conv_bn_relu(conv1, 32)
    conv1 = concatenate([conv1, inputs], axis=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)#_conv_bn_relu(conv1, 32, (2, 2), strds=(2, 2), useRegulizer=False)#

    conv2 = residual_conv(pool1, 64, (3,3)) # size: 128
    conv2 = multiScaleConv(conv2, 256) 
    conv2 = residual_conv(conv2, 64, (3,3))
    conv2 = residual_conv(conv2, 64, (3,3))
    conv2 = residual_conv(conv2, 64, (3,3))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)#_conv_bn_relu(conv2, 64, (2, 2), strds=(2, 2), useRegulizer=False)#

    conv3 = residual_conv(pool2, 128, (3,3)) # size: 64
    conv3 = multiScaleConv(conv3, 512)
    conv3 = residual_conv(conv3, 128, (3,3))
    conv3 = residual_conv(conv3, 128, (3,3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)#_conv_bn_relu(conv3, 128, (2, 2), strds=(2, 2), useRegulizer=False)#

    conv4 = residual_conv(pool3,256, (3,3))# size: 32
    conv4 = residual_conv(conv4,256, (3,3))
    conv4 = residual_conv(conv4,256, (3,3))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)#_conv_bn_relu(conv4, 256, (2, 2), strds=(2, 2), useRegulizer=False)#
    
    conv5 = residual_conv(pool4, 512, (3,3))# size: 16
    conv5 = residual_deconv(conv5, 512, (3,3))
    ### Add dropout here!!!###
    
    # Gland segmentation Head
    toConcat60 = _deconv_bn_relu(conv5, 256, (2, 2), strds=(2, 2))
    toConcat61 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//8,input_shape[1]//8),method=tf.image.ResizeMethod.BILINEAR))(conv1)
    toConcat62 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//8,input_shape[1]//8),method=tf.image.ResizeMethod.BILINEAR))(conv2)
    toConcat63 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//8,input_shape[1]//8),method=tf.image.ResizeMethod.BILINEAR))(conv3)
    up6 = concatenate([toConcat60, toConcat61, toConcat62, toConcat63, conv4], axis=3)
    conv6 = residual_deconv(up6, 256, (3,3))
    conv6 = residual_deconv(conv6, 256, (3,3))
    conv6 = residual_deconv(conv6, 256, (3,3))
    
    toConcat70 = _deconv_bn_relu(conv6, 128, (2, 2), strds=(2, 2))
    toConcat71 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//4,input_shape[1]//4),method=tf.image.ResizeMethod.BILINEAR))(conv1)
    toConcat72 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//4,input_shape[1]//4),method=tf.image.ResizeMethod.BILINEAR))(conv2)
    toConcat73 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//4,input_shape[1]//4),method=tf.image.ResizeMethod.BILINEAR))(conv4)
    up7 = concatenate([toConcat70, toConcat71, toConcat72, toConcat73, conv3], axis=3)
    conv7 = residual_deconv(up7, 128, (3,3))
    conv7 = multiScaleDeconv(conv7, 512)
    conv7 = residual_deconv(conv7, 128, (3,3))
    conv7 = residual_deconv(conv7, 128, (3,3))

    toConcat80 = _deconv_bn_relu(conv7, 64, (2, 2), strds=(2, 2))
    toConcat81 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//2,input_shape[1]//2),method=tf.image.ResizeMethod.BILINEAR))(conv1)
    toConcat82 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//2,input_shape[1]//2),method=tf.image.ResizeMethod.BILINEAR))(conv3)
    toConcat83 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//2,input_shape[1]//2),method=tf.image.ResizeMethod.BILINEAR))(conv4)
    up8 = concatenate([toConcat80, toConcat81, toConcat82, toConcat83, conv2], axis=3)
    conv8 = residual_deconv(up8, 64, (3,3))
    conv8 = multiScaleDeconv(conv8, 256)
    conv8 = residual_deconv(conv8, 64, (3,3))
    conv8 = residual_deconv(conv8, 64, (3,3))
    conv8 = residual_deconv(conv8, 64, (3,3))

    toConcat90 = _deconv_bn_relu(conv8, 32, (2, 2), strds=(2, 2))
    toConcat91 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0],input_shape[1]),method=tf.image.ResizeMethod.BILINEAR))(conv2)
    toConcat92 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0],input_shape[1]),method=tf.image.ResizeMethod.BILINEAR))(conv3)
    toConcat93 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0],input_shape[1]),method=tf.image.ResizeMethod.BILINEAR))(conv4)
    up9 = concatenate([toConcat90, toConcat91, toConcat92, toConcat93, conv1], axis=3)
    conv9 = _deconv_bn_relu(up9, 64, (3,3))
    conv9 = multiScaleDeconv(conv9, 128)
    conv9 = _deconv_bn_relu(conv9, 64, (3,3))
    conv9 = _deconv_bn_relu(conv9, 64, (3,3))
    
    cell_output = Conv2D(1, (1, 1), activation='sigmoid', name='cell_output')(conv9)

    model = Model(inputs=inputs, outputs=cell_output)
    model.compile(optimizer=Adam(lr=3e-3, decay=1e-4, amsgrad=True),
              loss=getLoss(cellLoss), metrics=[dice_coef])

    return model


def get_spagettiNet_twoHeaded_multiscale_residual(input_shape,cellLoss, marginLoss): # NAVID>>>> USE THIS
# ALL DECONV IN DECODING PATH
    inputs = Input(input_shape+(img_chls,), name='main_input') # size: 1024
    weights = Input(input_shape+(1,), name='weights')

    conv1 = _conv_bn_relu(inputs, 64)# size: 256
    conv1 = _conv_bn_relu(conv1, 64)
    conv1 = multiScaleConv(conv1, 128)
    conv1 = _conv_bn_relu(conv1, 32)
    conv1 = concatenate([conv1, inputs], axis=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)#_conv_bn_relu(conv1, 32, (2, 2), strds=(2, 2), useRegulizer=False)#

    conv2 = residual_conv(pool1, 64, (3,3)) # size: 128
    conv2 = multiScaleConv(conv2, 256) 
    conv2 = residual_conv(conv2, 64, (3,3))
    conv2 = residual_conv(conv2, 64, (3,3))
    conv2 = residual_conv(conv2, 64, (3,3))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)#_conv_bn_relu(conv2, 64, (2, 2), strds=(2, 2), useRegulizer=False)#

    conv3 = residual_conv(pool2, 128, (3,3)) # size: 64
    conv3 = multiScaleConv(conv3, 512)
    conv3 = residual_conv(conv3, 128, (3,3))
    conv3 = residual_conv(conv3, 128, (3,3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)#_conv_bn_relu(conv3, 128, (2, 2), strds=(2, 2), useRegulizer=False)#

    conv4 = residual_conv(pool3,256, (3,3))# size: 32
    conv4 = residual_conv(conv4,256, (3,3))
    conv4 = residual_conv(conv4,256, (3,3))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)#_conv_bn_relu(conv4, 256, (2, 2), strds=(2, 2), useRegulizer=False)#
    
    conv5 = residual_conv(pool4, 512, (3,3))# size: 16
    conv5 = residual_deconv(conv5, 512, (3,3))
    ### Add dropout here!!!###
    
    # Gland segmentation Head
    toConcat60 = _deconv_bn_relu(conv5, 256, (2, 2), strds=(2, 2))
    toConcat61 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//8,input_shape[1]//8),method=tf.image.ResizeMethod.BILINEAR))(conv1)
    toConcat62 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//8,input_shape[1]//8),method=tf.image.ResizeMethod.BILINEAR))(conv2)
    toConcat63 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//8,input_shape[1]//8),method=tf.image.ResizeMethod.BILINEAR))(conv3)
    up6 = concatenate([toConcat60, toConcat61, toConcat62, toConcat63, conv4], axis=3)
    conv6 = residual_deconv(up6, 256, (3,3))
    conv6 = residual_deconv(conv6, 256, (3,3))
    conv6 = residual_deconv(conv6, 256, (3,3))
    
    toConcat70 = _deconv_bn_relu(conv6, 128, (2, 2), strds=(2, 2))
    toConcat71 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//4,input_shape[1]//4),method=tf.image.ResizeMethod.BILINEAR))(conv1)
    toConcat72 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//4,input_shape[1]//4),method=tf.image.ResizeMethod.BILINEAR))(conv2)
    toConcat73 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//4,input_shape[1]//4),method=tf.image.ResizeMethod.BILINEAR))(conv4)
    up7 = concatenate([toConcat70, toConcat71, toConcat72, toConcat73, conv3], axis=3)
    conv7 = residual_deconv(up7, 128, (3,3))
    conv7 = multiScaleDeconv(conv7, 512)
    conv7 = residual_deconv(conv7, 128, (3,3))
    conv7 = residual_deconv(conv7, 128, (3,3))

    toConcat80 = _deconv_bn_relu(conv7, 64, (2, 2), strds=(2, 2))
    toConcat81 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//2,input_shape[1]//2),method=tf.image.ResizeMethod.BILINEAR))(conv1)
    toConcat82 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//2,input_shape[1]//2),method=tf.image.ResizeMethod.BILINEAR))(conv3)
    toConcat83 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//2,input_shape[1]//2),method=tf.image.ResizeMethod.BILINEAR))(conv4)
    up8 = concatenate([toConcat80, toConcat81, toConcat82, toConcat83, conv2], axis=3)
    conv8 = residual_deconv(up8, 64, (3,3))
    conv8 = multiScaleDeconv(conv8, 256)
    conv8 = residual_deconv(conv8, 64, (3,3))
    conv8 = residual_deconv(conv8, 64, (3,3))
    conv8 = residual_deconv(conv8, 64, (3,3))

    toConcat90 = _deconv_bn_relu(conv8, 32, (2, 2), strds=(2, 2))
    toConcat91 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0],input_shape[1]),method=tf.image.ResizeMethod.BILINEAR))(conv2)
    toConcat92 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0],input_shape[1]),method=tf.image.ResizeMethod.BILINEAR))(conv3)
    toConcat93 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0],input_shape[1]),method=tf.image.ResizeMethod.BILINEAR))(conv4)
    up9 = concatenate([toConcat90, toConcat91, toConcat92, toConcat93, conv1], axis=3)
    conv9 = _deconv_bn_relu(up9, 64, (3,3))
    conv9 = multiScaleDeconv(conv9, 128)
    conv9 = _deconv_bn_relu(conv9, 64, (3,3))
    conv9 = _deconv_bn_relu(conv9, 64, (3,3))
    
    cell_output = Conv2D(1, (1, 1), activation='sigmoid', name='cell_output')(conv9)
    
    margin_output = Conv2D(1, (1, 1), activation='sigmoid', name='margin_output')(conv9)

    model = Model(inputs=[inputs, weights], outputs=[cell_output, margin_output])
    model.compile(optimizer=Adam(lr=3e-3, decay=1e-4, amsgrad=True),
              loss={'cell_output': getLoss(cellLoss, weightMap=weights), 'margin_output': getLoss(marginLoss,weightMap=weights)},
              loss_weights={'cell_output': 1., 'margin_output': 1.}, metrics=[dice_coef])

    return model

def get_spagettiNet_twoHeaded_multiscale_residual_deep(input_shape,cellLoss, marginLoss):
# ALL DECONV IN DECODING PATH
    inputs = Input(input_shape+(img_chls,), name='main_input') # size: 1024
    weights = Input(input_shape+(1,), name='weights')

    conv1 = _conv_bn_relu(inputs, 64)# size: 256
    conv1 = _conv_bn_relu(conv1, 64)
    conv1 = multiScaleConv(conv1, 128)
    conv1 = _conv_bn_relu(conv1, 32)
    conv1 = concatenate([conv1, inputs], axis=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)#_conv_bn_relu(conv1, 32, (2, 2), strds=(2, 2), useRegulizer=False)#

    conv2 = residual_conv(pool1, 64, (3,3)) # size: 128
    conv2 = multiScaleConv(conv2, 256) 
    conv2 = residual_conv(conv2, 64, (3,3))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)#_conv_bn_relu(conv2, 64, (2, 2), strds=(2, 2), useRegulizer=False)#

    conv3 = residual_conv(pool2, 128, (3,3)) # size: 64
    conv3 = multiScaleConv(conv3, 512)
    conv3 = residual_conv(conv3, 128, (3,3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)#_conv_bn_relu(conv3, 128, (2, 2), strds=(2, 2), useRegulizer=False)#

    conv4 = residual_conv(pool3,256, (3,3))# size: 32
    conv4 = residual_conv_identity(conv4,256, (3,3))
    conv4 = residual_conv_identity(conv4,256, (3,3))
    conv4 = residual_conv_identity(conv4,256, (3,3))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)#_conv_bn_relu(conv4, 256, (2, 2), strds=(2, 2), useRegulizer=False)#
    
    conv5 = residual_conv(pool4, 512, (3,3))# size: 16
    conv5 = residual_conv_identity(conv5, 512, (3,3))# size: 16
    conv5 = residual_conv_identity(conv5, 512, (3,3))# size: 16
    conv5 = residual_deconv_identity(conv5, 512, (3,3))
    conv5 = residual_deconv_identity(conv5, 512, (3,3))
    conv5 = residual_deconv_identity(conv5, 512, (3,3))
    ### Add dropout here!!!###
    
    # Gland segmentation Head
    toConcat60 = _deconv_bn_relu(conv5, 256, (2, 2), strds=(2, 2))
    toConcat61 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//8,input_shape[1]//8),method=tf.image.ResizeMethod.BILINEAR))(conv1)
    toConcat62 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//8,input_shape[1]//8),method=tf.image.ResizeMethod.BILINEAR))(conv2)
    toConcat63 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//8,input_shape[1]//8),method=tf.image.ResizeMethod.BILINEAR))(conv3)
    up6 = concatenate([toConcat60, toConcat61, toConcat62, toConcat63, conv4], axis=3)
    conv6 = residual_deconv(up6, 256, (3,3))
    conv6 = residual_deconv_identity(conv6, 256, (3,3))
    conv6 = residual_deconv_identity(conv6, 256, (3,3))
    conv6 = residual_deconv_identity(conv6, 256, (3,3))
    
    toConcat70 = _deconv_bn_relu(conv6, 128, (2, 2), strds=(2, 2))
    toConcat71 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//4,input_shape[1]//4),method=tf.image.ResizeMethod.BILINEAR))(conv1)
    toConcat72 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//4,input_shape[1]//4),method=tf.image.ResizeMethod.BILINEAR))(conv2)
    toConcat73 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//4,input_shape[1]//4),method=tf.image.ResizeMethod.BILINEAR))(conv4)
    up7 = concatenate([toConcat70, toConcat71, toConcat72, toConcat73, conv3], axis=3)
    conv7 = residual_deconv(up7, 128, (3,3))
    conv7 = multiScaleDeconv(conv7, 512)
    conv7 = residual_deconv(conv7, 128, (3,3))

    toConcat80 = _deconv_bn_relu(conv7, 64, (2, 2), strds=(2, 2))
    toConcat81 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//2,input_shape[1]//2),method=tf.image.ResizeMethod.BILINEAR))(conv1)
    toConcat82 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//2,input_shape[1]//2),method=tf.image.ResizeMethod.BILINEAR))(conv3)
    toConcat83 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//2,input_shape[1]//2),method=tf.image.ResizeMethod.BILINEAR))(conv4)
    up8 = concatenate([toConcat80, toConcat81, toConcat82, toConcat83, conv2], axis=3)
    conv8 = residual_deconv(up8, 64, (3,3))
    conv8 = multiScaleDeconv(conv8, 256)
    conv8 = residual_deconv(conv8, 64, (3,3))

    toConcat90 = _deconv_bn_relu(conv8, 32, (2, 2), strds=(2, 2))
    toConcat91 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0],input_shape[1]),method=tf.image.ResizeMethod.BILINEAR))(conv2)
    toConcat92 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0],input_shape[1]),method=tf.image.ResizeMethod.BILINEAR))(conv3)
    toConcat93 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0],input_shape[1]),method=tf.image.ResizeMethod.BILINEAR))(conv4)
    up9 = concatenate([toConcat90, toConcat91, toConcat92, toConcat93, conv1], axis=3)
    conv9 = _deconv_bn_relu(up9, 32, (3,3))
    conv9 = multiScaleDeconv(conv9, 128)
    conv9 = _deconv_bn_relu(conv9, 64, (3,3))
    conv9 = _deconv_bn_relu(conv9, 64, (3,3))
    
    cell_output = Conv2D(1, (1, 1), activation='sigmoid', name='cell_output')(conv9)
    
    margin_output = Conv2D(1, (1, 1), activation='sigmoid', name='margin_output')(conv9)

    model = Model(inputs=[inputs, weights], outputs=[cell_output, margin_output])
    model.compile(optimizer=Adam(lr=3e-3, decay=1e-4, amsgrad=True),
              loss={'cell_output': getLoss(cellLoss, weightMap=weights), 'margin_output': getLoss(marginLoss,weightMap=weights)},
              loss_weights={'cell_output': 1., 'margin_output': 1.}, metrics=[dice_coef])

    return model


def get_spagettiNet_twoHeaded_multiscale_deep(input_shape,cellLoss, marginLoss): # NAVID>>>> USE THIS
# ALL DECONV IN DECODING PATH
    inputs = Input(input_shape+(img_chls,), name='main_input') # size: 1024
    weights = Input(input_shape+(1,), name='weights')

    conv1 = _conv_bn_relu(inputs, 64)# size: 256
    conv1 = _conv_bn_relu(conv1, 64)
    conv1 = multiScaleConv(conv1, 128)
    conv1 = _conv_bn_relu(conv1, 32)
    conv1 = concatenate([conv1, inputs], axis=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)#_conv_bn_relu(conv1, 32, (2, 2), strds=(2, 2), useRegulizer=False)#

    conv2 = _conv_bn_relu(pool1, 64, (3,3)) # size: 128
    conv2 = multiScaleConv(conv2, 256) 
    conv2 = _conv_bn_relu(conv2, 64, (3,3))
    conv2 = _conv_bn_relu(conv2, 64, (3,3))
    conv2 = _conv_bn_relu(conv2, 64, (3,3))
    conv2 = _conv_bn_relu(conv2, 64, (3,3))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)#_conv_bn_relu(conv2, 64, (2, 2), strds=(2, 2), useRegulizer=False)#

    conv3 = _conv_bn_relu(pool2, 128, (3,3)) # size: 64
    conv3 = multiScaleConv(conv3, 512)
    conv3 = _conv_bn_relu(conv3, 128, (3,3))
    conv3 = _conv_bn_relu(conv3, 128, (3,3))
    conv3 = _conv_bn_relu(conv3, 128, (3,3))
    conv3 = _conv_bn_relu(conv3, 128, (3,3))
    conv3 = _conv_bn_relu(conv3, 128, (3,3))
    conv3 = _conv_bn_relu(conv3, 128, (3,3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)#_conv_bn_relu(conv3, 128, (2, 2), strds=(2, 2), useRegulizer=False)#

    conv4 = _conv_bn_relu(pool3,256, (3,3))# size: 32
    conv4 = _conv_bn_relu(conv4,256, (3,3))
    conv4 = _conv_bn_relu(conv4,256, (3,3))
    conv4 = _conv_bn_relu(conv4,256, (3,3))
    conv4 = _conv_bn_relu(conv4,256, (3,3))
    conv4 = _conv_bn_relu(conv4,256, (3,3))
    conv4 = _conv_bn_relu(conv4,256, (3,3))
    conv4 = _conv_bn_relu(conv4,256, (3,3))
    conv4 = _conv_bn_relu(conv4,256, (3,3))
    conv4 = _conv_bn_relu(conv4,256, (3,3))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)#_conv_bn_relu(conv4, 256, (2, 2), strds=(2, 2), useRegulizer=False)#
    
    conv5 = _conv_bn_relu(pool4, 512, (3,3))# size: 16
    conv5 = _conv_bn_relu(conv5, 512, (3,3))# size: 16
    conv5 = _conv_bn_relu(conv5, 512, (3,3))# size: 16
    conv5 = _conv_bn_relu(conv5, 512, (3,3))# size: 16
    conv5 = _deconv_bn_relu(conv5, 512, (3,3))# size: 16
    conv5 = _deconv_bn_relu(conv5, 512, (3,3))
    conv5 = _deconv_bn_relu(conv5, 512, (3,3))
    conv5 = _deconv_bn_relu(conv5, 512, (3,3))
    ### Add dropout here!!!###
    
    # Gland segmentation Head
    toConcat60 = _deconv_bn_relu(conv5, 256, (2, 2), strds=(2, 2))
    toConcat61 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//8,input_shape[1]//8),method=tf.image.ResizeMethod.BILINEAR))(conv1)
    toConcat62 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//8,input_shape[1]//8),method=tf.image.ResizeMethod.BILINEAR))(conv2)
    toConcat63 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//8,input_shape[1]//8),method=tf.image.ResizeMethod.BILINEAR))(conv3)
    up6 = concatenate([toConcat60, toConcat61, toConcat62, toConcat63, conv4], axis=3)
    conv6 = _deconv_bn_relu(up6, 256, (3,3))
    conv6 = _deconv_bn_relu(conv6, 256, (3,3))
    conv6 = _deconv_bn_relu(conv6, 256, (3,3))
    conv6 = _deconv_bn_relu(conv6, 256, (3,3))
    conv6 = _deconv_bn_relu(conv6, 256, (3,3))
    conv6 = _deconv_bn_relu(conv6, 256, (3,3))
    conv6 = _deconv_bn_relu(conv6, 256, (3,3))
    conv6 = _deconv_bn_relu(conv6, 256, (3,3))
    conv6 = _deconv_bn_relu(conv6, 256, (3,3))
    conv6 = _deconv_bn_relu(conv6, 256, (3,3))
    
    toConcat70 = _deconv_bn_relu(conv6, 128, (2, 2), strds=(2, 2))
    toConcat71 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//4,input_shape[1]//4),method=tf.image.ResizeMethod.BILINEAR))(conv1)
    toConcat72 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//4,input_shape[1]//4),method=tf.image.ResizeMethod.BILINEAR))(conv2)
    toConcat73 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//4,input_shape[1]//4),method=tf.image.ResizeMethod.BILINEAR))(conv4)
    up7 = concatenate([toConcat70, toConcat71, toConcat72, toConcat73, conv3], axis=3)
    conv7 = _deconv_bn_relu(up7, 128, (3,3))
    conv7 = multiScaleDeconv(conv7, 512)
    conv7 = _deconv_bn_relu(conv7, 128, (3,3))
    conv7 = _deconv_bn_relu(conv7, 128, (3,3))
    conv7 = _deconv_bn_relu(conv7, 128, (3,3))
    conv7 = _deconv_bn_relu(conv7, 128, (3,3))
    conv7 = _deconv_bn_relu(conv7, 128, (3,3))
    conv7 = _deconv_bn_relu(conv7, 128, (3,3))

    toConcat80 = _deconv_bn_relu(conv7, 64, (2, 2), strds=(2, 2))
    toConcat81 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//2,input_shape[1]//2),method=tf.image.ResizeMethod.BILINEAR))(conv1)
    toConcat82 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//2,input_shape[1]//2),method=tf.image.ResizeMethod.BILINEAR))(conv3)
    toConcat83 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//2,input_shape[1]//2),method=tf.image.ResizeMethod.BILINEAR))(conv4)
    up8 = concatenate([toConcat80, toConcat81, toConcat82, toConcat83, conv2], axis=3)
    conv8 = _deconv_bn_relu(up8, 64, (3,3))
    conv8 = multiScaleDeconv(conv8, 256)
    conv8 = _deconv_bn_relu(conv8, 64, (3,3))
    conv8 = _deconv_bn_relu(conv8, 64, (3,3))
    conv8 = _deconv_bn_relu(conv8, 64, (3,3))
    conv8 = _deconv_bn_relu(conv8, 64, (3,3))

    toConcat90 = _deconv_bn_relu(conv8, 32, (2, 2), strds=(2, 2))
    toConcat91 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0],input_shape[1]),method=tf.image.ResizeMethod.BILINEAR))(conv2)
    toConcat92 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0],input_shape[1]),method=tf.image.ResizeMethod.BILINEAR))(conv3)
    toConcat93 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0],input_shape[1]),method=tf.image.ResizeMethod.BILINEAR))(conv4)
    up9 = concatenate([toConcat90, toConcat91, toConcat92, toConcat93, conv1], axis=3)
    conv9 = _deconv_bn_relu(up9, 32, (3,3))
    conv9 = multiScaleDeconv(conv9, 128)
    conv9 = _deconv_bn_relu(conv9, 64, (3,3))
    conv9 = _deconv_bn_relu(conv9, 64, (3,3))
    
    cell_output = Conv2D(1, (1, 1), activation='sigmoid', name='cell_output')(conv9)
    
    margin_output = Conv2D(1, (1, 1), activation='sigmoid', name='margin_output')(conv9)

    model = Model(inputs=[inputs, weights], outputs=[cell_output, margin_output])
    model.compile(optimizer=Adam(lr=3e-3, decay=1e-4, amsgrad=True),
              loss={'cell_output': getLoss(cellLoss, weightMap=weights), 'margin_output': getLoss(marginLoss,weightMap=weights)},
              loss_weights={'cell_output': 1., 'margin_output': 1.}, metrics=[dice_coef])

    return model

def get_UNET_twoHead_multiscale_residual(input_shape,cellLoss, marginLoss):
# ALL DECONV IN DECODING PATH
    inputs = Input(input_shape+(img_chls,), name='main_input') # size: 1024
    weights = Input(input_shape+(1,), name='weights')

    conv1 = _conv_bn_relu(inputs, 64)# size: 256
    conv1 = _conv_bn_relu(conv1, 64)
    conv1 = _conv_bn_relu(conv1, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)#_conv_bn_relu(conv1, 32, (2, 2), strds=(2, 2), useRegulizer=False)#

    conv2 = residual_conv(pool1, 64, (3,3))
    conv2 = multiScaleConv(conv2, 256) # size: 128
    conv2 = residual_conv(conv2, 64, (3,3))
    conv2 = residual_conv(conv2, 64, (3,3))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)#_conv_bn_relu(conv2, 64, (2, 2), strds=(2, 2), useRegulizer=False)#

    conv3 = residual_conv(pool2, 128, (3,3))
    conv3 = multiScaleConv(conv3, 256)# size: 64
    conv3 = residual_conv(conv3, 128, (3,3))
    conv3 = residual_conv(conv3, 128, (3,3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)#_conv_bn_relu(conv3, 128, (2, 2), strds=(2, 2), useRegulizer=False)#

    conv4 = residual_conv(pool3,256, (3,3))# size: 32
    conv4 = residual_conv(conv4,256, (3,3))# size: 32
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)#_conv_bn_relu(conv4, 256, (2, 2), strds=(2, 2), useRegulizer=False)#
    
    conv5 = residual_conv(pool4, 512, (3,3))# size: 16
    conv5 = residual_deconv(conv5, 512, (3,3))# size: 24
    
    toConcat60 = _deconv_bn_relu(conv5, 256, (2, 2), strds=(2, 2))
    up6 = concatenate([toConcat60, conv4], axis=3)
    conv6 = residual_deconv(up6, 256, (3,3))
    conv6 = residual_deconv(conv6, 256, (3,3))
    
    toConcat70 = _deconv_bn_relu(conv6, 128, (2, 2), strds=(2, 2))
    up7 = concatenate([toConcat70, conv3], axis=3)
    conv7 = residual_deconv(up7, 128, (3,3))
    conv7 = multiScaleDeconv(conv7, 256)
    conv7 = residual_deconv(conv7, 128, (3,3))
    conv7 = residual_deconv(conv7, 128, (3,3))

    toConcat80 = _deconv_bn_relu(conv7, 64, (2, 2), strds=(2, 2))
    up8 = concatenate([toConcat80, conv2], axis=3)
    conv8 = residual_deconv(up8, 64, (3,3))
    conv8 = multiScaleDeconv(conv8, 256)
    conv8 = residual_deconv(conv8, 64, (3,3))
    conv8 = residual_deconv(conv8, 64, (3,3))
    
	# First Head: Cell Marker
    toConcat90 = _deconv_bn_relu(conv8, 32, (2, 2), strds=(2, 2))
    up9 = concatenate([toConcat90, conv1], axis=3)
    conv9 = _deconv_bn_relu(up9, 64, (3,3))
    conv9 = _deconv_bn_relu(conv9, 64, (3,3))
    conv9 = _deconv_bn_relu(conv9, 32, (3,3))
    conv9 = _deconv_bn_relu(conv9, 32, (3,3))
    
    cell_output = Conv2D(1, (1, 1), activation='sigmoid', name='cell_output')(conv9)
	
    margin_output = Conv2D(1, (1, 1), activation='sigmoid', name='margin_output')(conv9)

    model = Model(inputs=[inputs, weights], outputs=[cell_output, margin_output])
    
    '''* Different compile options: Check to see which one is good!?!you can use a trustable learning rate like = 5e-3 for your experiments*'''
#    model.compile(optimizer=Adam(lr=2e-2),
#              loss={'cell_output': complex_loss(weights,a=100.,b=.002), 'margin_output': jaccard_loss}, #weighted_jaccard_loss and jaccard_loss could be used!
#              loss_weights={'cell_output': 1., 'margin_output': 500.},metrics=[dice_coef])

    model.compile(optimizer=Adam(lr=3e-3, decay=1e-4, amsgrad=True),
              loss={'cell_output': getLoss(cellLoss, weightMap=weights), 'margin_output': getLoss(marginLoss,weightMap=weights)},
              loss_weights={'cell_output': 1., 'margin_output': 1.}, metrics=[dice_coef])

    return model

def get_UNET_twoHead_multiscale_residual_shallow(input_shape,cellLoss, marginLoss):
# ALL DECONV IN DECODING PATH
    inputs = Input(input_shape+(img_chls,), name='main_input') # size: 1024
    weights = Input(input_shape+(1,), name='weights')

    conv1 = _conv_bn_relu(inputs, 64)# size: 256
    conv1 = _conv_bn_relu(conv1, 64)
    conv1 = _conv_bn_relu(conv1, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)#_conv_bn_relu(conv1, 32, (2, 2), strds=(2, 2), useRegulizer=False)#

    conv2 = residual_conv(pool1, 64, (3,3))
    conv2 = residual_conv(conv2, 64, (3,3))
    conv2 = multiScaleConv(conv2, 256) # size: 128
    conv2 = residual_conv(conv2, 128, (3,3))
    conv2 = residual_conv(conv2, 128, (3,3))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)#_conv_bn_relu(conv2, 64, (2, 2), strds=(2, 2), useRegulizer=False)#

    conv3 = residual_conv(pool2, 128, (3,3))
    conv3 = multiScaleConv(conv3, 512)# size: 64
    conv3 = residual_conv(conv3, 512, (3,3))
    conv3 = residual_conv(conv3, 512, (3,3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)#_conv_bn_relu(conv3, 128, (2, 2), strds=(2, 2), useRegulizer=False)#

    conv4 = residual_conv(pool3,1024, (3,3))# size: 32
    conv4 = residual_deconv(conv4,512, (3,3))# size: 32
    
    toConcat70 = _deconv_bn_relu(conv4, 512, (2, 2), strds=(2, 2))
    up7 = concatenate([toConcat70, conv3], axis=3)
    conv7 = residual_deconv(up7, 256, (3,3))
    conv7 = multiScaleDeconv(conv7, 512)
    conv7 = residual_deconv(conv7, 256, (3,3))
    conv7 = residual_deconv(conv7, 256, (3,3))

    toConcat80 = _deconv_bn_relu(conv7, 256, (2, 2), strds=(2, 2))
    up8 = concatenate([toConcat80, conv2], axis=3)
    conv8 = residual_deconv(up8, 128, (3,3))
    conv8 = residual_deconv(conv8, 128, (3,3))
    conv8 = multiScaleDeconv(conv8, 256)
    conv8 = residual_deconv(conv8, 128, (3,3))
    conv8 = residual_deconv(conv8, 128, (3,3))
    
	# First Head: Cell Marker
    toConcat90 = _deconv_bn_relu(conv8, 128, (2, 2), strds=(2, 2))
    up9 = concatenate([toConcat90, conv1], axis=3)
    conv9 = _deconv_bn_relu(up9, 128, (3,3))
    conv9 = _deconv_bn_relu(conv9, 128, (3,3))
    conv9 = _deconv_bn_relu(conv9, 64, (3,3))
    conv9 = _deconv_bn_relu(conv9, 64, (3,3))
    conv9 = _deconv_bn_relu(conv9, 32, (3,3))
    
    cell_output = Conv2D(1, (1, 1), activation='sigmoid', name='cell_output')(conv9)
	
	# Second Head: Cell Margin
#    conv10 = _deconv_bn_relu(up9, 128, (3,3))
#    conv10 = _deconv_bn_relu(conv10, 64, (3,3))
#    conv10 = _deconv_bn_relu(conv10, 32, (3,3))
#    conv10 = _deconv_bn_relu(conv10, 16, (3,3))
    
    margin_output = Conv2D(1, (1, 1), activation='sigmoid', name='margin_output')(conv9)

    model = Model(inputs=[inputs, weights], outputs=[cell_output, margin_output])
    
    '''* Different compile options: Check to see which one is good!?!you can use a trustable learning rate like = 5e-3 for your experiments*'''  
    model.compile(optimizer=Adam(lr=3e-3, decay=1e-4, amsgrad=True),
              loss={'cell_output': getLoss(cellLoss, weightMap=weights), 'margin_output': getLoss(marginLoss,weightMap=weights)},
              loss_weights={'cell_output': 1., 'margin_output': 1.}, metrics=[dice_coef])


    return model

def get_UNET_twoHead_multiscale_residual_deep(input_shape,cellLoss, marginLoss):
# ALL DECONV IN DECODING PATH
    inputs = Input(input_shape+(img_chls,), name='main_input') # size: 1024
    weights = Input(input_shape+(1,), name='weights')
    
    conv1 = _conv_bn_relu(inputs, 64)# size: 256
    conv1 = _conv_bn_relu(conv1, 64)
    conv1 = _conv_bn_relu(conv1, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)#_conv_bn_relu(conv1, 32, (2, 2), strds=(2, 2), useRegulizer=False)#
 
    conv2 = residual_conv(pool1, 64, (3,3))
    conv2 = multiScaleConv(conv2, 256) # size: 128
    conv2 = residual_conv(conv2, 64, (3,3))
    conv2 = residual_conv(conv2, 64, (3,3))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)#_conv_bn_relu(conv2, 64, (2, 2), strds=(2, 2), useRegulizer=False)#
 
    conv3 = residual_conv(pool2, 128, (3,3))
    conv3 = multiScaleConv(conv3, 256)# size: 64
    conv3 = residual_conv(conv3, 128, (3,3))
    conv3 = residual_conv(conv3, 128, (3,3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)#_conv_bn_relu(conv3, 128, (2, 2), strds=(2, 2), useRegulizer=False)#
 
    conv4 = residual_conv(pool3,256, (3,3))# size: 32
    conv4 = residual_conv(conv4,256, (3,3))# size: 32
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)#_conv_bn_relu(conv4, 256, (2, 2), strds=(2, 2), useRegulizer=False)#
     
    conv5 = residual_conv(pool4, 512, (3,3))# size: 16
    conv5 = residual_conv(conv5, 512, (3,3))# size: 16
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
     
    conv51 = residual_conv(pool5, 1024, (3,3))# size: 8
#    conv51 = residual_deconv(conv51, 512, (3,3))# size: 8
     
    toConcat601 = _deconv_bn_relu(conv51, 512, (2, 2), strds=(2, 2))
    up61 = concatenate([toConcat601, conv5], axis=3)
    conv61 = residual_deconv(up61, 512, (3,3))
    conv61 = residual_deconv(conv61, 512, (3,3))
     
    toConcat60 = _deconv_bn_relu(conv61, 256, (2, 2), strds=(2, 2))
    up6 = concatenate([toConcat60, conv4], axis=3)
    conv6 = residual_deconv(up6, 256, (3,3))
    conv6 = residual_deconv(conv6, 256, (3,3))
     
    toConcat70 = _deconv_bn_relu(conv6, 128, (2, 2), strds=(2, 2))
    up7 = concatenate([toConcat70, conv3], axis=3)
    conv7 = residual_deconv(up7, 128, (3,3))
    conv7 = multiScaleDeconv(conv7, 256)
    conv7 = residual_deconv(conv7, 128, (3,3))
    conv7 = residual_deconv(conv7, 128, (3,3))
 
    toConcat80 = _deconv_bn_relu(conv7, 64, (2, 2), strds=(2, 2))
    up8 = concatenate([toConcat80, conv2], axis=3)
    conv8 = residual_deconv(up8, 64, (3,3))
    conv8 = multiScaleDeconv(conv8, 256)
    conv8 = residual_deconv(conv8, 64, (3,3))
    conv8 = residual_deconv(conv8, 64, (3,3))
     
    # First Head: Cell Marker
    toConcat90 = _deconv_bn_relu(conv8, 32, (2, 2), strds=(2, 2))
    up9 = concatenate([toConcat90, conv1], axis=3)
    conv9 = _deconv_bn_relu(up9, 64, (3,3))
    conv9 = _deconv_bn_relu(conv9, 64, (3,3))
    conv9 = _deconv_bn_relu(conv9, 32, (3,3))
    conv9 = _deconv_bn_relu(conv9, 32, (3,3))
     
    cell_output = Conv2D(1, (1, 1), activation='sigmoid', name='cell_output')(conv9)
     
    margin_output = Conv2D(1, (1, 1), activation='sigmoid', name='margin_output')(conv9)

    model = Model(inputs=[inputs, weights], outputs=[cell_output, margin_output])
    
    '''* Different compile options: Check to see which one is good!?!you can use a trustable learning rate like = 5e-3 for your experiments*'''
    model.compile(optimizer=Adam(lr=3e-3, decay=1e-4, amsgrad=True),
              loss={'cell_output': getLoss(cellLoss, weightMap=weights), 'margin_output': getLoss(marginLoss,weightMap=weights)},
              loss_weights={'cell_output': 1., 'margin_output': 1.}, metrics=[dice_coef])
 
 
    return model

def get_spagettiNet_twoHeaded_multiscale_residual_veryDeep(input_shape,cellLoss, marginLoss):
# ALL DECONV IN DECODING PATH
    inputs = Input(input_shape+(img_chls,), name='main_input') # size: 1024
    weights = Input(input_shape+(1,), name='weights')

    conv1 = _conv_bn_relu(inputs, 64)# size: 256
    conv1 = _conv_bn_relu(conv1, 64)
    conv1 = _conv_bn_relu(conv1, 64)
#    conv1 = concatenate([conv1, inputs], axis=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)#_conv_bn_relu(conv1, 32, (2, 2), strds=(2, 2), useRegulizer=False)#

    conv2 = residual_conv(pool1, 64, (3,3)) # size: 128
    conv2 = multiScaleConv(conv2, 256) 
    conv2 = residual_conv(conv2, 64, (3,3))
    conv2 = residual_conv(conv2, 64, (3,3))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)#_conv_bn_relu(conv2, 64, (2, 2), strds=(2, 2), useRegulizer=False)#

    conv3 = residual_conv(pool2, 128, (3,3)) # size: 64
#    conv3 = multiScaleConv(conv3, 256)
    conv3 = residual_conv(conv3, 128, (3,3))
    conv3 = residual_conv(conv3, 128, (3,3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)#_conv_bn_relu(conv3, 128, (2, 2), strds=(2, 2), useRegulizer=False)#

    conv4 = residual_conv(pool3,256, (3,3))# size: 32
    conv4 = residual_conv(conv4,256, (3,3))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)#_conv_bn_relu(conv4, 256, (2, 2), strds=(2, 2), useRegulizer=False)#
    
    conv5 = residual_conv(pool4, 512, (3,3))# size: 16
    conv5 = residual_conv(conv5, 512, (3,3))# size: 16
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    
    conv55 = residual_conv(pool5, 1024, (3,3))# size: 16
    
    toConcat55 = _deconv_bn_relu(conv55, 512, (2, 2), strds=(2, 2))
    up55 = concatenate([toConcat55, conv5], axis=3)
    conv66 = residual_deconv(up55, 512, (3,3))
    conv66 = residual_deconv(conv66, 512, (3,3))
    ### Add dropout here!!!###
    
    # Gland segmentation Head
    toConcat60 = _deconv_bn_relu(conv66, 256, (2, 2), strds=(2, 2))
    toConcat61 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//8,input_shape[1]//8),method=tf.image.ResizeMethod.BILINEAR))(conv1)
    toConcat62 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//8,input_shape[1]//8),method=tf.image.ResizeMethod.BILINEAR))(conv2)
    toConcat63 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//8,input_shape[1]//8),method=tf.image.ResizeMethod.BILINEAR))(conv3)
    up6 = concatenate([toConcat60, toConcat61, toConcat62, toConcat63, conv4], axis=3)
    conv6 = residual_deconv(up6, 256, (3,3))
    conv6 = residual_deconv(conv6, 256, (3,3))
    conv6 = residual_deconv(conv6, 256, (3,3))
    conv6 = residual_deconv(conv6, 256, (3,3))
    
    toConcat70 = _deconv_bn_relu(conv6, 128, (2, 2), strds=(2, 2))
    toConcat71 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//4,input_shape[1]//4),method=tf.image.ResizeMethod.BILINEAR))(conv1)
    toConcat72 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//4,input_shape[1]//4),method=tf.image.ResizeMethod.BILINEAR))(conv2)
    toConcat73 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//4,input_shape[1]//4),method=tf.image.ResizeMethod.BILINEAR))(conv4)
    up7 = concatenate([toConcat70, toConcat71, toConcat72, toConcat73, conv3], axis=3)
    conv7 = residual_deconv(up7, 128, (3,3))
#    conv7 = multiScaleDeconv(conv7, 256)
    conv7 = residual_deconv(conv7, 128, (3,3))

    toConcat80 = _deconv_bn_relu(conv7, 64, (2, 2), strds=(2, 2))
    toConcat81 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//2,input_shape[1]//2),method=tf.image.ResizeMethod.BILINEAR))(conv1)
    toConcat82 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//2,input_shape[1]//2),method=tf.image.ResizeMethod.BILINEAR))(conv3)
    toConcat83 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0]//2,input_shape[1]//2),method=tf.image.ResizeMethod.BILINEAR))(conv4)
    up8 = concatenate([toConcat80, toConcat81, toConcat82, toConcat83, conv2], axis=3)
    conv8 = residual_deconv(up8, 64, (3,3))
    conv8 = multiScaleDeconv(conv8, 256)
    conv8 = residual_deconv(conv8, 64, (3,3))

    toConcat90 = _deconv_bn_relu(conv8, 32, (2, 2), strds=(2, 2))
    toConcat91 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0],input_shape[1]),method=tf.image.ResizeMethod.BILINEAR))(conv2)
    toConcat92 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0],input_shape[1]),method=tf.image.ResizeMethod.BILINEAR))(conv3)
    toConcat93 = Lambda(lambda image: tf.image.resize_images(image, (input_shape[0],input_shape[1]),method=tf.image.ResizeMethod.BILINEAR))(conv4)
    up9 = concatenate([toConcat90, toConcat91, toConcat92, toConcat93, conv1], axis=3)
    conv9 = _deconv_bn_relu(up9, 64, (3,3))
    conv9 = _deconv_bn_relu(conv9, 64, (3,3))
    conv9 = _deconv_bn_relu(conv9, 32, (3,3))
    conv9 = _deconv_bn_relu(conv9, 32, (3,3))
    
    cell_output = Conv2D(1, (1, 1), activation='sigmoid', name='cell_output')(conv9)
    
    margin_output = Conv2D(1, (1, 1), activation='sigmoid', name='margin_output')(conv9)

    model = Model(inputs=[inputs, weights], outputs=[cell_output, margin_output])
    model.compile(optimizer=Adam(lr=3e-3, decay=1e-4, amsgrad=True),
              loss={'cell_output': getLoss(cellLoss, weightMap=weights), 'margin_output': getLoss(marginLoss,weightMap=weights)},
              loss_weights={'cell_output': 1., 'margin_output': 1.}, metrics=[dice_coef])

    return model
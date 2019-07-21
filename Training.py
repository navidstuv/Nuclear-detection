# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 12:53:23 2018

@author: Navid
"""

from __future__ import print_function
import gc
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from skimage.io import imsave
import numpy as np
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.utils import multi_gpu_model
# from nasUnet import NASUNet
# from se import squeeze_excite_block
# from data import load_train_data, load_test_data
# from elastic_functions import perform_elastic_3image
# import matplotlib.pyplot as plt
# from data import load_train_data, load_test_data
from utils.image_segmentation import ImageDataGenerator
import scipy.io as sio
# %matplotlib inline
import warnings
from model_factory import getModel

warnings.filterwarnings("ignore")

# Setting Parameters
seeddd = 1
np.random.seed(seeddd)
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 256  # 480#640
img_cols = 256  # 768#1024
img_chnls = 7
input_shape = (img_rows, img_cols)

modelType = 'spagetti-singleHead-multiscale-residual-deep'
cellLoss = 'mean_absolute_error'
marginLoss = ''
batchSize = 4  # set this as large as possible
batchSizeVal = 1  # leaave this to 1 anyway

gpus = [x.name for x in K.device_lib.list_local_devices() if x.name[:4] == '/gpu']
multi_gpu = False  # NAVID DO THIS!


class ModelCheckpointMGPU(ModelCheckpoint):
    def __init__(self, original_model, filepath, monitor='val_loss', verbose=0, save_best_only=False,
                 save_weights_only=False, mode='auto', period=1):
        self.original_model = original_model
        super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

    def on_epoch_end(self, epoch, logs=None):
        self.model = self.original_model
        super().on_epoch_end(epoch, logs)


from skimage.color import rgb2hsv, rgb2lab


def addAuxialaryChannels(img):
    img = np.uint8(img)
    HSV = np.array(rgb2hsv(img))
    Lab = np.array(rgb2lab(img))
    output = img
    output = np.append(img, np.uint8(255 * HSV), axis=2)
    L = np.uint8(255 * Lab[:, :, 0] / 100)
    L = L[..., np.newaxis]
    output = np.append(output, L, axis=2)
    return output.astype(np.float32)


''' Loading and preprocessing data'''
print('-' * 30)
print('Loading training data...')
print('-' * 30)
mat_contents = sio.loadmat('data.mat')
imgs_all = mat_contents['images']
imgs_all = np.transpose(imgs_all,[3,0,1,2])
masks_all = mat_contents['GTs']
masks_all = np.transpose(masks_all ,[2,0,1])
masks_all = masks_all[..., np.newaxis]
masks_all = masks_all/255.

np.random.seed(seeddd)
rndIdx = np.arange(masks_all.shape[0])
np.random.shuffle(rndIdx)
rndValImgNumbers = rndIdx[0:masks_all.shape[0]//5]
rndTrImgNumbers = rndIdx[masks_all.shape[0]//5:]


# Initiating data generators
train_gen_args = dict(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=40.,
    # zoom_range=(.7, 1.4),  # (0.7, 1.3),
    # shear_range=.3,
    fill_mode='constant',  # Applicable to image onlyS
    cval='random',
    # 'random', # random option is newly added. It may could help neglecting color charts and clothes around images
    channel_shift_range=40.,  # This must be in range of 255?
    contrast_adjustment=True,  #####MOSI
    illumination_gradient=True,
    intensity_scale_range=0.3,  #####MOSI
    sharpness_adjustment=True,
    apply_noise=True,
    # elastic_deformation=True,
    rescale=1. / 255
)

image_datagen = ImageDataGenerator(**train_gen_args
                                   , preprocessing_function=addAuxialaryChannels
                                   )

image_datagen_val = ImageDataGenerator(preprocessing_function=addAuxialaryChannels,
                                       rescale=1. / 255)

'''
Cross-Validation::: Loop over the different folds and perform train on them.
Save the best model which has best performance on validation set in each fold.
'''
modelBaseName = 'Mask_%s_%s' % (modelType, cellLoss)
if not os.path.exists(modelBaseName):
    os.mkdir(modelBaseName)



train_generator = image_datagen.flow(
    imgs_all[rndTrImgNumbers], mask1=masks_all[rndTrImgNumbers],
    shuffle=True,
    batch_size=batchSize,
    color_mode='rgbhsvl',  # rgbhsvl
    seed=seeddd)
val_generator = image_datagen_val.flow(
    imgs_all[rndValImgNumbers], mask1=masks_all[rndValImgNumbers],
    shuffle=False,
    batch_size=batchSizeVal,
    color_mode='rgbhsvl',
    seed=seeddd)

num_train = len(rndTrImgNumbers)  # 0
num_val = len(rndValImgNumbers)  # 0

print('-' * 30)
print('Creating and compiling model...')
print('-' * 30)
modelName = "%s_fold" % (modelBaseName)
modelSaveName = "./%s/weights-%s.h5" % (modelBaseName, modelName)
modelLogName = "./%s/Log-%s.log" % (modelBaseName, modelName)
csv_logger = CSVLogger(modelLogName, append=True, separator='\t')

if multi_gpu:
    with K.tf.device("/cpu:0"):
        model = getModel(modelType, cellLoss, marginLoss, input_shape)
else:
    model = getModel(modelType, cellLoss, marginLoss, input_shape)

if multi_gpu:
    model = multi_gpu_model(model, len(gpus))

model_checkpoint = ModelCheckpointMGPU(model, filepath=modelSaveName, monitor='val_loss', mode='min',
                                       save_best_only=True)

print('-' * 30)
print('Fitting model...')
print('-' * 30)
history = model.fit_generator(train_generator, steps_per_epoch=num_train // batchSize, nb_epoch=10,
                              validation_data=val_generator,
                              validation_steps=num_val // batchSizeVal, callbacks=[model_checkpoint, csv_logger],
                              max_queue_size=50, workers=8)
history = model.fit_generator(train_generator, steps_per_epoch=num_train // batchSize, nb_epoch=300,
                              validation_data=val_generator,
                              validation_steps=num_val // batchSizeVal, callbacks=[model_checkpoint, csv_logger],
                              max_queue_size=50, workers=8)

print('-' * 30)
print('Predicting on validation...')
print('-' * 30)
model.load_weights(modelSaveName)
val_predicts1 = model.predict_generator(val_generator, steps=num_val // batchSize)
pred_dir = "./%s/valPred_%s" % (modelBaseName, modelBaseName)
imgs_mask_test1 = np.matrix.squeeze(val_predicts1, axis=3)

if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)
for image_id in range(0, len(imgs_mask_test1)):
    mask1 = np.uint8(imgs_mask_test1[image_id, :, :] * 255)
    imsave(os.path.join(pred_dir, str(image_id) + '_mask.png'), mask1)

del model
K.clear_session()
gc.collect()

print('*' * 90)
print('Cross validation experiments are done.')
print('*' * 90)


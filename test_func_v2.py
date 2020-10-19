'''
Testing time:
Perform each of model variations on the test images and average their output to
form the final output (Bagging style).
Test predictions can be made on the image crops or its full domain.
we also can use augomentation in testing phase to make better predictions.
'''
from __future__ import print_function
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from skimage.io import imsave, imread
from utils.image_segmentation import ImageDataGenerator
from model.model_factory import getModel
from skimage.feature import peak_local_max
from utils.util_funcs import addAuxialaryChannels, adaptiveIntensityScaling, contrastEnhancement, sharpnessEnhancement,crop


image_datagen_test = ImageDataGenerator(preprocessing_function=addAuxialaryChannels,
                                        rescale=1. / 255)

# print('-' * 30)
# print('Loading and preprocessing test data (ONE BY ONE APPROACH)...')
# print('-' * 30)

iter = 0
im_iter = 0
# for image_name in images:
#     start_time = time.time()
#     if not (fileFormat in image_name):
#         continuer
def model_initialize_detector(modelType = 'spagetti-singleHead-multiscale-residual-deep',cellLoss = '', marginLoss = '',image_size = 1024, wights_path = ''):
    input_shape = (image_size, image_size)
    model = getModel(modelType, cellLoss, marginLoss, input_shape)
    model.load_weights(wights_path)
    return model
def detector(img, model,image_size = 1024):
    # Creating instance of base model
    # modelType = 'spagetti-singleHead-multiscale-residual-deep'
    # cellLoss = 'mean_absolute_error'
    # marginLoss = ''
    test_batchSize = 1
    seeddd = 1
    Row = 1024
    Col = 1024
    h, w, cc = img.shape
    stride = 1024
    patch_size = 1024
    ims, im = crop(img, patch_size, patch_size, stride)
    IM = np.zeros((6, h + patch_size, w + patch_size))
    ONE = np.zeros(((6, h + patch_size, w + patch_size)))
    kk = 0
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            image_patches = np.expand_dims(ims[kk], axis=0)
            dummyWeight = np.float32(image_patches[:, :, :, 0:1])
            predInput = np.ndarray((6, Row, Col, 3), dtype=np.uint8)
            outputMarker = np.zeros((6, Row, Col, 1), dtype='float64')
            # image = np.subtract(image, mean)
            # image = np.divide(image, std)
            numAug = 6
            numOutputs = np.float(numAug)
            predInput[0, :, :, :] = image_patches[0, :, :, :]  # original image
            predInput[1, :, :, :] = image_patches[0, ::-1, ::-1, :]
            predInput[2, :, :, :] = adaptiveIntensityScaling(image_patches[0, ::-1, :, :])
            predInput[3, :, :, :] = contrastEnhancement(
                image_patches[0, ::-1, ::-1, :])  # contrastEnhancing(fliplr(flipud))
            predInput[4, :, :, :] = sharpnessEnhancement(image_patches[0, :, ::-1, :])  # sharpenning(original image)
            predInput[5, :, :, :] = sharpnessEnhancement(adaptiveIntensityScaling(image_patches[0, :, :, :]))

            test_generator = image_datagen_test.flow(predInput, shuffle=False,
                                                     batch_size=test_batchSize, color_mode='rgbhsvl', seed=seeddd)

            # parameters = model.predict(image)
            parameters = model.predict_generator(test_generator, steps=numAug // test_batchSize)
            # parameters = model.predict_generator(test_generator, steps=numAug // test_batchSize)
            parameters[1, :, :, :] = parameters[1, ::-1, ::-1, :]
            parameters[2, :, :, :] = parameters[2, ::-1, :, :]
            parameters[3, :, :, :] = parameters[3, ::-1, ::-1, :]
            parameters[4, :, :, :] = parameters[4, :, ::-1, :]
            outputMarker[:, :, :, :] = parameters

            kk = kk + 1
            ONE[:, i:i + patch_size, j:j + patch_size] = ONE[:, i:i + patch_size, j:j + patch_size] + 1
            IM[:, i:i + patch_size, j:j + patch_size] = np.squeeze(outputMarker) + IM[:, i:i + patch_size, j:j + patch_size]

    IM = IM[:, 0:h, 0:w]
    ONE = ONE[:, 0:h, 0:w]
    IM = IM / ONE
    marker = IM[:, 0:h, 0:w]
    marker_mean = np.mean(marker, axis=0)

    marker_mean[marker_mean < 0.1] = 0
    coordinates = peak_local_max(np.squeeze(marker_mean), min_distance=5)
    return marker_mean,  coordinates

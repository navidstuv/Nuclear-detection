import numpy as np
from skimage.filters import gaussian
from skimage import exposure
from skimage.color import rgb2hsv, rgb2lab





def crop(im, height, width, stride):
    im = np.pad(im, ((0, height), (0, width), (0, 0)), 'reflect')
    imgheight, imgwidth, _ = im.shape
    ims = []
    for i in range(0, imgheight - height, stride):
        for j in range(0, imgwidth - width, stride):
            box = im[i:i + height, j: j + width, :]
            ims.append(box)
    ims = np.array(ims)
    return ims, im
def _unsharp_mask_single_channel(image, radius, amount):
    """Single channel implementation of the unsharp masking filter."""
    blurred = gaussian(image, sigma=radius, mode='reflect')
    result = image + (image - blurred) * amount
    result = np.clip(result, 0, 1)
    return result

def sharpnessEnhancement(imgs):  # needs the input to be in range of [0,1]
    imgs_out = imgs.copy()
    for channel in range(imgs_out.shape[-1]):
        imgs_out[..., channel] = 255 * _unsharp_mask_single_channel(imgs_out[..., channel] / 255., 2, .8)
    return imgs_out

def contrastEnhancement(imgs):  # needs the input to be in range of [0,255]
    imgs_out = imgs.copy()
    p2, p98 = np.percentile(imgs_out, (2, 98))  #####
    if p2 == p98:
        p2, p98 = np.min(imgs_out), np.max(imgs_out)
    if p98 > p2:
        imgs_out = exposure.rescale_intensity(imgs_out, in_range=(p2, p98), out_range=(0., 255.))
    return imgs_out

def adaptiveIntensityScaling(imgs):
    imgs_out = np.float32(imgs) / 255.
    m = 0.6 / (np.mean(imgs_out) + 1e-7)
    if m < 0.93:
        s = .9
    else:
        s = np.min([np.max([m, 1.3]), 1.6])

    return np.uint8(np.clip(imgs * s, 0., 255.))

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
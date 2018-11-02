import cv2
import numpy as np


def process_image(img, img_size):


    # read the image
    img = cv2.imread(img)

    # flip image at random if flag is selected
    img = cv2.flip(img, 1)

    # rescale image
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32)

    return img


def normalize_image(img, pixel_depth, mean_vec=None):

    if mean_vec==None:
        img = np.array(img, dtype=np.float32)

        img[:, :, 0] = (img[:, :, 0].astype(float) - pixel_depth / 2) / pixel_depth
        img[:, :, 1] = (img[:, :, 1].astype(float) - pixel_depth / 2) / pixel_depth
        img[:, :, 2] = (img[:, :, 2].astype(float) - pixel_depth / 2) / pixel_depth
    else:
        img -= mean_vec

    return img

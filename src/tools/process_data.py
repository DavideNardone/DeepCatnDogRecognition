import cv2
import numpy as np


def process_image(img, img_size):
    '''
        Maintain aspect ratio of image while resizing
    '''
    # img = cv2.imread(img, cv2.IMREAD_COLOR)
    # if (img.shape[0] >= img.shape[1]):  # height is greater than width
    #     resizeto = (img_size, int(
    #         round(img_size * (float(img.shape[1]) / img.shape[0]))))
    # else:
    #     resizeto = (int(round(img_size * (float(img.shape[0]) / img.shape[1]))), img_size)
    #
    # img = cv2.resize(img, (resizeto[1], resizeto[
    #     0]), interpolation=cv2.INTER_CUBIC)
    # img = cv2.copyMakeBorder(
    #     img, 0, img_size - img.shape[0], 0, img_size - img.shape[1], cv2.BORDER_CONSTANT, 0)

    # read the image
    img = cv2.imread(img)

    # flip image at random if flag is selected
    img = cv2.flip(img, 1)

    # rescale image
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32)

    # img = normalize_image(img, pixel_depth)

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

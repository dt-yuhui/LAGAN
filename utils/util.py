import torch
import os
import numpy as np
import cv2


def tensor2im(input_image,  imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            if input_image.dim() == 4:
                assert input_image.shape[0] == 1, 'the batchsize of img tensor should be 1'
                input_image.squeeze_(0)
        else:
            return input_image
        image_numpy = input_image.data.cpu().float().numpy()  # convert it into a numpy array
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def scale_image(img, target_size=None):
    '''
    min-max normalization
    Args:
        img: input image
        target_size: target size

    Returns:
        normalized_img
    '''
    img = np.float32(img)
    img = img - np.min(img)
    img = img / (1e-7 + np.max(img))
    if target_size is not None:
        img = cv2.resize(img, target_size)

    return img

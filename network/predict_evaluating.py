# coding:utf-8

from __future__ import print_function
import os
from skimage.transform import resize
from skimage.io import imsave
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import optimizers
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from keras import backend as K
import RC_SPCNet
import numpy as np
from data import load_train_data, load_test_data
import tensorflow as tf

import cv2

os.environ['CUDA_VISIBLE_DEVICES']='0'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# input image dimensions
img_rows, img_cols = 512, 512
mask_rows, mask_cols = 512, 512
# Images are RGB.
img_channels = 1

smooth = 1.


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols, imgs.shape[3]), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        # [i, :, :, 0]   # 取中间的img_rows和img_cols
        imgs_p[i, :, :, 0] = resize(imgs[i, :, :, 0], (img_cols, img_rows), preserve_range=True)
    return imgs_p


if __name__ == '__main__':

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()

    mean = np.mean(imgs_test)  # mean for data centering
    std = np.std(imgs_test)  # std for data normalization

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model = RC_SPCNet.RC_SPCNet([img_rows, img_cols, img_channels])

    model.compile(optimizer="Adam",
                  loss='mse', metrics=['accuracy'])
    model.load_weights('RC_SPCNet.hdf5')

    print('-' * 30)
    print('evaluating the test...')
    print('-' * 30)


    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_predict = model.predict(imgs_test, batch_size=10, verbose=1)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_predict, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        filename = os.path.join(pred_dir, str(image_id) + '_pred.png')  # save imgs
        imsave(filename, image)




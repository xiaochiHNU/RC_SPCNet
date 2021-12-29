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
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'  # 仅使用第2块GPU, 从0开始

# input image dimensions
img_rows, img_cols = 512, 512
mask_rows, mask_cols = 512, 512
# Images are RGB.
img_channels = 1


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols, imgs.shape[3]), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        # [i, :, :, 0]   # 取中间的img_rows和img_cols
        imgs_p[i, :, :, 0] = resize(imgs[i, :, :, 0], (img_rows, img_cols), preserve_range=True)
    return imgs_p


if __name__ == '__main__':

    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)

    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    # change imgs_mask_train form (*,512,512,1) to (*,512,512,2)
    mask_train = np.ndarray((imgs_mask_train.shape[0], mask_rows, mask_cols, 2), dtype=np.float32)
    mask_train[:, :, :, 0] = imgs_mask_train[:, :, :, 0]
    mask_train[:, :, :, 1] = 1 - imgs_mask_train[:, :, :, 0]

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = RC_SPCNet.RC_SPCNet([img_rows, img_cols, img_channels])
    model_checkpoint = ModelCheckpoint('RC_SPCNet.{epoch:02d}-{val_loss:.5f}.hdf5', save_best_only=True, save_weights_only=True,
                                       verbose=1)   # save_best_only=True

    model.compile(optimizer=Adam(lr=1e-4),
                  loss='mse', metrics=['accuracy'])   # loss = 'mse'

    print('-' * 30)
    print('loading pretrain weights...')
    print('-' * 30)

    print('-'*30)
    print('compiling and Fitting model...')
    print('-'*30)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    Reduce_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=3, verbose=1)

    history = model.fit(imgs_train, mask_train, batch_size=2, epochs=50, verbose=1, shuffle=True,
              validation_split=0.3,
              callbacks=[model_checkpoint, early_stopping, Reduce_rate])

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()




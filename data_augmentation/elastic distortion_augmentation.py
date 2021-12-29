import numpy as np
import os
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from skimage.io import imsave
from PIL import Image

import matplotlib.pyplot as plt

# Function to distort image
def elastic_transform(image1, image2, alpha, sigma):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    assert image1.shape == image2.shape
    # Take measurements
    imshape = image1.shape

    rng = np.random.RandomState(None)

    # Make random fields
    dx = rng.uniform(-1, 1, imshape) * alpha
    dy = rng.uniform(-1, 1, imshape) * alpha
    # Smooth dx and dy
    sdx = gaussian_filter(dx, sigma=sigma, mode='reflect')
    sdy = gaussian_filter(dy, sigma=sigma, mode='reflect')
    # Make meshgrid
    x, y = np.meshgrid(np.arange(imshape[1]), np.arange(imshape[0]))
    # Distort meshgrid indices
    distinds = (y + sdy).reshape(-1, 1), (x + sdx).reshape(-1, 1)
    # Map cooordinates from image to distorted index set
    transformedimage1 = map_coordinates(image1, distinds, mode='reflect').reshape(imshape)
    transformedimage2 = map_coordinates(image2, distinds, mode='reflect').reshape(imshape)
    return transformedimage1, transformedimage2

# Define function to draw a grid
def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))

if __name__ == '__main__':

    elastic_times = 40

    # data_path
    data_path = 'E:/RC_SPCNet/trains'
    train_data_path = os.path.join(data_path)
    images = os.listdir(train_data_path)
    total = int(len(images) / 2 * elastic_times * 8)


    # pred_dir
    pred_dir = 'E:/RC_SPCNet/trains_aug/'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    num = 1

    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.png'

        im = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)

        rows, cols = im.shape

        # elastic transform process
        for index in range(elastic_times):
            im_t, mask_t = elastic_transform(im, im_mask, 2500, 50)

            for i in range(4):
                img = Image.fromarray(im_t)
                mask = Image.fromarray(mask_t)

                rotate_img = img.rotate(i * 90)   # 旋转角度
                rotate_img.save(pred_dir + str(num) + '.png')

                rotate_mask = mask.rotate(i * 90)
                rotate_mask.save(pred_dir + str(num) + '_mask.png')

                num += 1
                if num % 100 == 0:
                    print('Done: {0}/{1} images'.format(num, total))

                rotate_img.transpose(Image.FLIP_LEFT_RIGHT).save(pred_dir + str(num) + '.png')
                rotate_mask.transpose(Image.FLIP_LEFT_RIGHT).save(pred_dir + str(num) + '_mask.png')
                num += 1

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

    elastic_times = 20    # 每张图elastic transform 20 次，旋转翻转8次，一共扩大160倍

    # 图片读取路径
    data_path = 'D:/Multicut/ourdata/ISBI2012 challenge/ISBI2012 traindata/hist_trainsform/trains/'
    train_data_path = os.path.join(data_path)
    images = os.listdir(train_data_path)
    total = int(len(images) / 2 * elastic_times * 8)     # total应该是整数才对


    # 图片保存路径
    pred_dir = 'D:/Multicut\ourdata/ISBI2012 challenge/ISBI2012 traindata/raw/active learning/trains/'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    num = 1

    # 读取30张图片
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.png'

        #im = Image.open(os.path.join(train_data_path, image_name))
        #im_mask = Image.open(os.path.join(train_data_path, image_mask_name))

        im = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)

        rows, cols = im.shape

        # 每张图片elastic transform 20 次 保存
        for index in range(elastic_times):
            im_t, mask_t = elastic_transform(im, im_mask, 2500, 50)

            for i in range(4):     # 图片旋转上下翻转，扩充8倍
                img = Image.fromarray(im_t)   # 将cv2的图像格式转成PIL格式
                mask = Image.fromarray(mask_t)

                rotate_img = img.rotate(i * 90)   # 旋转角度
                rotate_img.save(pred_dir + str(num) + '.png')     #保存

                rotate_mask = mask.rotate(i * 90)
                rotate_mask.save(pred_dir + str(num) + '_mask.png')

                num += 1
                if num % 100 == 0:
                    print('Done: {0}/{1} images'.format(num, total))

                rotate_img.transpose(Image.FLIP_LEFT_RIGHT).save(pred_dir + str(num) + '.png')   # 左右变换  保存
                rotate_mask.transpose(Image.FLIP_LEFT_RIGHT).save(pred_dir + str(num) + '_mask.png')
                num += 1


'''
                # 保存图片
                imgt_filename = os.path.join(pred_dir, str(num) + '.png')         # img_t的保存名称 需要将int型转为string型
                maskt_filename = os.path.join(pred_dir, str(num) + '_mask.png')   # mask_t的保存名称
                imsave(imgt_filename, rotate_img)
                imsave(maskt_filename, rotate_mask)
'''



    #im = cv2.imread("D:/Multicut/ourdata/ISBI2012 challenge/ISBI2012 traindata/source imgs/trains/img001.png", -1)
    #im_mask = cv2.imread("D:/Multicut/ourdata/ISBI2012 challenge/ISBI2012 traindata/source imgs/masks/img001.png", -1)

    #im_t, mask_t = elastic_transform(im, im_mask, 2500, 50)

    # Draw grid lines
    #draw_grid(im, 50)
    #draw_grid(im_mask, 50)

    # Merge images into separete channels (shape will be (cols, rols, 2))
    #im_merge = np.concatenate((im[..., None], im_mask[..., None]), axis=2)

    # First sample...

    #%matplotlib inline

    # Apply transformation on image
    #im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)

    # Split image and mask
    #im_t = im_merge_t[..., 0]
    #im_mask_t = im_merge_t[..., 1]

    #cv2.imshow('image', im)
    #cv2.imshow('mask', im_mask)
    #cv2.imshow('image1', im_t)
    #cv2.imshow('mask1', mask_t)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Display result
    #plt.figure(figsize=(16, 14))
    #plt.imshow(np.c_[np.r_[im, im_mask], np.r_[im_t, im_mask_t]], cmap='gray')
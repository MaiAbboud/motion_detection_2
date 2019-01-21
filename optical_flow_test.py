"""
from distutils.cmd import install_misc

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
import scipy.ndimage.filters as fltr
import timeit
from random import randint
import random
im1 = cv.imread('/home/mai/Pictures/vlcsnap-2018-12-03-16h29m13s551.jpg')
im1_gray = cv.cvtColor(im1, cv.COLOR_RGB2GRAY)
im2 = cv.imread('/home/mai/Pictures/vlcsnap-2018-12-03-16h29m19s700.jpg')
im2_gray = cv.cvtColor(im2, cv.COLOR_RGB2GRAY)

h = np.shape(im1_gray)[0]
w = np.shape(im1_gray)[1]
print(w)
step = 10
margin = 5
thresh_angle = 1
num_grid = int(h*w/(step**2))
# print(num_grid)
grid_test = True

high_step = 20
width_step = 20
half_step = np.int(high_step/2)

h_num = np.int(h/high_step)
w_num = np.int(w/width_step)

# p0 = np.zeros((h_num*w_num, 1, 2))
feature_params = dict(maxCorners=1,
                      qualityLevel=0.01,
                      minDistance=7,
                      blockSize=7)

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
p0 = [[[0, 0]]]
color = np.random.randint(0, 255, (num_grid, 3))
in_loop = False
in_loop_once = True
ctr = 0
ctr2 = 0
if grid_test:
    mask_use = np.zeros(im1_gray.shape, np.uint8)
    for n in range(2, w_num):
        for m in range(2, h_num):
            mask_use = np.zeros(im1_gray.shape, np.uint8)
            mask_use[(m-1)*high_step:m*high_step, (n-1)*width_step:n*width_step] =\
                im1_gray[(m-1)*high_step:m*high_step, (n-1)*width_step:n*width_step]
            # if in_loop:
            #     if np.size(p0_temp) > 1:
            #         p_prev = p0_temp
            # # start = timeit.default_timer()
            p0_temp = cv.goodFeaturesToTrack(im1_gray, mask=mask_use, **feature_params)
            # print(np.shape(p0_temp))
            # # p0_temp2 = np.array([w_num, h_num])
            # # p0_temp = np.reshape(p0_temp2, (1, 1, 2))
            ctr = ctr + 1
            # print(ctr)
            # if np.size(p0_temp) > 1:
            #     for i in p0_temp:
            #         x, y = i.ravel()
            #         im_draw = cv.circle(im1_gray, (x, y), 1, 255, 2)
            #     cv.imshow('frame', im_draw)
            #     cv.waitKey(1)

            # else:
            #     p0_temp = p_prev
            #
            # print(np.size(p0_temp))
            if in_loop_once: #& np.size(p0_temp) > 1:
                p0 = p0_temp
            elif np.size(p0_temp) > 1:
                p0 = np.concatenate((p0, p0_temp))
                ctr2 = ctr2 + 1
            elif np.size(p0_temp) == 1:
                # in_loop_once == False:
            # if np.size(p0_temp) == 1:  # in_loop_once == False:
                p0_temp = [[[(n-1)*width_step + half_step, (m-1)*high_step + half_step]]]
                p0 = np.concatenate((p0, p0_temp))
            in_loop_once = False
            ctr = ctr + 1
            # if ctr == (w_num-1)*(h_num-1)-1:
            #     break
    # print(np.shape(p0))

    # print(np.shape(p0))
    # print(w_num)
    # print(h_num)
    p0 = np.float32(p0)
    mask = np.zeros_like(im1)
    p1, st, err = cv.calcOpticalFlowPyrLK(im1_gray, im2_gray, p0, None, **lk_params)
    print('w', w_num)
    print('h', h_num)

    print('ctr', ctr)
    print('ctr2', ctr2)
    # diff = abs(p0[st != 1] - p1[st != 1])
    print(np.shape(p0))
    print(np.shape(p1))
    print('p0_max = ', max(p0[:, :, 0]))
    print('p0_min = ', max(p0[:, :, 1]))
    print('p1_max = ', max(p1[:, :, 0]))
    print('p1_min = ', max(p1[:, :, 1]))
    # print('st == \n')
    # print(np.shape(p1[st == 1]))
    # print('tot size == \n')
    # print(np.shape(p1))
    # print('diff = ')
    # print(p0[st != 1] - p1[st != 1])

    good_new = p1#[st == 1]
    good_old = p0#[st == 1]
    # print(np.shape(good_new))
    # delta_x = good_new[:, 0]-good_old[:, 0]
    # delta_y = good_new[:, 1]-good_old[:, 1]

    delta_x = p1[:, :, 0] - p0[:, :, 0]
    delta_y = p1[:, :, 1] - p0[:, :, 1]
    # theta = np.arctan2(delta_y, delta_x)

    # mask = np.zeros_like(im1)
    # for i, (new, old) in enumerate(zip(good_new, good_old)):
    #     a, b = new.ravel()
    #     c, d = old.ravel()
    #     mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
    #     im2 = cv.circle(im2, (a, b), 1, color[i].tolist(), -1)
    # img = cv.add(im2, mask)
    # # print(p0)
    # # print(np.shape(p0))
    # cv.imshow('frame', img)
    # cv.waitKey()

    grid_shape = (h_num-2, w_num-2)
    grid_size = (h_num-2) * (w_num-2)
    theta = np.arctan2(delta_x, delta_y)
    # magnitude = np.sqrt(np.add(delta_y**2, delta_x**2))
    # magnitude = np.add(delta_y ** 2, delta_x ** 2)
    theta_2d = np.reshape(theta, grid_shape)
    theta_2d_filt = fltr.laplace(theta_2d)
    # # magnitude = np.sqrt(np.add(delta_y**2, delta_x**2))
    delta_x_2d = np.reshape(delta_x, grid_shape)
    delta_y_2d = np.reshape(delta_y, grid_shape)
    # du_grad_x = fltr.sobel(fltr.median_filter(delta_x_2d, 3), 0)
    # du_grad_y = fltr.sobel(fltr.median_filter(delta_x_2d, 3), 1)
    # dv_grad_x = fltr.sobel(fltr.median_filter(delta_y_2d, 3), 0)
    # dv_grad_y = fltr.sobel(fltr.median_filter(delta_y_2d, 3), 1)

    du_grad_x = fltr.sobel(delta_x_2d, 0)
    du_grad_y = fltr.sobel(delta_x_2d, 1)
    dv_grad_x = fltr.sobel(delta_y_2d, 0)
    dv_grad_y = fltr.sobel(delta_y_2d, 1)

    du_grad_x_1d = np.reshape(du_grad_x, grid_size)
    du_grad_y_1d = np.reshape(du_grad_y, grid_size)
    dv_grad_x_1d = np.reshape(dv_grad_x, grid_size)
    dv_grad_y_1d = np.reshape(dv_grad_y, grid_size)
    theta_filt = np.reshape(theta_2d_filt, grid_size)
    # # print(np.shape(theta_filt))
    # # print(np.shape(dv_grad_y_1d))
    motion = np.zeros(grid_size)
    # # print(np.shape(motion))
    motion[theta_filt > thresh_angle] = np.maximum(abs(du_grad_x_1d[theta_filt > thresh_angle]) + abs(dv_grad_x_1d[theta_filt > thresh_angle]),
                                              abs(du_grad_y_1d[theta_filt > thresh_angle]) + abs(dv_grad_y_1d[theta_filt > thresh_angle]))
    motion[motion < 15] = 0
    # print(min(p0[st != 1] - p1[st != 1]))
    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = p0[:, :, 1]
    # X = good_old[:, 0]
    # X = np.arange(1, 64)
    # X = np.reshape(X,(np.shape(X)[0], 1))
    # print(np.shape(X))
    # print(np.max(X))
    # Y = good_old[:, 1]
    Y = p0[:, :, 0]
    # Y = np.arange(1, 36)
    # Y = np.reshape(Y, (np.shape(Y)[0], 1))
    # print(np.shape(Y))
    # print(np.max(Y))
    # print(np.shape(magnitude))
    # z2 = theta_2d_filt

    cx = X[motion != 0]
    cy = Y[motion != 0]

    print(cx)
    # print(np.int32(np.shape(cx)[0]))
    for i in range(0, np.int32(np.shape(cx)[0])):
        cv.circle(im2, (cx[i], cy[i]), 5, (0, 0, 255), -1)
        img2 = cv.add(im2, mask)
    cv.imshow('frame2', img2)
    cv.waitKey()

    ax.scatter(X, Y, dv_grad_y_1d, c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # x_sort = np.sort(X, 0)
    np.set_printoptions(threshold=np.nan)
    print(motion)
    plt.show()
"""
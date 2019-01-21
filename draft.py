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
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# im1 = cv.imread('/home/mai/Downloads/scbudata/Walking/INPUT/0269.png')
# im1_gray = cv.cvtColor(im1, cv.COLOR_RGB2GRAY)
# im2 = cv.imread('/home/mai/Downloads/scbudata/Walking/INPUT/0270.png')
# im2_gray = cv.cvtColor(im2, cv.COLOR_RGB2GRAY)
# im1 = cv.imread('/home/mai/Pictures/vlcsnap-2018-11-09-11h35m50s841.jpg')
# im1_gray = cv.cvtColor(im1, cv.COLOR_RGB2GRAY)
# im2 = cv.imread('/home/mai/Pictures/vlcsnap-2018-11-09-11h35m59s446.jpg')
# im2_gray = cv.cvtColor(im2, cv.COLOR_RGB2GRAY)
im1 = cv.imread('/home/mai/Downloads/Dataset/BMS_dataset/videos/seq1/seq8.jpg')
im1_gray = cv.cvtColor(im1, cv.COLOR_RGB2GRAY)
im2 = cv.imread('/home/mai/Downloads/Dataset/BMS_dataset/videos/seq1/seq9.jpg')
im2_gray = cv.cvtColor(im2, cv.COLOR_RGB2GRAY)
start = timeit.default_timer()

print(np.shape(im2_gray))
h = np.shape(im1_gray)[0]
w = np.shape(im1_gray)[1]
print(w)
step = 10
margin = 2
thresh_angle = 0
motion_threshold = 10
num_grid = int(h*w/(step**2))


grid_test = True
high_step = 10
width_step = 10
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
color = np.random.randint(0, 255, (200000, 3))

in_loop = False
in_loop_once = True
ctr = 0
ctr2 = 0
if grid_test:
    mask_use = np.zeros(im1_gray.shape, np.uint8)
    for m in range(margin, h_num - margin):
        for n in range(margin, w_num - margin):
            p0_temp = [[[(n - 1) * width_step + half_step, (m - 1) * high_step + half_step]]]
            if in_loop_once:
                p0 = p0_temp
            else:
                p0 = np.concatenate((p0, p0_temp))
            in_loop_once = False

            # mask_use = np.zeros(im1_gray.shape, np.uint8)
            # mask_use[(m-1)*high_step:m*high_step, (n-1)*width_step:n*width_step] =\
            #     im1_gray[(m-1)*high_step:m*high_step, (n-1)*width_step:n*width_step]
            # # self.fast = cv.FastFeatureDetector_create(1)
            # p0_temp = cv.goodFeaturesToTrack(im1_gray, mask=mask_use, **feature_params)
            # if in_loop_once:
            #     p0 = p0_temp
            # elif np.size(p0_temp) > 1:
            #     p0 = np.concatenate((p0, p0_temp))
            #     ctr2 = ctr2 + 1
            # elif np.size(p0_temp) == 1:
            #     p0_temp = [[[(n-1)*width_step + half_step, (m-1)*high_step + half_step]]]
            #     p0 = np.concatenate((p0, p0_temp))
            # in_loop_once = False
            ctr = ctr + 1

    # print(np.shape(p0))
        # p0_temp = cv.goodFeaturesToTrack(im1_gray, mask=mask_use, **feature_params)

    # print(np.shape(p0))
    # print(w_num)
    # print(h_num)
    p0 = np.float32(p0)
    mask = np.zeros_like(im1)
    p1, st, err = cv.calcOpticalFlowPyrLK(im1_gray, im2_gray, p0, None, **lk_params)
    # print('w', w_num)
    # print('h', h_num)
    #
    # print('ctr', ctr)
    # print('ctr2', ctr2)
    # # diff = abs(p0[st != 1] - p1[st != 1])
    # print(np.shape(p0))
    # print(np.shape(p1))
    # print('p0_max = ', max(p0[:, :, 0]))
    # print('p0_min = ', max(p0[:, :, 1]))
    # print('p1_max = ', max(p1[:, :, 0]))
    # print('p1_min = ', max(p1[:, :, 1]))
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
    # src_pts = np.float32(p0)
    # dst_pts = np.float32(p1)
    # M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    # print('M', M)

    # # test Homography warping
    # result = cv.warpPerspective(im1_gray, M, (im1_gray.shape[1], im1_gray.shape[0]))
    # # result[0:img1.shape[0], 0:img1.shape[1]] = img1
    # # plt.imshow(result,'gray'),plt.show()
    #
    # a = cv.absdiff(im2_gray, im1_gray)
    # b = cv.absdiff(im2_gray, result)
    # fig = plt.figure()
    # columns = 2
    # rows = 1
    # th, im_th = cv.threshold(b, 50, 255, cv.THRESH_BINARY_INV)
    # # plt.imshow(255-im_th, 'gray')
    # # plt.show()
    #
    # # Copy the thresholded image.
    # im_floodfill = im_th.copy()
    # im_floodfill_inv = cv.bitwise_not(im_floodfill)
    # # Display images.
    # # cv.imshow("Thresholded Image", im_th)
    # # cv.imshow("Floodfilled Image", im_floodfill)
    # # cv.imshow("Inverted Floodfilled Image", im_floodfill_inv)
    # # cv.imshow("Foreground", im_out)
    # # cv.waitKey(0)
    # kernel = np.ones((5, 5), np.uint8)
    # mask_skin = cv.morphologyEx(im_floodfill_inv, cv.MORPH_OPEN, kernel)
    #
    # stop = timeit.default_timer()
    # print('Time: ', stop - start)
    # fig.add_subplot(rows, columns, 1)
    # plt.imshow(a, 'gray')
    #
    # fig.add_subplot(rows, columns, 2)
    # plt.imshow(b, 'gray')
    # plt.show()
    # cv.imshow("mask_skin", mask_skin)
    # cv.waitKey(0)
    # # end test Homography warping


    delta_x = p1[:, :, 0] - p0[:, :, 0]
    delta_y = p1[:, :, 1] - p0[:, :, 1]
    # theta = np.arctan2(delta_y, delta_x)

    mask = np.zeros_like(im1)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        im2 = cv.circle(im2, (a, b), 1, color[i].tolist(), -1)
    img = cv.add(im2, mask)
    cv.imshow('frame', img)
    cv.waitKey()

    grid_shape = (h_num-(2*margin), w_num-(2*margin))
    grid_size = (h_num-(2*margin)) * (w_num-(2*margin))
    theta = np.arctan2(delta_x, delta_y)
    magnitude = np.sqrt(np.add(delta_y**2, delta_x**2))
    # magnitude = np.add(delta_y ** 2, delta_x ** 2)
    theta_2d = np.reshape(theta, grid_shape)
    theta_2d_filt = fltr.laplace(theta_2d)
    # # magnitude = np.sqrt(np.add(delta_y**2, delta_x**2))
    #
    delta_x_2d = np.reshape(delta_x, grid_shape)
    delta_y_2d = np.reshape(delta_y, grid_shape)
    #
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
    motion = np.zeros((grid_size, 1))
    # P_motion = np.zeros((grid_size, 1, 2))
    # P_motion_y = np.zeros(grid_size)

    # # print(np.shape(motion))
    motion[theta_filt > thresh_angle, 0] = np.maximum(abs(du_grad_x_1d[theta_filt > thresh_angle]) + abs(dv_grad_x_1d[theta_filt > thresh_angle]),
                                              abs(du_grad_y_1d[theta_filt > thresh_angle]) + abs(dv_grad_y_1d[theta_filt > thresh_angle]))

    motion[motion < motion_threshold] = 0
    motion[motion >= motion_threshold] = 255
    # print(min(p0[st != 1] - p1[st != 1]))
    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = p0[:, :, 0]
    # X = good_old[:, 0]
    # X = np.arange(1, 64)
    # X = np.reshape(X,(np.shape(X)[0], 1))
    # print(np.shape(X))
    # print(np.max(X))
    # Y = good_old[:, 1]
    Y = p0[:, :, 1]
    # Y = np.arange(1, 36)
    # Y = np.reshape(Y, (np.shape(Y)[0], 1))
    # print(np.shape(Y))
    # print(np.max(Y))
    # print(np.shape(magnitude))
    # z2 = theta_2d_filt
    cx = X[motion != 0]
    cy = Y[motion != 0]

    print(cx)
    print(np.int32(np.shape(cx)[0]))
    for i in range(0, np.int32(np.shape(cx)[0])):
        cv.circle(im1, (cx[i], cy[i]), 5, (0, 0, 255), -1)
        img2 = cv.add(im1, mask)
    cv.imshow('frame2', img2)
    cv.waitKey()
    ax.scatter(X, Y, motion, c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # x_sort = np.sort(X, 0)
    # np.set_printoptions(threshold=np.nan)
    # print(motion)
    plt.show()

    # Threshold.
    # Set values equal to or above 220 to 0.
    # Set values below 220 to 255.
    motion_2d = np.reshape(motion, grid_shape)
    motion_2d = np.uint8(motion_2d)
    th, im_th = cv.threshold(motion_2d, 125, 255, cv.THRESH_BINARY_INV);

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # # Mask used to flood filling.
    # # Notice the size needs to be 2 pixels than the image.
    # h, w = im_th.shape[:2]
    # mask = np.zeros((h + 2, w + 2), np.uint8)
    #
    # print(type(im_floodfill))
    # im_floodfill = np.uint8(im_floodfill)
    # # Floodfill from point (0, 0)
    # cv.floodFill(im_floodfill, mask, (0, 0), 255);
    #
    # Invert floodfilled image
    im_floodfill_inv = cv.bitwise_not(im_floodfill)
    #
    # # Combine the two images to get the foreground.
    # im_out = im_th | im_floodfill_inv

    # Display images.
    cv.imshow("Thresholded Image", im_th)
    # cv.imshow("Floodfilled Image", im_floodfill)
    cv.imshow("Inverted Floodfilled Image", im_floodfill_inv)
    # cv.imshow("Foreground", im_out)
    cv.waitKey(0)

    kernel = np.ones((2, 2), np.uint8)
    mask_skin = cv.morphologyEx(im_floodfill_inv, cv.MORPH_CLOSE, kernel)
    # kernel = np.ones((3, 3), np.uint8)
    # mask_skin = cv.morphologyEx(mask_skin, cv.MORPH_OPEN, kernel)
    cv.imshow("mask_skin", mask_skin)
    cv.waitKey(0)

    im_out = im_floodfill_inv ^ mask_skin
    cv.imshow("Foreground", im_out)
    cv.waitKey(0)

    motion_1d = np.reshape(mask_skin, grid_size)
    cx = X[motion_1d != 0]
    cy = Y[motion_1d != 0]

    print(cx)
    print(np.int32(np.shape(cx)[0]))
    for i in range(0, np.int32(np.shape(cx)[0])):
        cv.circle(im2, (cx[i], cy[i]), 5, (0, 255, 0), -1)
        img2 = cv.add(im2, mask)
    cv.imshow('frame2', img2)
    cv.waitKey()

# magnitude_2d = np.reshape(magnitude, grid_shape)
# # # np.set_printoptions(threshold=np.nan)
# # img = cv.imread('/home/mai/Pictures/vlcsnap-2018-12-03-15h48m49s592.jpg')
# # im1_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
# # print(im1_gray)
# # Z = im1_gray.reshape((-1, 3))
# Z = magnitude_2d
# # convert to np.float32
# # Z = np.float32(Z)
# # print(Z)
# # define criteria, number of clusters(K) and apply kmeans()
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# K = 2
# ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
# # Now convert back into uint8, and make original image
# # center = np.uint8(center)
# res = center[label.flatten()]
# res2 = res.reshape(theta_2d.shape)
# print(res)
# ax.scatter(X, res, c='r', marker='o')
# plt.show()
#
# # cv.imshow('res2', res2)
# # cv.waitKey(0)
# # cv.destroyAllWindows()
# plt.scatter(theta, magnitude, c='r', marker='o')

# # clustering
#     points = np.zeros((np.shape(magnitude)[0], 5))
#     points[:, 0] = magnitude[:, 0]
#     points[:, 1] = theta[:, 0]
#     # points[:, 2] = motion[:, 0]  # theta[:, 0]
#     # points[:, 3] = Y[:, 0]
#
#     print(np.shape(points))
#     points = np.float32(points)
#
#     criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#     ret, label, center = cv.kmeans(points, 4, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
#
#     # Now separate the data, Note the flatten()
#     A = points[label.ravel() == 0]
#     B = points[label.ravel() == 1]
#     C = points[label.ravel() == 2]
#     D = points[label.ravel() == 3]
#
#
#     # Plot the data
#     plt.scatter(A[:, 0], A[:, 1])
#     plt.scatter(B[:, 0], B[:, 1], c='r')
#     plt.scatter(C[:, 0], C[:, 1])
#     plt.scatter(D[:, 0], D[:, 1])
#     plt.scatter(center[:, 0], center[:, 1], s=80, c='y', marker='s')
#     plt.xlabel('Height'), plt.ylabel('Weight')
#     plt.show()
#
#     cx0 = X[label.ravel() == 0]
#     cy0 = Y[label.ravel() == 0]
#     cx1 = X[label.ravel() == 1]
#     cy1 = Y[label.ravel() == 1]
#     cx2 = X[label.ravel() == 2]
#     cy2 = Y[label.ravel() == 2]
#     cx3 = X[label.ravel() == 3]
#     cy3 = Y[label.ravel() == 3]
#
#     for i in range(0, np.int32(np.shape(cx0)[0])):
#         imm = cv.circle(im1, (cx0[i], cy0[i]), 4, (0, 0, 255), -1)
#         img2 = cv.add(imm, mask)
#     for i in range(0, np.int32(np.shape(cx1)[0])):
#         imm = cv.circle(im1, (cx1[i], cy1[i]), 4, (0, 255, 0), -1)
#         img2 = cv.add(imm, mask)
#     for i in range(0, np.int32(np.shape(cx2)[0])):
#         imm = cv.circle(im1, (cx2[i], cy2[i]), 4, (255, 0, 0), -1)
#         img2 = cv.add(imm, mask)
#     for i in range(0, np.int32(np.shape(cx3)[0])):
#         imm = cv.circle(im1, (cx3[i], cy3[i]), 4, (255, 255, 255), -1)
#         img2 = cv.add(imm, mask)
#     cv.imshow('frame2', img2)
#     cv.waitKey()
#
"""
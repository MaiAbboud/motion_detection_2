
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
import scipy.ndimage.filters as fltr
import random
import timeit
# my_url = '/home/mai/Videos/Domingo'
my_url = '/home/mai/Downloads/Dataset/BMS_dataset/videos/cars5'
my_url2 = '/home/mai/Downloads/Dataset/BMS_dataset/algorithm2/cars5/'
cap = cv.VideoCapture(my_url+'.avi')
# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")
im1 = cap.read()[1]
# cv.imshow('Frame', im1)
# cv.waitKey()
im1_gray = cv.cvtColor(im1, cv.COLOR_RGB2GRAY)
im2 = cap.read()[1]
im2_gray = cv.cvtColor(im2, cv.COLOR_RGB2GRAY)

h = np.shape(im1_gray)[0]
w = np.shape(im1_gray)[1]

step = 20
margin = 3
thresh_angle = 5
motion_threshold = 15
# margin for threshold
margin_x = np.int(w/20)#np.int8(10*abs(M[0, 2]))
margin_y = np.int(h/20)#np.int8(10*abs(M[1, 2]))

in_loop_once = True
monitor = False
record = True
ctr = 0
# algorithm 1 good features to track + optical flow + RANSAC
# algorithm 2 features on grid + optical flow + RANSAC
# algorithm 3 features on grid + optical flow + gardient
# algorithm 4 ORB + RANSAC
algorithm = 2
# parameters of algorithm 1
feature_params = dict(maxCorners=1000,
                      qualityLevel=0.01,
                      minDistance=1,
                      blockSize=7)

lk_params = dict(winSize=(30, 30),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.03))
color = np.random.randint(0, 255, (10000, 3))
file = open(my_url2+'parameters_time.txt', 'w')
file.write('feature_params = %s\n' % feature_params)
file.write('lk_params = %s\n' % lk_params)

while cap.isOpened():
    im1 = im2  # .copy()
    im1_gray = im2_gray  # .copy()

    ret, im2 = cap.read()
    if np.shape(im2) == ():
        break
# for ctr_img in range(2, 74):
    # im2.copyTo(im1)

    # im2_gray.copyTo(im1_gray)

    # im1 = cv.imread('/home/mai/Pictures/vlcsnap-2018-12-03-17h39m37s720.jpg')
    # im1_gray = cv.cvtColor(im1, cv.COLOR_RGB2GRAY)
    # im2 = cv.imread('/home/mai/Downloads/Dataset/BMS_dataset/videos/seq1/seq%s.jpg' % ctr_img)
    im2_gray = cv.cvtColor(im2, cv.COLOR_RGB2GRAY)

    start = timeit.default_timer()
    if algorithm == 1:
        p0 = cv.goodFeaturesToTrack(im1_gray, mask=None, **feature_params)
        mask = np.zeros_like(im1)
        p1, st, err = cv.calcOpticalFlowPyrLK(im1_gray, im2_gray, p0, None, **lk_params)
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        src_pts = np.float32(p0)
        dst_pts = np.float32(p1)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        result = cv.warpPerspective(im1_gray, M, (im1_gray.shape[1], im1_gray.shape[0]))
        a = cv.absdiff(im2_gray, im1_gray)
        b = cv.absdiff(im2_gray, result)
        b[im1_gray.shape[0] - margin_x:, :] = 0
        b[:, im1_gray.shape[1] - margin_y:] = 0
        b[:margin_x, :] = 0
        b[:, :margin_y] = 0
        th, im_th = cv.threshold(b, 50, 255, cv.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        mask_skin = cv.morphologyEx(im_th, cv.MORPH_CLOSE, kernel, iterations=3)

    if algorithm == 2:
        # print(h, w)
        y, x = np.mgrid[margin*step:h-(margin*step-1):step, margin*step:w-(margin*step-1):step]
        pair = np.vstack((x.flatten(), y.flatten())).T
        p0 = np.reshape(pair, (np.shape(pair)[0], 1, 2))
        p0 = np.float32(p0)

        mask = np.zeros_like(im1)
        p1, st, err = cv.calcOpticalFlowPyrLK(im1_gray, im2_gray, p0, None, **lk_params)
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        src_pts = np.float32(p0)
        dst_pts = np.float32(p1)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        # margin_x = np.int8(abs(M[0, 2]))
        # margin_y = np.int8(abs(M[1, 2]))
        # print(M)
        result = cv.warpPerspective(im1_gray, M, (im1_gray.shape[1], im1_gray.shape[0]))
        a = cv.absdiff(im2_gray, im1_gray)
        b = cv.absdiff(im2_gray, result)
        b[im1_gray.shape[0]-margin_x:, :] = 0
        b[:, im1_gray.shape[1]-margin_y:] = 0
        b[:margin_x, :] = 0
        b[:, :margin_y] = 0
        th, im_th = cv.threshold(b, 50, 255, cv.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        mask_skin = cv.morphologyEx(im_th, cv.MORPH_CLOSE, kernel, iterations = 3)


    if algorithm == 3:
        p0 = [[[0, 0]]]
        # mask_use = np.zeros(im1_gray.shape, np.uint8)
        # print('mask_use.shape', mask_use.shape)
        # print('margin', margin)
        # for m in range(margin, h_num - margin):
        #     for n in range(margin, w_num - margin):
        #         p0_temp = [[[(n - 1) * width_step + half_step, (m - 1) * high_step + half_step]]]
        #         if in_loop_once:
        #             p0 = p0_temp
        #         else:
        #             p0 = np.concatenate((p0, p0_temp))
        #         in_loop_once = False
        # print('p0_shape',np.shape(p0))
        # p0 = np.float32(p0)
        # margin = 0
        # y, x = np.mgrid[0:h:step, 0:w:step]
        y, x = np.mgrid[margin*step:h-(margin*step-1):step, margin*step:w-(margin*step-1):step]
        print('shape_x', np.shape(x))
        print('shape_y', np.shape(y))
        pair = np.vstack((x.flatten(), y.flatten())).T
        p0 = np.reshape(pair, (np.shape(pair)[0], 1, 2))
        p0 = np.float32(p0)

        mask = np.zeros_like(im1)
        p1, st, err = cv.calcOpticalFlowPyrLK(im1_gray, im2_gray, p0, None, **lk_params)
        good_new = p1
        good_old = p0
        delta_x = p1[:, :, 0] - p0[:, :, 0]
        delta_y = p1[:, :, 1] - p0[:, :, 1]
        # if monitor:
        #     mask = np.zeros_like(im1)
        #     for i, (new, old) in enumerate(zip(good_new, good_old)):
        #         a, b = new.ravel()
        #         c, d = old.ravel()
        #         mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        #         im2 = cv.circle(im2, (a, b), 1, color[i].tolist(), -1)
        #     img = cv.add(im2, mask)
        #     cv.imshow('frame', img)
        #     cv.waitKey()

        # grid_shape = (h_num-2*margin+1, w_num-2*margin+1)
        # grid_size = (h_num-2*margin+1)*(w_num-2*margin+1)
        grid_shape = np.shape(x)
        grid_size = np.size(x)
        theta = np.arctan2(delta_x, delta_y)
        magnitude = np.sqrt(np.add(delta_y**2, delta_x**2))
        theta_2d = np.reshape(theta, grid_shape)
        theta_2d_filt = fltr.laplace(theta_2d)
        delta_x_2d = np.reshape(delta_x, grid_shape)
        delta_y_2d = np.reshape(delta_y, grid_shape)

        du_grad_x = fltr.sobel(delta_x_2d, 0)
        du_grad_y = fltr.sobel(delta_x_2d, 1)
        dv_grad_x = fltr.sobel(delta_y_2d, 0)
        dv_grad_y = fltr.sobel(delta_y_2d, 1)

        du_grad_x_1d = np.reshape(du_grad_x, grid_size)
        du_grad_y_1d = np.reshape(du_grad_y, grid_size)
        dv_grad_x_1d = np.reshape(dv_grad_x, grid_size)
        dv_grad_y_1d = np.reshape(dv_grad_y, grid_size)
        theta_filt = np.reshape(theta_2d_filt, grid_size)

        motion = np.zeros((grid_size, 1))
        motion[theta_filt > thresh_angle, 0] = np.maximum(abs(du_grad_x_1d[theta_filt > thresh_angle]) + abs(dv_grad_x_1d[theta_filt > thresh_angle]),
                                                  abs(du_grad_y_1d[theta_filt > thresh_angle]) + abs(dv_grad_y_1d[theta_filt > thresh_angle]))

        motion[motion < motion_threshold] = 0
        motion[motion >= motion_threshold] = 255
        # 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X = p0[:, :, 0]
        Y = p0[:, :, 1]
        cx = X[motion != 0]
        cy = Y[motion != 0]

        # print(cx)
        # print(np.int32(np.shape(cx)[0]))
        # if monitor:
        #     for i in range(0, np.int32(np.shape(cx)[0])):
        #         cv.circle(im1, (cx[i], cy[i]), 5, (0, 0, 255), -1)
        #         img2 = cv.add(im1, mask)
        #     cv.imshow('frame2', img2)
        #     cv.waitKey()
        #     ax.scatter(X, Y, motion, c='r', marker='o')
        #     ax.set_xlabel('X Label')
        #     ax.set_ylabel('Y Label')
        #     ax.set_zlabel('Z Label')
        #     plt.show()

        # Threshold.
        motion_2d = np.reshape(motion, grid_shape)
        motion_2d = np.uint8(motion_2d)
        th, im_th = cv.threshold(motion_2d, 125, 255, cv.THRESH_BINARY_INV);

        # Copy the thresholded image.
        im_floodfill = im_th.copy()
        im_floodfill_inv = cv.bitwise_not(im_floodfill)

        # if monitor:
        #     # Display images.
        #     cv.imshow("Thresholded Image", im_th)
        #     cv.imshow("Inverted Floodfilled Image", im_floodfill_inv)
        #     cv.waitKey(0)

        kernel = np.ones((2, 2), np.uint8)
        mask_skin = cv.morphologyEx(im_floodfill_inv, cv.MORPH_CLOSE, kernel)
        # if monitor:
        #     cv.imshow("mask_skin", mask_skin)
        #     cv.waitKey(0)

        im_out = im_floodfill_inv ^ mask_skin
        # if monitor:
        #     cv.imshow("Foreground", im_out)
        #     cv.waitKey(40)

        motion_1d = np.reshape(mask_skin, grid_size)
        cx = X[motion_1d != 0]
        cy = Y[motion_1d != 0]

        # print(cx)
        # print(np.int32(np.shape(cx)[0]))
        if monitor:
            for i in range(0, np.int32(np.shape(cx)[0])):
                cv.circle(im2, (cx[i], cy[i]), 6, (0, 255, 0), -1)
                img2 = cv.add(im2, mask)
            cv.imshow('frame2', im2)
            cv.waitKey(1)

    # if grid:
    #     good_new = p1
    #     good_old = p0
    #     delta_x = p1[:, :, 0] - p0[:, :, 0]
    #     delta_y = p1[:, :, 1] - p0[:, :, 1]
    # else:
    if algorithm == 4:
        MIN_MATCH_COUNT = 10
        orb = cv.ORB_create()
        # find the keypoints with ORB
        kp1 = orb.detect(im1, None)
        # compute the descriptors with ORB
        kp1, des1 = orb.compute(im1, kp1)
        # find the keypoints with ORB
        kp2 = orb.detect(im2, None)
        # compute the descriptors with ORB
        kp2, des2 = orb.compute(im2, kp2)
        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        good = []
        for m in matches:
            # if m.distance < 0.7:
            good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            # margin_x = 10#np.int8(np.pi*2*abs(M[0, 2]))
            # margin_y = 10#np.int8(np.pi*2*abs(M[1, 2]))
            # print(np.shape(src_pts))
            # print('H', M[0, 0])
            matchesMask = mask.ravel().tolist()
            # h,w,d = img1.shape
            # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            # dst = cv.perspectiveTransform(pts,M)
            # img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
        else:
            print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
            matchesMask = None

        # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
        #                    singlePointColor = None,
        #                    matchesMask = matchesMask, # draw only inliers
        #                    flags = 2)
        # img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        # plt.imshow(img3, 'gray'),plt.show()
        # print(im1.shape)
        rows = im1.shape[0]
        cols = im1.shape[1]
        # result = img2
        result = cv.warpPerspective(im1_gray, M, (im1.shape[1], im1.shape[0]))
        # result[0:img1.shape[0], 0:img1.shape[1]] = img1
        # plt.imshow(result,'gray'),plt.show()

        a = cv.absdiff(im2_gray, im1_gray)
        b = cv.absdiff(im2_gray, result)
        b[im1_gray.shape[0] - margin_x:, :] = 0
        b[:, im1_gray.shape[1] - margin_y:] = 0
        b[:margin_x, :] = 0
        b[:, :margin_y] = 0
        th, im_th = cv.threshold(b, 50, 255, cv.THRESH_BINARY)

        # stop = timeit.default_timer()
        # print('Time: ', stop - start)
        if monitor:
            # fig = plt.figure()
            # columns = 2
            # rows = 1
            # fig.add_subplot(rows, columns, 1)
            # plt.imshow(a, 'gray')
            #
            # fig.add_subplot(rows, columns, 2)
            # plt.imshow(b, 'gray')
            # plt.show()
            cv.imshow('im2', im2)
            cv.waitKey(20)

            cv.imshow('im_th', im_th)
            cv.waitKey(20)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    time = stop - start
    if (monitor & (algorithm != 4)):

        mask = np.zeros_like(im1)
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            im2 = cv.circle(im2, (a, b), 1, color[i].tolist(), -1)
        img = cv.add(im2, mask)
        cv.imshow('frame', img)
        cv.waitKey(1)
        cv.imshow("th", mask_skin)
        cv.waitKey(12)
    ctr = ctr + 1
    if record:
        cv.imwrite(my_url2+'%s.png' % ctr, mask_skin)
        file.write('%s\n' % time)

file.close()

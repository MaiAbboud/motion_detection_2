"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
import scipy.ndimage.filters as fltr
import random
im1 = cv.imread('/home/mai/Pictures/vlcsnap-2018-11-16-15h10m53s032.jpg')
im1_gray = cv.cvtColor(im1, cv.COLOR_RGB2GRAY)
im2 = cv.imread('/home/mai/Pictures/vlcsnap-2018-11-16-15h10m56s574.jpg')
im2_gray = cv.cvtColor(im2, cv.COLOR_RGB2GRAY)

h = np.shape(im1_gray)[0]
w = np.shape(im1_gray)[1]

print('h', h)
print('w', w)

step = 10
margin = 10
thresh_angle = 0.4
num_grid = int(h*w/(step**2))
print('num_grid', num_grid)
all_features = True
feature_params = dict(maxCorners=100000,
                      qualityLevel=0.01,
                      minDistance=1,
                      blockSize=7)

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
#change 10000 number
color = np.random.randint(0, 255, (10000, 3))

# change here and test another detectors --canny --
ctr = 0
# p0 = cv.goodFeaturesToTrack(im1_gray, mask=None, **feature_params)

if all_features:
    # print(h, w)
    y, x = np.mgrid[margin+step:h-margin:step, margin+step:w-margin:step]
    print('x', np.shape(x))
    print('y', np.shape(y))
    # print(x)
    pair = np.vstack((x.flatten(), y.flatten())).T
    # print(np.shape(pair)[0])
    p0 = np.reshape(pair, (np.shape(pair)[0], 1, 2))
    p0 = np.float32(p0)
    # np.random.shuffle(p0)
    # print(p0.dtype)
    # print(np.shape(p0))
    # print(p0)
# else:
    # A = [random.randint(0, w) for p in range(0, 100)]
    # B = [random.randint(0, h) for p in range(0, 100)]
    # C = np.transpose([np.tile(A, len(B)), np.repeat(B, len(A))])
    # p0 = np.reshape(C, (np.shape(C)[0], 1, 2))
    # p0 = np.float32(p0)
    # print(p0.dtype)
    # print(np.shape(p0))
    # print(p0)
# print('ctr', np.shape(p0))
# print('max p0 0', max(p0[:, :, 0]))
# print('max p0 1', max(p0[:, :, 1]))
mask = np.zeros_like(im1)
p1, st, err = cv.calcOpticalFlowPyrLK(im1_gray, im2_gray, p0, None, **lk_params)
if all_features:
    good_new = p1
    good_old = p0
    delta_x = p1[:, :, 0] - p0[:, :, 0]
    delta_y = p1[:, :, 1] - p0[:, :, 1]
else:
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    delta_x = good_new[:, 0] - good_old[y:, 0]
    delta_y = good_new[:, 1] - good_old[:, 1]

theta = np.arctan2(delta_y, delta_x)
magnitude = np.sqrt(np.add(delta_y**2, delta_x**2))

mask = np.zeros_like(im1)
for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
    im2 = cv.circle(im2, (a, b), 1, color[i].tolist(), -1)
img = cv.add(im2, mask)
cv.imshow('frame', img)
cv.waitKey()
src_pts = np.float32(p0)
dst_pts = np.float32(p1)
M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
print(M)
result = cv.warpPerspective(im1_gray, M, (im1_gray.shape[1], im1_gray.shape[0]))
# test Homography warping
im2_mean = im2_gray.mean()
im1_mean = im1_gray.mean()
res_mean = result.mean()
# a = cv.absdiff(im2_gray-im2_mean, im1_gray-im1_mean)
# b = cv.absdiff(im2_gray-im2_mean, result-res_mean)
a = cv.absdiff(im2_gray, im1_gray)
b = cv.absdiff(im2_gray, result)
# fig = plt.figure()
columns = 2
rows = 1
th, im_th = cv.threshold(b, 10, 255, cv.THRESH_BINARY)
im_th_filtered = fltr.median_filter(im_th, (5, 5))
kernel = np.ones((10, 10), np.uint8)
mask_skin = cv.morphologyEx(im_th_filtered, cv.MORPH_CLOSE, kernel, iterations=4)

cv.imshow("th", a)
cv.waitKey()
cv.imshow("th_filtered", b)
cv.waitKey()

th, im_th = cv.threshold(b, 50, 255, cv.THRESH_BINARY)
im_th_filtered = fltr.median_filter(im_th, (5, 5))
kernel = np.ones((10, 10), np.uint8)
mask_skin = cv.morphologyEx(im_th_filtered, cv.MORPH_CLOSE, kernel, iterations=4)
# mask_skin = cv.morphologyEx(im_th, cv.MORPH_DILATE, kernel, iterations=2)

cv.imshow("th", im_th)
cv.waitKey()
cv.imshow("th_filtered", im_th_filtered)
cv.waitKey()
"""
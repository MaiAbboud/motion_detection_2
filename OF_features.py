"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import timeit

from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
import scipy.ndimage.filters as fltr
im1 = cv.imread('/home/mai/Downloads/Dataset/BMS_dataset/videos/seq1/seq33.jpg')
im1_gray = cv.cvtColor(im1, cv.COLOR_RGB2GRAY)
im2 = cv.imread('/home/mai/Downloads/Dataset/BMS_dataset/videos/seq1/seq34.jpg')
im2_gray = cv.cvtColor(im2, cv.COLOR_RGB2GRAY)
start = timeit.default_timer()
h = np.shape(im1_gray)[0]
w = np.shape(im1_gray)[1]
step = 10
margin = 10
num_grid = int(h*w/(step**2))
print(num_grid)
all_features = True
random_pts = True
feature_params = dict(maxCorners=500,
                      qualityLevel=0.01,
                      minDistance=1,
                      blockSize=7)

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0, 255, (num_grid, 3))

# change here and test another detectors --canny --
# if all_features:
#     # print(h, w)
#     x, y = np.mgrid[margin:w-margin:step, margin:h-margin:step]
#     pair = np.vstack((x.flatten(), y.flatten())).T
#     print(np.shape(pair)[0])
#     p0 = np.reshape(pair, (np.shape(pair)[0], 1, 2))
#     p0 = np.float32(p0)
# else:
# if random_pts:
#
# else:

p0 = cv.goodFeaturesToTrack(im1_gray, mask=None, **feature_params)
mask = np.zeros_like(im1)
p1, st, err = cv.calcOpticalFlowPyrLK(im1_gray, im2_gray, p0, None, **lk_params)

good_new = p1[st == 1]
good_old = p0[st == 1]
X = good_old[:, 0]
Y = good_old[:, 1]

delta_x = good_new[:, 0]-good_old[:, 0]
delta_y = good_new[:, 1]-good_old[:, 1]
theta = np.arctan2(delta_y, delta_x)
magnitude = np.sqrt(np.add(delta_y**2, delta_x**2))

#Draw OF
mask = np.zeros_like(im1)
for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
    im2 = cv.circle(im2, (a, b), 1, color[i].tolist(), -1)
img = cv.add(im2, mask)
cv.imshow('frame', img)
# cv.waitKey()



src_pts = np.float32(p0)
dst_pts = np.float32(p1)
M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
print('M', M)

result = cv.warpPerspective(im1_gray, M, (im1_gray.shape[1], im1_gray.shape[0]))
# result[0:img1.shape[0], 0:img1.shape[1]] = img1
# plt.imshow(result,'gray'),plt.show()

# test Homography warping
a = cv.absdiff(im2_gray, im1_gray)
b = cv.absdiff(im2_gray, result)
fig = plt.figure()
columns = 2
rows = 1
th, im_th = cv.threshold(b, 40, 255, cv.THRESH_BINARY_INV)
# plt.imshow(255-im_th, 'gray')
# plt.show()

# Copy the thresholded image.
im_floodfill = im_th.copy()
im_floodfill_inv = cv.bitwise_not(im_floodfill)
# Display images.
# cv.imshow("Thresholded Image", im_th)
# cv.imshow("Floodfilled Image", im_floodfill)
cv.imshow("Inverted Floodfilled Image", im_floodfill_inv)
# cv.imshow("Foreground", im_out)
# cv.waitKey(0)
kernel = np.ones((3, 3), np.uint8)
mask_skin = cv.morphologyEx(im_floodfill_inv, cv.MORPH_OPEN, kernel)
#
stop = timeit.default_timer()
print('Time: ', stop - start)
# fig.add_subplot(rows, columns, 1)
# plt.imshow(a, 'gray')
#
# fig.add_subplot(rows, columns, 2)
# plt.imshow(b, 'gray')
# plt.show()
cv.imshow("mask_skin", mask_skin)
cv.waitKey(0)

# theta2 = fltr.laplace(fltr.median_filter(theta, 5))
# magnitude2 = fltr.median_filter(magnitude, 5)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x = good_new[:, 1]
# y = good_new[:, 0]
# z = magnitude
# z2 = theta2
# ax.scatter(x, y, z2, c='r', marker='o')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# plt.show()
"""

"""
# clustering
points = np.zeros((np.shape(magnitude)[0], 5))
points[:, 0] = delta_y
points[:, 1] = delta_x
points[:, 2] = good_old[:, 0]
points[:, 3] = good_old[:, 1]
points[:, 4] = im1_gray[np.int32(good_old[:, 1]), np.int32(good_old[:, 0])]
print(np.shape(points))
points = np.float32(points)

# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# ret, label, center = cv.kmeans(points, 3, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
#
# # Now separate the data, Note the flatten()
# A = points[label.ravel() == 0]
# B = points[label.ravel() == 1]
# C = points[label.ravel() == 2]
# # D = points[label.ravel() == 3]
#
#
# # Plot the data
# plt.scatter(A[:, 0], A[:, 1], c='r')
# plt.scatter(B[:, 0], B[:, 1], c='g')
# plt.scatter(C[:, 0], C[:, 1], c='b')
# # plt.scatter(C[:, 0], C[:, 1])
# # plt.scatter(D[:, 0], D[:, 1])
# plt.scatter(center[:, 0], center[:, 1], s=80, c='y', marker='s')
# plt.xlabel('Height'), plt.ylabel('Weight')
# plt.show()
#
# cx0 = X[label.ravel() == 0]
# cy0 = Y[label.ravel() == 0]
#
# cx1 = X[label.ravel() == 1]
# cy1 = Y[label.ravel() == 1]
#
# cx2 = X[label.ravel() == 2]
# cy2 = Y[label.ravel() == 2]
# for i in range(0, np.int32(np.shape(cx0)[0])):
#     cv.circle(im2, (cx0[i], cy0[i]), 4, (0, 0, 255), -1)
#     img3 = cv.add(im2, mask)
# for i in range(0, np.int32(np.shape(cx1)[0])):
#     cv.circle(im2, (cx1[i], cy1[i]), 4, (0, 255, 0), -1)
#     img3 = cv.add(im2, mask)
# for i in range(0, np.int32(np.shape(cx2)[0])):
#     cv.circle(im1, (cx2[i], cy2[i]), 4, (255, 0, 0), -1)
#     img2 = cv.add(im1, mask)
# cv.imshow('frame3', img3)
# cv.waitKey()


# ================================
# test other methods of clustering
# ================================

print(__doc__)

import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

np.random.seed(0)

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
# print(noisy_circles)
# print(np.shape(noisy_circles))
# print(noisy_circles)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)

# ============
# Set up cluster parameters
# ============
plt.figure(figsize=(9 * 2 + 3, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1

default_base = {'quantile': .3,
                'eps': .3,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': 5}

datasets = [
    # (noisy_circles, {'damping': .77, 'preference': -240,
    #                  'quantile': .2, 'n_clusters': 2}),
    # (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
    # (varied, {'eps': .18, 'n_neighbors': 2}),
    # (aniso, {'eps': .15, 'n_neighbors': 2}),
    # (blobs, {}),
    (no_structure, {})]

for i_dataset, (dataset,algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    # X, y = dataset
    X = points
    # print(y)
    print(np.shape(X))
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params['n_neighbors'], include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    ward = cluster.AgglomerativeClustering(
        n_clusters=params['n_clusters'], linkage='ward',
        connectivity=connectivity)
    spectral = cluster.SpectralClustering(
        n_clusters=params['n_clusters'], eigen_solver='arpack',
        affinity="nearest_neighbors")
    dbscan = cluster.DBSCAN(eps=params['eps'])
    affinity_propagation = cluster.AffinityPropagation(
        damping=params['damping'], preference=params['preference'])
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock",
        n_clusters=params['n_clusters'], connectivity=connectivity)
    birch = cluster.Birch(n_clusters=params['n_clusters'])
    gmm = mixture.GaussianMixture(
        n_components=params['n_clusters'], covariance_type='full')

    clustering_algorithms = (
        # ('MiniBatchKMeans', two_means),
        # ('AffinityPropagation', affinity_propagation),
        # ('MeanShift', ms),
        # ('SpectralClustering', spectral),
        ('Ward', ward),
        # ('AgglomerativeClustering', average_linkage),
        # ('DBSCAN', dbscan),
        # ('Birch', birch),
        # ('GaussianMixture', gmm),
    )

    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                "connectivity matrix is [0-9]{1,2}" +
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                " may not work as expected.",
                category=UserWarning)
            algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
            print(y_pred)
        else:
            y_pred = algorithm.predict(X)

        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1

        cx0 = good_new[y_pred == 0, 0]
        cy0 = good_new[y_pred == 0, 1]

        cx1 = good_new[y_pred == 1, 0]
        cy1 = good_new[y_pred == 1, 1]

        cx2 = good_new[y_pred == 2, 0]
        cy2 = good_new[y_pred == 2, 1]

        cx3 = good_new[y_pred == 3, 0]
        cy3 = good_new[y_pred == 3, 1]

        cx4 = good_new[y_pred == 4, 0]
        cy4 = good_new[y_pred == 4, 1]

        for i in range(0, np.int32(np.shape(cx0)[0])):
            imm = cv.circle(im1, (cx0[i], cy0[i]), 4, (0, 0, 255), -1)
            img2 = cv.add(imm, mask)
        for i in range(0, np.int32(np.shape(cx1)[0])):
            imm = cv.circle(im1, (cx1[i], cy1[i]), 4, (0, 255, 0), -1)
            img2 = cv.add(imm, mask)
        for i in range(0, np.int32(np.shape(cx2)[0])):
            imm = cv.circle(im1, (cx2[i], cy2[i]), 4, (255, 0, 0), -1)
            img2 = cv.add(imm, mask)
        for i in range(0, np.int32(np.shape(cx3)[0])):
            imm = cv.circle(im1, (cx3[i], cy3[i]), 4, (255, 255, 0), -1)
            img2 = cv.add(imm, mask)
        for i in range(0, np.int32(np.shape(cx4)[0])):
            imm = cv.circle(im1, (cx4[i], cy4[i]), 4, (255, 0, 255), -1)
            img2 = cv.add(imm, mask)
        cv.imshow('frame2', img2)
        cv.waitKey()

plt.show()
"""
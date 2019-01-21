"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
import scipy.ndimage.filters as fltr
import random
im1 = cv.imread('/home/mai/Pictures/vlcsnap-2018-11-08-17h39m18s247.jpg')
im1_gray = cv.cvtColor(im1, cv.COLOR_RGB2GRAY)
im2 = cv.imread('/home/mai/Pictures/vlcsnap-2018-11-08-17h39m29s485.jpg')
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
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
#change 10000 number
color = np.random.randint(0, 255, (10000, 3))

# change here and test another detectors --canny --
ctr = 0
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
print('ctr', np.shape(p0))
print('max p0 0', max(p0[:, :, 0]))
print('max p0 1', max(p0[:, :, 1]))
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

# magnitude2 = fltr.median_filter(magnitude, 5)
print('x_shape', np.shape(x))
print('prod', np.shape(p0))
# k = cv.waitKey(30) & 0xff
# if k == 27:
#     break
# old_gray = frame_gray.copy()
# p0 = good_new.reshape(-1,1,2)
# p0 = cv.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
# cv.destroyAllWindows()
# cap.release()

# if all_features:
#     x, y = np.mgrid[margin:w-margin:step, margin:h-margin:step]
#     pair = np.vstack((x.flatten(), y.flatten())).T
#     p0 = np.reshape(pair, (np.shape(pair)[0], 1, 2))
#     p0 = np.float32(p0)
# else:
#     p0 = cv.goodFeaturesToTrack(im1_gray, mask=None, **feature_params)

if all_features:
    print('theta shape', np.shape(theta))
    theta_2d = np.reshape(theta, np.shape(x))

    # theta_2d_filt = fltr.laplace(theta_2d)
    theta_2d_filt = theta_2d
# magnitude = np.sqrt(np.add(delta_y**2, delta_x**2))

    delta_x_2d = np.reshape(delta_x, (np.shape(x)[0], np.shape(x)[1]))
    delta_y_2d = np.reshape(delta_y, (np.shape(x)[0], np.shape(x)[1]))
    print('delta_x_2d', np.shape(delta_x_2d))
    # du_grad_x = fltr.sobel(fltr.median_filter(delta_x_2d, 3), 0)
    # du_grad_y = fltr.sobel(fltr.median_filter(delta_x_2d, 3), 1)
    # dv_grad_x = fltr.sobel(fltr.median_filter(delta_y_2d, 3), 0)
    # dv_grad_y = fltr.sobel(fltr.median_filter(delta_y_2d, 3), 1)

    du_grad_x = fltr.sobel(delta_x_2d, 0)
    du_grad_y = fltr.sobel(delta_x_2d, 1)
    dv_grad_x = fltr.sobel(delta_y_2d, 0)
    dv_grad_y = fltr.sobel(delta_y_2d, 1)
    print('x_shape', (np.shape(x)[1], np.shape(x)[0]))
    print('size', np.size(x))
    du_grad_x_1d = np.reshape(du_grad_x, np.size(x))
    du_grad_y_1d = np.reshape(du_grad_y, np.size(x))
    dv_grad_x_1d = np.reshape(dv_grad_x, np.size(x))
    dv_grad_y_1d = np.reshape(dv_grad_y, np.size(x))
    theta_filt = np.reshape(theta_2d_filt, np.size(x))
    # print(np.shape(theta_filt))
    # print(np.shape(dv_grad_y_1d))
    motion = np.zeros(np.size(x))
    # print(np.shape(motion))
    motion[theta_filt > thresh_angle] = np.maximum(abs(du_grad_x_1d[theta_filt > thresh_angle]) + abs(dv_grad_x_1d[theta_filt > thresh_angle]),
                        abs(du_grad_y_1d[theta_filt > thresh_angle]) + abs(dv_grad_y_1d[theta_filt > thresh_angle]))
    # motion = np.maximum(abs(du_grad_x_1d) + abs(dv_grad_x_1d),
    #                     abs(du_grad_y_1d) + abs(dv_grad_y_1d))
    print('motion_size', np.shape(motion))
    motion[motion < 15] = 0
# 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = p0[:, :, 0]
    Y = p0[:, :, 1]
    print('X_size', np.shape(X))
    print('Y_size', np.shape(Y))
    np.set_printoptions(threshold=np.nan)
    print(X)
    # z2 = theta_2d_filt
    ax.scatter(X, Y, magnitude, c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # x_sort = np.sort(X, 0)
    # np.set_printoptions(threshold=np.nan)
    # print(x_sort)
    plt.show()

# clustering
points = np.zeros((np.shape(magnitude)[0], 1))
points[:, 0] = magnitude[:, 0]
# points[:, 1] = theta[:, 0]
# points[:, 2] = good_old[:, 0, 0]
# points[:, 3] = good_old[:, 0, 1]
points = np.float32(points)
#
# # ================================
# # test other methods of clustering
# # ================================
#
# print(__doc__)
#
# import time
# import warnings
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# from sklearn import cluster, datasets, mixture
# from sklearn.neighbors import kneighbors_graph
# from sklearn.preprocessing import StandardScaler
# from itertools import cycle, islice
#
# np.random.seed(0)
#
# # ============
# # Generate datasets. We choose the size big enough to see the scalability
# # of the algorithms, but not too big to avoid too long running times
# # ============
# n_samples = 1500
# noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
#                                       noise=.05)
# # print(noisy_circles)
# # print(np.shape(noisy_circles))
# # print(noisy_circles)
# noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
# blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
# no_structure = np.random.rand(n_samples, 2), None
#
# # Anisotropicly distributed data
# random_state = 170
# X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
# transformation = [[0.6, -0.6], [-0.4, 0.8]]
# X_aniso = np.dot(X, transformation)
# aniso = (X_aniso, y)
#
# # blobs with varied variances
# varied = datasets.make_blobs(n_samples=n_samples,
#                              cluster_std=[1.0, 2.5, 0.5],
#                              random_state=random_state)
#
# # ============
# # Set up cluster parameters
# # ============
# plt.figure(figsize=(9 * 2 + 3, 12.5))
# plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
#                     hspace=.01)
#
# plot_num = 1
#
# default_base = {'quantile': .3,
#                 'eps': .3,
#                 'damping': .7,
#                 'preference': -200,
#                 'n_neighbors': 10,
#                 'n_clusters': 2}
#
# datasets = [
#     # (noisy_circles, {'damping': .77, 'preference': -240,
#     #                  'quantile': .2, 'n_clusters': 2}),
#     # (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
#     # (varied, {'eps': .18, 'n_neighbors': 2}),
#     # (aniso, {'eps': .15, 'n_neighbors': 2}),
#     # (blobs, {}),
#     (no_structure, {})]
#
# for i_dataset, (dataset,algo_params) in enumerate(datasets):
#     # update parameters with dataset-specific values
#     params = default_base.copy()
#     params.update(algo_params)
#
#     # X, y = dataset
#     X = points
#     # print(y)
#     print(np.shape(X))
#     # normalize dataset for easier parameter selection
#     X = StandardScaler().fit_transform(X)
#
#     # estimate bandwidth for mean shift
#     bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])
#
#     # connectivity matrix for structured Ward
#     connectivity = kneighbors_graph(
#         X, n_neighbors=params['n_neighbors'], include_self=False)
#     # make connectivity symmetric
#     connectivity = 0.5 * (connectivity + connectivity.T)
#
#     # ============
#     # Create cluster objects
#     # ============
#     ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
#     two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
#     ward = cluster.AgglomerativeClustering(
#         n_clusters=params['n_clusters'], linkage='ward',
#         connectivity=connectivity)
#     spectral = cluster.SpectralClustering(
#         n_clusters=params['n_clusters'], eigen_solver='arpack',
#         affinity="nearest_neighbors")
#     dbscan = cluster.DBSCAN(eps=params['eps'])
#     affinity_propagation = cluster.AffinityPropagation(
#         damping=params['damping'], preference=params['preference'])
#     average_linkage = cluster.AgglomerativeClustering(
#         linkage="average", affinity="cityblock",
#         n_clusters=params['n_clusters'], connectivity=connectivity)
#     birch = cluster.Birch(n_clusters=params['n_clusters'])
#     gmm = mixture.GaussianMixture(
#         n_components=params['n_clusters'], covariance_type='full')
#
#     clustering_algorithms = (
#         # ('MiniBatchKMeans', two_means),
#         # ('AffinityPropagation', affinity_propagation),
#         # ('MeanShift', ms),
#         # ('SpectralClustering', spectral),
#         # ('Ward', ward),
#         # ('AgglomerativeClustering', average_linkage),
#         ('DBSCAN', dbscan),
#         # ('Birch', birch),
#         # ('GaussianMixture', gmm),
#     )
#
#     for name, algorithm in clustering_algorithms:
#         t0 = time.time()
#
#         # catch warnings related to kneighbors_graph
#         with warnings.catch_warnings():
#             warnings.filterwarnings(
#                 "ignore",
#                 message="the number of connected components of the " +
#                 "connectivity matrix is [0-9]{1,2}" +
#                 " > 1. Completing it to avoid stopping the tree early.",
#                 category=UserWarning)
#             warnings.filterwarnings(
#                 "ignore",
#                 message="Graph is not fully connected, spectral embedding" +
#                 " may not work as expected.",
#                 category=UserWarning)
#             algorithm.fit(X)
#
#         t1 = time.time()
#         if hasattr(algorithm, 'labels_'):
#             y_pred = algorithm.labels_.astype(np.int)
#             print(y_pred)
#         else:
#             y_pred = algorithm.predict(X)
#
#         plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
#         if i_dataset == 0:
#             plt.title(name, size=18)
#
#         colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
#                                              '#f781bf', '#a65628', '#984ea3',
#                                              '#999999', '#e41a1c', '#dede00']),
#                                       int(max(y_pred) + 1))))
#         # add black color for outliers (if any)
#         colors = np.append(colors, ["#000000"])
#         # plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
#         #
#         # plt.xlim(-2.5, 2.5)
#         # plt.ylim(-2.5, 2.5)
#         # plt.xticks(())
#         # plt.yticks(())
#         # plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
#         #          transform=plt.gca().transAxes, size=15,
#         #          horizontalalignment='right')
#         # plot_num += 1
#
#         cx0 = good_new[y_pred == 0, :, 0]
#         cy0 = good_new[y_pred == 0, :, 1]
#
#         cx1 = good_new[y_pred == 1, :, 0]
#         cy1 = good_new[y_pred == 1, :, 1]
#
#         cx2 = good_new[y_pred == 2, :, 0]
#         cy2 = good_new[y_pred == 2, :, 1]
#
#         for i in range(0, np.int32(np.shape(cx0)[0])):
#             imm = cv.circle(im1, (cx0[i], cy0[i]), 4, (0, 0, 255), -1)
#             img2 = cv.add(imm, mask)
#         for i in range(0, np.int32(np.shape(cx1)[0])):
#             imm = cv.circle(im1, (cx1[i], cy1[i]), 4, (0, 255, 0), -1)
#             img2 = cv.add(imm, mask)
#         for i in range(0, np.int32(np.shape(cx2)[0])):
#             imm = cv.circle(im1, (cx2[i], cy2[i]), 4, (255, 0, 0), -1)
#             img2 = cv.add(imm, mask)
#         cv.imshow('frame2', img2)
#         cv.waitKey()
#
# plt.show()
"""
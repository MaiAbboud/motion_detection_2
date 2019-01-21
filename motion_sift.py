# import cv2 as cv
# import matplotlib.pyplot as plt
#
# img = cv.imread('Orange_Fiber_Arance1.jpg')
# surf = cv.xfeatures2d.SURF_create(400)
# # Find keypoints and descriptors directly
# kp, des = surf.detectAndCompute(img ,None)
# print(len(kp))
# img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
# plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB)),plt.show()
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import timeit
MIN_MATCH_COUNT = 10
img1 = cv.imread('/home/mai/Pictures/vlcsnap-2018-11-09-12h18m09s410.jpg', 0)
img2 = cv.imread('/home/mai/Pictures/vlcsnap-2018-11-09-12h18m04s992.jpg', 0)
start = timeit.default_timer()

# Initiate SIFT detector
sift = cv.xfeatures2d.SURF_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm= FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks= 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    print(M)
    matchesMask = mask.ravel().tolist()
    # h,w,d = img1.shape
    # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    # dst = cv.perspectiveTransform(pts,M)
    # img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None



# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 2)
# img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
# plt.imshow(img3, 'gray'),plt.show()
rows, cols = img1.shape
# result = img2
result = cv.warpPerspective(img1, M, (img1.shape[1] , img1.shape[0]))
# result[0:img1.shape[0], 0:img1.shape[1]] = img1
# plt.imshow(result,'gray'),plt.show()

a = cv.absdiff(img2, img1)
b = cv.absdiff(img2, result)

stop = timeit.default_timer()
print('Time: ', stop - start)

fig=plt.figure()
columns = 2
rows = 1
fig.add_subplot(rows, columns, 1)
plt.imshow(a, 'gray')

fig.add_subplot(rows, columns, 2)
plt.imshow(b, 'gray')
plt.show()

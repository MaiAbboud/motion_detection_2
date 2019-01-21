"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import timeit
MIN_MATCH_COUNT = 10
img1 = cv.imread('/home/mai/Downloads/Dataset/BMS_dataset/videos/seq1/seq8.jpg', 0)
# im1_gray = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
img2 = cv.imread('/home/mai/Downloads/Dataset/BMS_dataset/videos/seq1/seq9.jpg', 0)
# im2_gray = cv.cvtColor(im2, cv.COLOR_RGB2GRAY)

start = timeit.default_timer()

orb = cv.ORB_create()
# find the keypoints with ORB
kp1 = orb.detect(img1,None)
# compute the descriptors with ORB
kp1, des1 = orb.compute(img1, kp1)
# find the keypoints with ORB
kp2 = orb.detect(img2,None)
# compute the descriptors with ORB
kp2, des2 = orb.compute(img2, kp2)
 # create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1, des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

good = []
for m in matches:
    # if m.distance < 0.7:
    good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    print(np.shape(src_pts))
    print(M)
    matchesMask = mask.ravel().tolist()
    # h,w,d = img1.shape
    # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    # dst = cv.perspectiveTransform(pts,M)
    # img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
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
"""
"""
# import cv2
# import numpy as np

# im = cv2.imread('Orange_Fiber_Arance1.jpg')
# h, w = im.shape[:2]
# print(h, w)
# cv2.imshow('a', im)
# cv2.waitKey(2000)
#
# im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
# im_gary2 = 2*im_gray
# cv2.imshow('a', im_gary2)
# cv2.waitKey(2000)
#
# cap = cv2.VideoCapture('Mercedes')
# frames = []
# while True:
#     ret, im = cap.read()
#     cv2.imshow('v', im)
#     im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
#     frames.append(im_gray)
#     if cv2.waitKey(10) == 27:
#         break
# frames = np.array(frames)
# print(im_gray.shape)
# print(frames.shape)
#
#
#
#
# def draw_flow(im,flow,step=16):
#     h,w = im.shape[:2]
#     print(h,w)
#     x,y = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
#     fx , fy = flow[y,x].T
#     # print(fx,fy)
#     # priqqnt(flow[:])
#     # lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
#     # lines = np.int32(lines)
#     vis = cv2.cvtColor(im , cv2.COLOR_RGB2GRAY)
#     # for (x1,y1),(x2,y2) in lines:
#     #     cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
#     #     cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
#     return vis


# cap = cv2.VideoCapture('Mercedes')
# ret, im = cap.read()
# prev_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
# while True:
#     ret, im = cap.read()
#     gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
#     #
#     flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#
#     prev_gray = gray
#     print(flow)
#     # im2 = draw_flow(im ,flow)
#     # cv2.imshow('optical_flow', draw_flow(gray, flow))
#     if cv2.waitKey(10) == 27:
#         break
# import numpy as np
# import cv2 as cv
# cap = cv.VideoCapture('/home/mai/Videos/T3.mp4')
# # params for ShiTomasi corner detection
# feature_params = dict( maxCorners = 1000,
#                        qualityLevel = 0.01,
#                        minDistance = 7,
#                        blockSize = 7 )
# # Parameters for lucas kanade optical flow
# lk_params = dict( winSize  = (15,15),
#                   maxLevel = 2,
#                   criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# # Create some random colors
# color = np.random.randint(0,255,(1000,3))
# # Take first frame and find corners in it
# ret, old_frame = cap.read()
# old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# # Create a mask image for drawing purposes
# mask = np.zeros_like(old_frame)
# while(1):
#     ret,frame = cap.read()
#     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     p0 = cv.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
#     # calculate optical flow
#     p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#     # Select good points
#     good_new = p1[st == 1]
#     good_old = p0[st == 1]
#     # draw the tracks
#     for i,(new,old) in enumerate(zip(good_new, good_old)):
#         a,b = new.ravel()
#         c,d = old.ravel()
#         mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
#         frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
#     img = cv.add(frame,mask)
#     cv.imshow('frame',img)
#     k = cv.waitKey(30) & 0xff
#     if k == 27:
#         break
#     # Now update the previous frame and previous points
#     old_gray = frame_gray.copy()
#     p0 = good_new.reshape(-1,1,2)
# cv.destroyAllWindows()
# cap.release()

import numpy as np
import cv2 as cv

cap = cv.VideoCapture('/home/mai/Videos/T3.mp4')
feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.1,
                       minDistance = 7,
                       blockSize = 7 )


lk_params = dict( winSize=(15, 15),
                  maxLevel=2,
                  criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT, 30, 0.01))

color = np.random.randint(0, 255, (1000, 3))

ret, old_frame = cap.read()
# try algorithms with colors
old_gray = cv.cvtColor(old_frame, cv.COLOR_RGB2GRAY)

#change here and test another detectors --canny --
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    mask = np.zeros_like(old_frame)
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv.circle(frame,(a,b),1,color[i].tolist(),-1)
    img = cv.add(frame,mask)
    cv.imshow('frame',img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    old_gray = frame_gray.copy()
    # p0 = good_new.reshape(-1,1,2)
    p0 = cv.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

cv.destroyAllWindows()
cap.release()
"""
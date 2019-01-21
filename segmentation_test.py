"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('/home/mai/Downloads/Dataset/BMS_dataset/videos/seq1/seq28.jpg', 0)
edges = cv.Canny(img, 100, 200)
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
#
# kernel = np.ones((3, 3), np.uint8)
# mask_skin = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=10)
# cv2.imshow("th", mask_skin)
# cv2.waitKey()

# import numpy as np
# import cv2
# img = cv2.imread('home.jpg')
Z = img.reshape((-1, 3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 6
ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv.imshow('res2', res2)
cv.waitKey(0)
cv.destroyAllWindows()
"""
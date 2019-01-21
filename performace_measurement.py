import numpy as np
import cv2 as cv
from os import listdir
from os.path import isfile, join

my_url = '/home/mai/Downloads/Dataset/BMS_dataset/algorithm2/cars4/'
gnd_url = '/home/mai/Downloads/Dataset/BMS_dataset/GroundTruth/cars4/'
only_files_my_url = [f for f in listdir(my_url) if isfile(join(my_url, f))]
only_files_gnd_url = [f for f in listdir(my_url) if isfile(join(my_url, f))]
num_result_images = len(only_files_my_url)
num_gnd_images = len(only_files_gnd_url)
im1 = cv.imread(my_url+'%s.png' % 1)

TP = np.zeros(np.size(im1))
FP = np.zeros(np.size(im1))
FN = np.zeros(np.size(im1))

file_precision = open(my_url+'precision.txt', 'w')
file_recall = open(my_url+'recall.txt', 'w')
file_F_measure = open(my_url+'F_measure.txt', 'w')

# precision = np.zeros(np.size(num_gnd_images))
for i in range(1, 1000, 1):
    # j = np.int32(1 + (4 * (i - 1)))
    result = cv.imread(my_url+'%s.png' % i)
    if np.shape(result) == ():
        break
    # cv.imshow("im", result)
    # cv.waitKey(10)
    im_gth = cv.imread(gnd_url+'%s.png' % i)
    result_1d = np.reshape(result, np.size(result))
    im_gth_1d = np.reshape(im_gth, np.size(im_gth))

    TP = np.zeros(np.size(im1))
    FP = np.zeros(np.size(im1))
    FN = np.zeros(np.size(im1))

    TP[(result_1d == 255) & (im_gth_1d == 255)] = 1
    FP[(result_1d == 255) & (im_gth_1d == 0)] = 1
    FN[(result_1d == 0) & (im_gth_1d == 255)] = 1
    TP_sum = sum(TP)
    FP_sum = sum(FP)
    FN_sum = sum(FN)
    precision = TP_sum / (TP_sum + FP_sum)
    recall = TP_sum/(TP_sum + FN_sum)
    F_measure = (2*precision*recall)/(precision+recall)
    file_precision.write('%s\n' % precision)
    file_recall.write('%s\n' % recall)
    file_F_measure.write('%s\n' % F_measure)
    print(precision)
    print(recall)
    print(F_measure)
file_precision.close()
file_F_measure.close()
file_recall.close()

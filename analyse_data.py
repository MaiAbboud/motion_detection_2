import numpy as np
import cv2 as cv
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
name_of_movie = 'cars5'
parameter = 'parameters_time'
algo1_url = '/home/mai/Downloads/Dataset/BMS_dataset/algorithm1/'
algo2_url = '/home/mai/Downloads/Dataset/BMS_dataset/algorithm2/'
algo4_url = '/home/mai/Downloads/Dataset/BMS_dataset/algorithm4/'
file1 = open(algo1_url + name_of_movie + '/' + parameter + '.txt', 'r')
file2 = open(algo2_url + name_of_movie + '/' + parameter + '.txt', 'r')
file4 = open(algo4_url + name_of_movie + '/' + parameter + '.txt', 'r')
list1 = []
list2 = []
list4 = []
ctr = 0
# for line in file1:
for line in file1.readlines():
    ctr = ctr + 1
    if ctr > 2:
        list1.append(float(line))
ctr = 0
for line in file2.readlines():
    ctr = ctr + 1
    if ctr > 2:
        list2.append(float(line))
# ctr = 0
for line in file4.readlines():
    ctr = ctr + 1
    if ctr > 2:
        list4.append(float(line))

# file4 = open(algo4_url + 'cars4/precision.txt', 'r')
# a = file1.read()
# print(a[3])
plt.plot(list1, 'r', label='algorithm 1')
plt.plot(list2, 'y', label='algorithm 2')
plt.plot(list4, 'b', label='algorithm 4')
plt.grid(True)
legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large')
plt.title(parameter + ' ' + name_of_movie)

plt.show()
print(list)
file1.close()
# precision1
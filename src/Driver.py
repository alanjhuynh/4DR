import numpy as np
import cv2
import time
import os, os.path
import SingleCheckerboard as sc
import ExternalCalibration as ec
from random import randrange

import matplotlib.pyplot as plt

mtx1, dist1, roi1, newmtx1 = sc.getInternalCharacteristics(0)

mtx1 = np.loadtxt('internal_mtx1.csv', delimiter=',')
dist1 = np.loadtxt('internal_dist1.csv', delimiter=',')
roi1 = np.loadtxt('internal_roi1.csv', delimiter=',')
newmtx1 = np.loadtxt('internal_newmtx1.csv', delimiter=',')
print(mtx1)
print(dist1)
print(roi1)
print(newmtx1)

mtx2 = np.loadtxt('internal_mtx2.csv', delimiter=',')
dist2 = np.loadtxt('internal_dist2.csv', delimiter=',')
roi2 = np.loadtxt('internal_roi2.csv', delimiter=',')
newmtx2 = np.loadtxt('internal_newmtx2.csv', delimiter=',')
print(mtx2)
print(dist2)
print(roi2)
print(newmtx2)
#ec.getExternalCharacteristics()

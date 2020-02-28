import numpy as np
import cv2

imagePoints1 = np.array([[-1.0], [-1.0]])
tempPoints1 = np.array([[-1.0], [-1.0]])

F = np.array([[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]]])

print(F)
N = 0

for i in range(0, 10, 1):
    if(N == 0):
        imagePoints1[0][N] = float(i*.001)
        imagePoints1[1][N] = float(i*.001)
        N += 1
    else:
        tempPoints1[0][0] = float(i*.001)
        tempPoints1[1][0] = float(i*.001)
        imagePoints1 = np.append(imagePoints1, tempPoints1, axis = 1)
print(imagePoints1)
print(imagePoints1.shape)
imagePoints1 = np.reshape(np.ravel(imagePoints1, order='F'), (1, np.size(imagePoints1, 1), 2), order='C')
print(imagePoints1)

testArray = np.array([[[-1.0,-1.0],[-2,-2],[-3,-3]]])
print(testArray)

print("Shapes")
print(imagePoints1.shape)
print(testArray.shape)

imagePoints1, imagePoints2 = cv2.correctMatches(F, imagePoints1, imagePoints1)

print(imagePoints1)

imagePoints1 = np.reshape(np.ravel(imagePoints1, order='F'), (2, np.size(imagePoints1, 1)), order='C')
print(imagePoints1)

a = np.arange(6).reshape((3, 2))
print(a)
a = np.reshape(np.ravel(a, order='F'),(2, 3), order='C')
print(a)
a = np.reshape(np.ravel(a, order='F'),(3, 2), order='C')
print(a)

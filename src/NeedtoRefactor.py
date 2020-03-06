import numpy as np
import cv2
import time
import os, os.path
import SingleCheckerboard as sc
from random import randrange

import matplotlib.pyplot as plt

def createLine(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Creating Line for " + param[0])
        print(param[1])
        if param[0] == "Frame1":
            imageIndex = 1
            otherIndex = 2
        else:
            imageIndex = 2
            otherIndex = 1

        color = (randrange(256), randrange(256) , randrange(256))

        point = np.array([x,y])
        line = [0,0,0]
        line = cv2.computeCorrespondEpilines(point.reshape(-1,1,2), imageIndex, param[1])
        line = line[0][0]
        size = param[2].shape[1]
        x0, y0 = map(int, [0, -line[2]/line[1]])
        x1, y1 = map(int, [size, -(line[2]+(line[0]*float(size)))/line[1]])
        cv2.line(param[2], (x0, y0),(x1, y1),color, 2)

        point = np.array([x1,y1])
        line = [0,0,0]
        line = cv2.computeCorrespondEpilines(point.reshape(-1,1,2), otherIndex, param[1])
        line = line[0][0]
        x0, y0 = map(int, [0, -line[2]/line[1]])
        x1, y1 = map(int, [size, -(line[2]+(line[0]*float(size)))/line[1]])
        cv2.line(param[3], (x0, y0),(x1, y1),color, 2)
        
def correspondPoint(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        windowEnd = param[5]
        windowStart = - windowEnd
        windowSize = (2 * windowEnd) + 1
        print("Creating Point for " + param[0])
        if param[0] == "NewFrame1":
            imageIndex = 1
            otherIndex = 2
        else:
            imageIndex = 2
            otherIndex = 1
        color = (randrange(256), randrange(256) , randrange(256))

        window = np.zeros(windowSize*windowSize)
        n = 0
        for i in range(windowStart, windowEnd + 1, 1): #Outside Loop is Y Value
            for j in range(windowStart, windowEnd + 1, 1): #InsideLoop is X Value
                #print(" ( " + str(x + j) + ", " + str(y + i) + ")")
                window[n] = param[2][y + i][x + j]
                n += 1
        mean = np.mean(window)
        n = 0
        
        for i in range(windowStart, windowEnd + 1, 1): #Outside Loop is Y Value
            for j in range(windowStart, windowEnd + 1, 1): #InsideLoop is X Value
                window[n] = window[n] - mean
                n += 1
        magnitude = np.linalg.norm(window)
                
        
        
        maxValue = 0
        bestN = -1
        xValue = np.zeros(1)
        yValue = np.zeros(1)
        windowPrime = np.zeros(windowSize * windowSize)
        
        for n in range(windowEnd, 635 + windowStart, 1):
            
            k = 0
            for i in range(windowStart, windowEnd + 1, 1): #Outside Loop is Y Value
                for j in range(windowStart, windowEnd + 1, 1): #InsideLoop is X Value
                    windowPrime[k] = param[1][y + i][n + j]
                    #print(" ( " + str(n + j) + ", " + str(y + i) + ")")
                    k += 1
            mean = np.mean(windowPrime)        
            k = 0       
            for i in range(windowStart, windowEnd + 1, 1): #Outside Loop is Y Value
                for j in range(windowStart, windowEnd + 1, 1): #InsideLoop is X Value
                    windowPrime[k] = windowPrime[k] - mean
                    k += 1
            magnitudePrime = np.linalg.norm(windowPrime)
            
            testValue = np.dot(window, windowPrime)/(magnitude * magnitudePrime)
            xValue = np.append(xValue, n)
            yValue = np.append(yValue, testValue)
            if( maxValue < testValue):
                print(n)
                print(testValue)
                bestN = n
                maxValue = testValue

        plt.plot(xValue, yValue, label = "U")
        plt.xlabel('Pixel')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
        if(bestN != -1):
            cv2.circle(param[4], (x,y), 4, color)
            cv2.circle(param[2], (x,y), 4, color)
            cv2.circle(param[3], (bestN,y), 4, color)
            cv2.circle(param[1], (bestN,y), 4, color)

          
ret, mtx1, dist1, mtx2, dist2, Rmtx, Tmtx, Emtx, Fmtx = cv2.stereoCalibrate(objpoints, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2, gray2.shape[::-1], 256, criteria)



h = gray1.shape[0]
w = gray1.shape[1]
print(mtx1.shape)
print(newmtx1.shape)

## START
while(True):
    name = input("Enter image name you want to test:")
    if(os.path.isfile("testImage/" + name + "1.jpg") == False):
        print("Doesn't Exist")
        continue
    
    frame1 = cv2.imread("testImage/" + name + "1.jpg")
    frame2 = cv2.imread("testImage/" + name + "2.jpg")
    break
    
##END

#Before Rectification
undistort1 = cv2.undistort(frame1,mtx1,dist1, None, newmtx1)
undistort2 = cv2.undistort(frame2,mtx2,dist2, None, newmtx2)

x,y,w,h = roi1
undistort1 = undistort1[y:y+h,x:x+w]
print(roi1)
x,y,w,h = roi2
print(roi2)
undistort2 = undistort2[y:y+h,x:x+w]

cv2.namedWindow('Frame1')
cv2.namedWindow('Frame2')
cv2.imshow('Frame1',undistort1)
cv2.imshow('Frame2',undistort2)
cv2.waitKey(1)
cv2.setMouseCallback('Frame1',createLine, ["Frame1", Fmtx, undistort2, undistort1])
cv2.setMouseCallback('Frame2',createLine, ["Frame2", Fmtx, undistort1, undistort2])

Rmtx1, Rmtx2, Pmtx1, Pmtx2, Qmtx, roinew1, roinew2 = cv2.stereoRectify(mtx1, dist1, mtx2, dist2, gray2.shape[::-1], Rmtx, Tmtx)
print(Rmtx1)
print(Rmtx2)
print(Pmtx1)
print(Pmtx2)
print(roinew1)
print(roinew2)

map1_1, map1_2 = cv2.initUndistortRectifyMap(mtx1, dist1, Rmtx1, Pmtx1, gray1.shape[::-1], cv2.CV_32FC1)
map2_1, map2_2 = cv2.initUndistortRectifyMap(mtx2, dist2, Rmtx2, Pmtx2, gray2.shape[::-1], cv2.CV_32FC1)

#After Rectification
newFrame1 = cv2.remap(frame1, map1_1, map1_2, cv2.INTER_LINEAR)
newFrame2 = cv2.remap(frame2, map2_1, map2_2, cv2.INTER_LINEAR)

testFrame1 = newFrame1.copy()
testFrame2 = newFrame2.copy()
cleanFrame1 = newFrame1.copy()
cleanFrame2 = newFrame2.copy()

x,y,w,h = roinew1
cv2.line(newFrame1, (x, y),(x+w, y),(0,255,0), 2)
cv2.line(newFrame1, (x, y),(x, y+h),(0,255,0), 2)
cv2.line(newFrame1, (x+w, y+h),(x+w, y),(0,255,0), 2)
cv2.line(newFrame1, (x+w, y+h),(x, y+h),(0,255,0), 2)
x,y,w,h = roinew2
cv2.line(newFrame2, (x, y),(x+w, y),(0,255,0), 2)
cv2.line(newFrame2, (x, y),(x, y+h),(0,255,0), 2)
cv2.line(newFrame2, (x+w, y+h),(x+w, y),(0,255,0), 2)
cv2.line(newFrame2, (x+w, y+h),(x, y+h),(0,255,0), 2)
h = gray1.shape[0]
w = gray1.shape[1]
color = 80
test1 = frame1.copy()
test2 = frame2.copy()

for n in range(0, h, 20):
    color += 20
    if(color >= 255):
        color = 100
    cv2.line(newFrame1, (0, n),(w, n),(color,0,0), 1)
    cv2.line(newFrame2, (0, n),(w, n),(color,0,0), 1)
    cv2.line(test1, (0, n),(w, n),(color,0,0), 1)
    cv2.line(test2, (0, n),(w, n),(color,0,0), 1)

final = np.concatenate((newFrame1,newFrame2),axis=1)
before = np.concatenate((test1,test2),axis=1)

while(True):
    cv2.imshow('Frame1',undistort1)
    cv2.imshow('Frame2',undistort2)
    cv2.imshow('NewFrame1',newFrame1)
    cv2.imshow('NewFrame2',newFrame2)
    cv2.imshow('Before',before)
    cv2.imshow('Final',final)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

gray1 = cv2.cvtColor(testFrame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(testFrame2, cv2.COLOR_BGR2GRAY)

cv2.imshow('NewFrame1',testFrame1)
cv2.imshow('NewFrame2',testFrame2)
cv2.waitKey(1)
cv2.setMouseCallback('NewFrame1',correspondPoint, ["NewFrame1", gray2, gray1, testFrame2, testFrame1, 20])
cv2.setMouseCallback('NewFrame2',correspondPoint, ["NewFrame2", gray1, gray2, testFrame1, testFrame2, 20])

while(True):
    cv2.imshow('NewFrame1',testFrame1)
    cv2.imshow('NewFrame2',testFrame2)
    cv2.imshow('Gray1', gray1)
    cv2.imshow('Gray2', gray2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
        

windowStart = -5
windowEnd = 5
windowSize = 11
window = np.zeros(windowSize*windowSize)
windowPrime = np.zeros(windowSize*windowSize)
threshold = .5
imagePoints1 = np.array([[-1.0], [-1.0]])
imagePoints2 = np.array([[-1.0], [-1.0]])

gray1 = cv2.cvtColor(cleanFrame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(cleanFrame2, cv2.COLOR_BGR2GRAY)
N = 0

#CORRESPONDENCE
while(True):
    cropXstart = int(input("New start X: "))
    cropXend = int(input("New end X: "))
    cropYstart = int(input("New start Y: "))
    cropYend = int(input("New end Y: "))
    if(cropXstart < windowEnd or cropXend > w + windowStart - 1):
        print("cropX won't work")
        continue
    if(cropYstart < windowEnd or cropYend > h + windowStart - 1):
        print("cropY won't work")
        continue

    break;


magPrimeArray = np.zeros((h - (2*windowEnd),w - (2*windowEnd) - cropXstart))
winPrimeArray = np.zeros((h - (2*windowEnd),w - (2*windowEnd) - cropXstart, windowSize*windowSize))


for y in range(cropYstart, cropYend + windowStart, 1):
    for n in range (windowEnd, w + windowStart - cropXstart, 1):
        k = 0
        for i in range(windowStart, windowEnd + 1, 1):
            for j in range(windowStart, windowEnd + 1, 1):
                windowPrime[k] = gray2[y+i][n+j]
                k += 1
        mean = np.mean(windowPrime)  
        k = 0
        for i in range(windowStart, windowEnd + 1, 1):
            for j in range(windowStart, windowEnd + 1, 1):
                windowPrime[k] = windowPrime[k] - mean
                k += 1
        magnitudePrime = np.linalg.norm(windowPrime)
        magPrimeArray[y - windowEnd][n - windowEnd] = magnitudePrime
        winPrimeArray[y  - windowEnd][n - windowEnd] = windowPrime.copy()
        
for y in range(cropYstart, cropYend + windowStart, 1):
    for x in range(cropXstart, cropXend + windowStart, 1):

        n = 0
        for i in range(windowStart, windowEnd + 1, 1):
            for j in range(windowStart, windowEnd + 1, 1):
                window[n] = gray1[y+i][x+j]
                n += 1
        mean = np.mean(window)
        n = 0
        for i in range(windowStart, windowEnd + 1, 1):
            for j in range(windowStart, windowEnd + 1, 1):
                window[n] = window[n] - mean
                n += 1
        magnitude = np.linalg.norm(window)

        maxValue = 0
        bestN = -1
        
        for n in range (windowEnd, w + windowStart - cropXstart, 1):
            
            data = magPrimeArray[y - windowEnd][n - windowEnd] * magnitude
            if(data == 0):
                continue

            testValue = np.dot(window, winPrimeArray[y - windowEnd][n - windowEnd])/(data)
            if(maxValue<testValue):
                bestN = n
                maxValue = testValue

        if(maxValue > threshold):
            if(N == 0):
                #MAKE SURE THIS IS RIGHT (X is 0 and Y is 1)
                imagePoints1[0][N] = float(x)
                imagePoints1[1][N] = float(y)
                imagePoints2[0][N] = float(bestN)
                imagePoints2[1][N] = float(y)
                N += 1
            else:
                imagePoints1 = np.append(imagePoints1, np.array([[float(x)], [float(y)]]), axis = 1)
                imagePoints2 = np.append(imagePoints2, np.array([[float(bestN)], [float(y)]]), axis = 1)
                N += 1

print(imagePoints1)
print(imagePoints2)
print(np.size(imagePoints1, 0))
print(np.size(imagePoints1, 1))
print(N)

imagePoints1 = np.reshape(np.ravel(imagePoints1, order='F'), (1, np.size(imagePoints1, 1), 2), order='C')
imagePoints2 = np.reshape(np.ravel(imagePoints2, order='F'), (1, np.size(imagePoints2, 1), 2), order='C')
imagePoints1, imagePoints2 = cv2.correctMatches(Fmtx, imagePoints1, imagePoints2)
imagePoints1 = np.reshape(np.ravel(imagePoints1, order='F'), (2, np.size(imagePoints1, 1)), order='C')
imagePoints2 = np.reshape(np.ravel(imagePoints2, order='F'), (2, np.size(imagePoints2, 1)), order='C')


raw3dpoints = cv2.triangulatePoints(Pmtx1, Pmtx2, imagePoints1, imagePoints2)

print(raw3dpoints)

f = open("test2.ply", "w")
f.write("ply \n")
f.write("format ascii 1.0\n")
f.write("element vertex " + str(np.size(raw3dpoints,1))+"\n")
f.write("property float x \n")
f.write("property float y \n")
f.write("property float z \n")
f.write("property uchar red \n")
f.write("property uchar green \n")
f.write("property uchar blue \n")
f.write("end_header \n")
for i in range(0, np.size(raw3dpoints,1), 1):
    x = float(raw3dpoints[0][i]/raw3dpoints[3][i])
    y = float(raw3dpoints[1][i]/raw3dpoints[3][i])
    z = float(raw3dpoints[2][i]/raw3dpoints[3][i])
    red = cleanFrame1[int(imagePoints1[1][i])][int(imagePoints1[0][i])][2]
    green = cleanFrame1[int(imagePoints1[1][i])][int(imagePoints1[0][i])][1]
    blue = cleanFrame1[int(imagePoints1[1][i])][int(imagePoints1[0][i])][0]
    f.write(str(x) + " " + str(y)+ " "  + str(z) + " " + str(red) + " " + str(green) + " " + str(blue) + "\n")
f.close()




## 100 to 600
## 100 to 420














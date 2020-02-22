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

          
    
imagePoints1 = np.array([[-1.0, -1.0]])
print(type(imagePoints1[0]))
print(imagePoints1[0])
imagePoints1 = np.append(imagePoints1, [[1,1]], axis = 0)
print(imagePoints1)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

ret1, frame1 = cap1.read()
ret2, frame2 = cap2.read()

directory = "images"
list = os.listdir(directory)
number_files = len(list)
print(number_files)


start = 0
frameNum = 0

GREEN = (124,252,0)

while(True):

    while(True):
        if frameNum == 0:
            start = time.time()
        
        
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        saveFrame1 = frame1.copy()
        saveFrame2 = frame2.copy()
        
        # Our operations on the frame come here
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)            
        gray1 = np.float32(gray1)
        gray2 = np.float32(gray2)
        gray1 = cv2.cornerHarris(gray1,3,3,0.04)
        gray2 = cv2.cornerHarris(gray2,3,3,0.04)
        gray1 = cv2.dilate(gray1,None)
        gray2 = cv2.dilate(gray2,None)
        frame1[gray1>0.01*gray1.max()]=[0,0,255]
        frame2[gray2>0.01*gray2.max()]=[0,0,255]
        
        h = gray1.shape[0]
        w = gray1.shape[1]

        cv2.line(frame1, (0,int(h/2)), (w,int(h/2)), GREEN)
        cv2.line(frame1, (0,int(h/2) + 1), (w,int(h/2) + 1), GREEN)
        cv2.line(frame1, (int(w/2),0), (int(w/2),h), GREEN)
        cv2.line(frame1, (int(w/2) + 1,0), (int(w/2) + 1,h), GREEN)
        cv2.line(frame2, (0,int(h/2)), (w,int(h/2)), GREEN)
        cv2.line(frame2, (0,int(h/2) + 1), (w,int(h/2) + 1), GREEN)
        cv2.line(frame2, (int(w/2),0), (int(w/2),h), GREEN)
        cv2.line(frame2, (int(w/2) + 1,0), (int(w/2) + 1,h), GREEN)
        

        cv2.imshow('Frame1',frame1)
        cv2.imshow('Frame2',frame2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capturing images")
            number_files += 1
            title = "images/" + str(number_files) + ".jpg"
            cv2.imwrite(title, saveFrame1)
            number_files += 1
            title = "images/" + str(number_files) + ".jpg"
            cv2.imwrite(title, saveFrame2)
        if cv2.waitKey(1) & 0xFF == ord('w'):
            break

        frameNum += 1
        if frameNum == 45:
            end = time.time()
            fps = frameNum/(end-start)
            frameNum = 0
            print(fps)
    
    print("Take Another Photo? Y = Yes, N = No")
    choice = input()
    if(choice == 'Y'):
        cap1 = cv2.VideoCapture(0)
        cap2 = cv2.VideoCapture(1)
        continue
    else:
        break
    


## START
print("Create Test Pictures")
while(True):
 
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        saveFrame1 = frame1.copy()
        saveFrame2 = frame2.copy()       

        cv2.imshow('Frame1',frame1)
        cv2.imshow('Frame2',frame2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capturing Test Image")
            name = input("Input Name:")
            number_files += 1
            title = "testImage/" + str(name) + "1.jpg"
            cv2.imwrite(title, saveFrame1)
            number_files += 1
            title = "testImage/" + str(name) + "2.jpg"
            cv2.imwrite(title, saveFrame2)
        if cv2.waitKey(1) & 0xFF == ord('w'):
            cap1.release()
            cap2.release()
            break
## END
cap1.release()
cap2.release()


index = 1
for i in range(1, number_files, 2):
    
    
    imgname1 = str(i)+ ".jpg"
    imgname2 = str(i + 1) + ".jpg"

    if(os.path.isfile("images/" + imgname1) == False):
        continue
    
    img1 = cv2.imread("images/" + imgname1)
    img2 = cv2.imread("images/" + imgname2)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret1, temp = cv2.findChessboardCorners(gray1, (8,6), None)
    ret2, temp = cv2.findChessboardCorners(gray2, (8,6), None)

    

    if (ret1 and ret2) == True:
        title = "images/" + str(index) + ".jpg"
        cv2.imwrite(title, img1)
        index += 1
        title = "images/" + str(index) + ".jpg"
        cv2.imwrite(title, img2)
        index += 1
        continue
    else:
        try: 
            os.remove("images/" + imgname1)
            os.remove("images/" + imgname2)
        except: pass

objp = np.zeros((6*8,3),np.float32)
objp[:,:2] = .25*(np.mgrid[0:8,0:6].T.reshape(-1,2))

objpoints1 = []
imgpoints1 = []
objpoints2 = []
imgpoints2 = []
objpoints = []

for i in range(1, number_files, 2):
    index = i
    imgname1 = str(index)+ ".jpg"
    index += 1
    imgname2 = str(index) + ".jpg"
    print(index)

    if(os.path.isfile("images/" + imgname1) == False):
        continue
    
    frame1 = cv2.imread("images/" + imgname1)
    frame2 = cv2.imread("images/" + imgname2)

    
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray1, (8,6), None)
    objpoints1.append(objp)
    corners2 = cv2.cornerSubPix(gray1, corners, (11,11), (-1,-1), criteria)
    imgpoints1.append(corners2)

    ret, corners = cv2.findChessboardCorners(gray2, (8,6), None)
    objpoints2.append(objp)
    corners2 = cv2.cornerSubPix(gray2, corners, (11,11), (-1,-1), criteria)
    imgpoints2.append(corners2)


    objpoints.append(objp)
    
print(gray1.shape[::-1],gray2.shape[::-1])
    
mtx1, dist1, roi1, newmtx1 = sc.getInternalCharacteristics(0) #From My Library
mtx2, dist2, roi2, newmtx2 = sc.getInternalCharacteristics(1)

print(mtx1)
print(dist1)
print(mtx2)
print(dist2)

    
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
    if(cropXstart < windowEnd or cropXend > h + windowStart - 1):
        print("cropX won't work")
        continue
    if(cropYstart < windowEnd or cropYend > w + windowStart - 1):
        print("cropY won't work")
        continue

    break;


magPrimeArray = np.zeros((h - (2*windowEnd),w - (2*windowEnd)))
winPrimeArray = np.zeros((h - (2*windowEnd),w - (2*windowEnd), windowSize*windowSize))


for y in range(cropYstart, cropYend + windowStart, 1):
    for n in range (windowEnd, w + windowStart, 1):
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
        
        for n in range (windowEnd, w + windowStart, 1):
            
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



















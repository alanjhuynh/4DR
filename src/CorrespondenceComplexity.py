import numpy as np
import cv2
import time
import os, os.path

def Compare(original, new):
    originalSize = original.size
    newSize = new.size
    print(originalSize, newSize)
    print(original)
    print(new)
    if(originalSize != newSize):
        print("Size")
        return False
    return True

while(True):
    name = input("Enter image name you want to test:")
    if(os.path.isfile("testImage/" + name + "1.jpg") == False):
        print("Doesn't Exist")
        continue
    
    frame1 = cv2.imread("testImage/" + name + "1.jpg")
    frame2 = cv2.imread("testImage/" + name + "2.jpg")
    break

windowStart = -10
windowEnd = 10
windowSize = 21
window = np.zeros(windowSize*windowSize)
windowPrime = np.zeros(windowSize*windowSize)
threshold = .5
imagePoints1 = np.array([[-1.0, -1.0]])
imagePoints2 = np.array([[-1.0, -1.0]])
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
N = 0
h = 40
w = 40


start_time = time.time()
for y in range(windowEnd, h + windowStart, 1):
    for x in range(windowEnd, w + windowStart, 1):
        
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

            if(magnitudePrime * magnitude == 0):
                continue

            testValue = np.dot(window, windowPrime)/(magnitude * magnitudePrime)
            if(maxValue<testValue):
                bestN = n
                maxValue = testValue

        if(maxValue > threshold):
            if(N == 0):
                #MAKE SURE THIS IS RIGHT (X is 0 and Y is 1)
                imagePoints1[N][0] = float(x)
                imagePoints1[N][1] = float(y)
                imagePoints2[N][0] = float(bestN)
                imagePoints2[N][1] = float(y)
                N += 1
            else:
                np.append(imagePoints1, [[float(x),float(y)]], axis = 0)
                np.append(imagePoints2, [[float(bestN),float(y)]], axis = 0)

end_time = time.time()

print("Current Process: " + str(end_time - start_time) + " Seconds")


multTestPoints_1 = np.array([[-1.0, -1.0]])
multTestPoints_2 = np.array([[-1.0, -1.0]])
multArray = np.zeros((w, h, w)) #Depth Dimension, Rows, Columns
magPrimeArray = np.zeros((w - (2*windowEnd), h - (2*windowEnd)))
winPrimeArray = np.zeros((w - (2*windowEnd), h - (2*windowEnd), windowSize*windowSize))
N = 0

start_time = time.time()
for y in range(windowEnd, h + windowStart, 1):
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
        
for y in range(windowEnd, h + windowStart, 1):
    for x in range(windowEnd, w + windowStart, 1):

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
                multTestPoints_1[N][0] = float(x)
                multTestPoints_1[N][1] = float(y)
                multTestPoints_2[N][0] = float(bestN)
                multTestPoints_2[N][1] = float(y)
                N += 1
            else:
                print("APPEND " + str(x) + ", " + str(y) )
                np.append(multTestPoints_1, [[float(x),float(y)]], axis = 1)
                print(multTestPoints_1)
                np.append(multTestPoints_2, [[float(bestN),float(y)]], axis = 1)

end_time = time.time()

compareBool = Compare(imagePoints1, multTestPoints_1)
if(not compareBool):
    print("Test Failed")

print("Current Process: " + str(end_time - start_time) + " Seconds")  

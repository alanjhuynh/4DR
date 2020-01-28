import numpy as np
import cv2
import time
import os, os.path

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

ret1, frame1 = cap1.read()
ret2, frame2 = cap2.read()

directory = "CalibrationImages"
list = os.listdir(directory)
number_files = len(list)
print(number_files)


# Our operations on the frame come here
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
gray1 = np.float32(gray1)
gray2 = np.float32(gray2)
gray1 = cv2.cornerHarris(gray1,2,3,0.04)
gray2 = cv2.cornerHarris(gray2,2,3,0.04)
gray1 = cv2.dilate(gray1,None)
gray2 = cv2.dilate(gray2,None)
print(gray1.size)

frame1[gray1>0.01*gray1.max()]=[0,0,255]
frame2[gray2>0.01*gray2.max()]=[0,0,255]
    
# Display the resulting frame
cv2.imshow('frame1',frame1)
cv2.imshow('frame2',frame2)

number_files += 1
title = "CalibrationImages/" + str(number_files) + ".jpg"
cv2.imwrite(title, frame1)
number_files += 1
title = "CalibrationImages/" + str(number_files) + ".jpg"
cv2.imwrite(title, frame2)

start = 0
frameNum = 0


while(True):

    if frameNum == 0:
        start = time.time()
    
    
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
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

    cv2.imshow('frame1',frame1)
    cv2.imshow('frame2',frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('c'):
        print("Saving Images")
        number_files += 1
        title = "CalibrationImages/" + str(number_files) + ".jpg"
        cv2.imwrite(title, frame1)
        number_files += 1
        title = "CalibrationImages/" + str(number_files) + ".jpg"
        cv2.imwrite(title, frame2)

    frameNum += 1
    if frameNum == 45:
        end = time.time()
        fps = frameNum/(end-start)
        frameNum = 0
        print(fps)
        


# When everything done, release the capture
cap1.release()
cap2.release()
cv2.destroyAllWindows()




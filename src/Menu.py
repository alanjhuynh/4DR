import numpy as np
import cv2
import time
import os, os.path
import SingleCheckerboard as sc
from random import randrange

#Change 0's to left, and 1's to right
# 0 and 1 are the camera number, not left or right

def PrintActions():
    print("Actions:")
    #print("Direction: Determine which camera is on the left and right")
    print("CalibrateLeft: Calibrate the left camera, internal parameters")
    print("CalibrateRight: Calibrate the right camera, internal parameters")
    print("CalibrateExternal: Calibrate the external parameters")
    print("EdgeDetection: Edge detection is ran on both cameras")
    print("Checkerboard: Checkerboard Detection is ran on both cameras")
    print("Rectify: Rectify a pair of images")
    print("Exit: Exit the program")
    print("Help: Reprint all the possible actions")
    print("\n")

def CalibrateInternal(Side, directory):
    cap = cv2.VideoCapture(Side)
    ret, frame = cap.read()

    list = os.listdir(directory)
    number_files = len(list)
    print(number_files)

    while(True):
        ret, frame = cap.read()
     
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capturing images")
            number_files += 1
            title = directory + "/" + str(number_files) + ".jpg"
            cv2.imwrite(title, saveFrame1)       
        if cv2.waitKey(1) & 0xFF == ord('w'):
            cap1.release()
            cap2.release()
            break
        
    for index in range(1, number_files, 1):
        imgname = str(index)+ ".jpg"
        
        img = cv2.imread(directory + "/" + imgname)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, temp = cv2.findChessboardCorners(gray, (8,6), None)

        if (ret) == True:
            continue
        else:
            try: 
                os.remove("images/" + imgname)
            except: pass
            
    objp = np.zeros((6*8,3),np.float32)
    objp[:,:2] = .25*(np.mgrid[0:8,0:6].T.reshape(-1,2))

    objpoints = []
    imgpoints = []
        
    mtx, dist, roi, newmtx = sc.getInternalCharacteristics(0) #From My Library
    
    return mtx, dist, roi, newmtx

def CalibrateExternal():
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

    return imgpoints1, imgpoints2, objpoints, gray2.shape[::-1]
    
print("Welcome to 4DR")
print("Loading Data...")

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

PrintActions()
Left = 0
Right = 1

mtx1, dist1, roi1, newmtx1 = sc.getInternalCharacteristics(0) #From My Library
mtx2, dist2, roi2, newmtx2 = sc.getInternalCharacteristics(1)
imgpoints1, imgpoints2, objpoints, size = CalibrateExternal()

ret, mtx1, dist1, mtx2, dist2, Rmtx, Tmtx, Emtx, Fmtx = cv2.stereoCalibrate(objpoints, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2, size, 256, criteria)

while(True):
    action = input("Enter your next action: ")

    if(action == "CalibrateLeft"):
        mtx1, dist1, roi1, newmtx1 = CalibrateInternal(Left, "imagesLeft")
    elif(action == "CalibrateRight"):
        mtx2, dist2, roi2, newmtx2 = CalibrateInternal(Right, "imagesRight")
    elif(action == "CalibrateExternal"):
        CalibrateExternal()
    elif(action == "Exit"):
        break;
    elif(action == "Help"):
        PrintActions()
    else:
        print(action + " is not a proper input")
        print("\n")

    

import numpy as np
import cv2
import time
import os, os.path

def createLine(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Creating Line for " + param[0])
        print(type(param[1]))
        print(x,y)
        if param[0] == "Frame1":
            imageIndex = 2
        else:
            imageIndex = 1

        point = np.array([x,y])
        print(point)

        line = [0,0,0]

        line = cv2.computeCorrespondEpilines(point.reshape(-1,1,2), imageIndex, param[1])
        line = line[0][0]
        print(line)

        size = param[2].shape[1]
        print(size)

        x0, y0 = map(int, [0, -line[2]/line[1]])
        x1, y1 = map(int, [size, -(line[2]+(line[0]*float(size)))/line[1]])
        cv2.line(param[2], (x0, x1),(x1, y1),(0,255,0), 1)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

ret1, frame1 = cap1.read()
ret2, frame2 = cap2.read()

directory = "CalibrationImages"
list = os.listdir(directory)
number_files = len(list)
print(number_files)


start = 0
frameNum = 0

objp = np.zeros((6*8,3),np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

while(True):

    while(True):
        if frameNum == 0:
            start = time.time()
        
        
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        # Our operations on the frame come here
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        ret1, temp = cv2.findChessboardCorners(gray1, (8,6), None)
        ret2, temp = cv2.findChessboardCorners(gray2, (8,6), None)

        if (ret1 and ret2) == True:
            print(objp)
            
            objpoints1 = []
            imgpoints1 = []

            ret, corners = cv2.findChessboardCorners(gray1, (8,6), None)
            objpoints1.append(objp)
            corners2 = cv2.cornerSubPix(gray1, corners, (11,11), (-1,-1), criteria)
            imgpoints1.append(corners2)
            img1 = frame1.copy()

            cv2.drawChessboardCorners(img1, (8,6), corners2, ret)

            
            ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints1, imgpoints1, gray1.shape[::-1],None,None)

            objpoints2 = []
            imgpoints2 = []

            ret, corners = cv2.findChessboardCorners(gray2, (8,6), None)
            objpoints2.append(objp)
            corners2 = cv2.cornerSubPix(gray2, corners, (11,11), (-1,-1), criteria)
            imgpoints2.append(corners2)
            img2 = frame2.copy()
            
            cv2.drawChessboardCorners(img2, (8,6), corners2, ret)

            ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints2, imgpoints2, gray2.shape[::-1],None,None)

            cap1.release()
            cap2.release()
            cv2.imshow('Frame1',frame1)
            cv2.imshow('newFrame1',img1)
            cv2.imshow('Frame2',frame2)
            cv2.imshow('newFrame2',img2)

            objpoints = []
            objpoints.append(objp)


            ret, mtx1, dist1, mtx2, dist2, Rmtx, Tmtx, Emtx, Fmtx = cv2.stereoCalibrate(objpoints, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2, gray2.shape[::-1])

            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
            
        gray1 = np.float32(gray1)
        gray2 = np.float32(gray2)
        gray1 = cv2.cornerHarris(gray1,3,3,0.04)
        gray2 = cv2.cornerHarris(gray2,3,3,0.04)
        gray1 = cv2.dilate(gray1,None)
        gray2 = cv2.dilate(gray2,None)
        frame1[gray1>0.01*gray1.max()]=[0,0,255]
        frame2[gray2>0.01*gray2.max()]=[0,0,255]

        cv2.imshow('Frame1',frame1)
        cv2.imshow('Frame2',frame2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap1.release()
            cap2.release()
            cv2.destroyAllWindows()
            break
        elif cv2.waitKey(1) & 0xFF == ord('c'):
            cap1.release()
            cap2.release()
            cv2.destroyAllWindows()
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
h = gray1.shape[0]
w = gray1.shape[1]
print(mtx1.shape)

newmtx1, roi1 = cv2.getOptimalNewCameraMatrix(mtx1, dist1, (w,h), 1, (w,h))
newmtx2, roi2 = cv2.getOptimalNewCameraMatrix(mtx2, dist2, (w,h), 1, (w,h))

print(newmtx1.shape)

frame1 = cv2.undistort(frame1,mtx1,dist1, None, newmtx1)
frame2 = cv2.undistort(frame2,mtx2,dist2, None, newmtx2)

x,y,w,h = roi1
frame1 = frame1[y:y+h,x:x+w]
x,y,w,h = roi2
frame2 = frame2[y:y+h,x:x+w]

cv2.namedWindow('Frame1')
cv2.namedWindow('Frame2')
cv2.imshow('Frame1',frame1)
cv2.imshow('Frame2',frame2)
cv2.waitKey(1)
cv2.setMouseCallback('Frame1',createLine, ["Frame1", Fmtx, frame2])
cv2.setMouseCallback('Frame2',createLine, ["Frame2", Fmtx, frame1])

while(True):
    cv2.imshow('Frame1',frame1)
    cv2.imshow('Frame2',frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break



# When everything done, release the capture

import numpy as np
import cv2
import time
import os, os.path

def getInternalCharacteristics(camera):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    cap1 = cv2.VideoCapture(camera)

    ret1, frame1 = cap1.read()

    directory = "images" + str(camera + 1)
    list = os.listdir(directory)
    number_files = len(list)
    #print(number_files)


    start = 0
    frameNum = 0

    objp = np.zeros((6*8,3),np.float32)
    objp[:,:2] = .25 * np.mgrid[0:8,0:6].T.reshape(-1,2)

    objpoints = []
    imgpoints = []


    while(True):

        while(True):
            if frameNum == 0:
                start = time.time()
                
            ret1, frame1 = cap1.read()
            saveFrame1 = frame1.copy()
            
            # Our operations on the frame come here
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)      
            gray1 = np.float32(gray1)
            gray1 = cv2.cornerHarris(gray1,3,3,0.04)
            gray1 = cv2.dilate(gray1,None)
            frame1[gray1>0.01*gray1.max()]=[0,0,255]

            cv2.imshow('Frame1',frame1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Capturing images")
                number_files += 1
                title = directory + "/" + str(number_files) + ".jpg"
                cv2.imwrite(title, saveFrame1)
            if cv2.waitKey(1) & 0xFF == ord('w'):
                break

            frameNum += 1
            if frameNum == 45:
                end = time.time()
                fps = frameNum/(end-start)
                frameNum = 0
                #print(fps)
        
        print("Take Another Photo? Y = Yes, N = No")
        choice = input()
        if(choice == 'Y'):
            cap1 = cv2.VideoCapture(0)
            continue
        else:
            break

    cv2.destroyAllWindows()
    cap1.release()

    for i in range(1, number_files+1, 1):
        index = i
        imgname1 = str(index)+ ".jpg"
        #print(index)
        
        frame1 = cv2.imread(directory + "/" + imgname1)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray1, (8,6), None)
        if(ret == False):
            continue
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray1, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        
        
    #print(gray1.shape[::-1])
        
    ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints, gray1.shape[::-1],None,None)

    #print(mtx1)
    #print(dist1)

    h = gray1.shape[0]
    w = gray1.shape[1]
    #print(mtx1.shape)

    frame1 = cv2.imread(directory + "/" + "10.jpg")
    originalFrame1 = frame1.copy()

    newmtx1, roi1 = cv2.getOptimalNewCameraMatrix(mtx1, dist1, (w,h), 1, (w,h))
    frame1 = cv2.undistort(frame1,mtx1,dist1, None, newmtx1)
    x,y,w,h = roi1
    frame1 = frame1[y:y+h,x:x+w]
    cv2.namedWindow('Frame1')
    cv2.imshow('Frame1',frame1)
    cv2.namedWindow('Original')
    cv2.imshow('Original',originalFrame1)
    cv2.waitKey(1)
    while(True):
        cv2.imshow('Frame1',frame1)
        cv2.imshow('Original',originalFrame1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    np.savetxt('calibration/internal_mtx{}'.format(camera+1) + '.csv', mtx1, delimiter=',')
    np.savetxt('calibration/internal_dist{}'.format(camera+1) + '.csv', dist1, delimiter=',')
    np.savetxt('calibration/internal_roi{}'.format(camera+1) + '.csv', roi1, delimiter=',')
    np.savetxt('calibration/internal_newmtx{}'.format(camera+1) + '.csv', newmtx1, delimiter=',')
    return mtx1, dist1, roi1, newmtx1



















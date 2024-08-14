### Overlay marker ID and video
import cv2
from cv2 import aruco
import time
import os
from os import listdir
from PIL import Image
import numpy as np
import csv

dict_aruco = aruco.getPredefinedDictionary(aruco.DICT_7X7_1000)
parameters = aruco.DetectorParameters()

# Camera matrix and distrotion parameters
cameraMatrix = np.array([[692.87286,0,357],
                [0,692.23956,269],
                [0,0,1]])
distCoeffs = np.array([0.0,0.0,0.0,0.0,0.0])

# Directories
img_dir = "./Images/Rec_20230309T153850Z/"
res_dir = "./Images/ArucoDetectImages/"

# Open file and write data to the file
fn_res = img_dir +"ArucoPoseResult.csv"
f = open(fn_res,'w')
writer = csv.writer(f)
header = ['trans_x','trans_y','trans_z','rotv_x','rotv_y','rotv_z']
writer.writerow(header)


for images in sorted(os.listdir(img_dir)):
    fn = img_dir + images
    frame = cv2.imread(fn)    
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)
    rotation_vectors, translation_vectors, _objPoints = aruco.estimatePoseSingleMarkers(corners, 0.035, cameraMatrix, distCoeffs)

    # Write to csv file
    try:
        row = [translation_vectors[0,0,0],translation_vectors[0,0,1],translation_vectors[0,0,2],rotation_vectors[0,0,0],rotation_vectors[0,0,1],rotation_vectors[0,0,2]]
    except:
        row = [0,0,0,0,0,0]
    writer.writerow(row)

    fn_img = res_dir + images
    try:
        for rvec, tvec in zip(rotation_vectors, translation_vectors):
            ProjectImage = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
            ProjectImage = cv2.drawFrameAxes(ProjectImage, cameraMatrix, distCoeffs, rvec, tvec, 1)
            # cv2.imshow('ProjectImage',ProjectImage)
            cv2.imwrite(fn_img,ProjectImage)
    except:
        # cv2.imshow('ProjectImage',frame)
        cv2.imwrite(fn_img,frame)

    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     break
cv2.destroyWindow('frame')
f.close()

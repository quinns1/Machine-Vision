# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 18:47:27 2021

@author: Shane
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import random
random.seed(7)


'To view videos/images generated set PLOT to True'
PLOT = True  

def task_1():
    
    print('-'*50+'\n\t\t\t\tTask 1\n'+'-'*50)
    
    'Task 1_A - Checkerboard Corners'
    
    cal_img_dir = os.getcwd() + r'\Assignment_MV_02_calibration'        #Save calibration images in specified folder
    img_list = os.listdir(cal_img_dir)  
    X = []                                              #3D coordinate list
    op = np.zeros((7*5,3), np.float32)                  #Prep 3D coordinates for all checkboard corners
    op[:,:2] = np.mgrid[0:7, 0:5].T.reshape(-1,2)
    x = []                                              #2D Coordinates of checkboard corners
    
    for i in img_list:
        """
        Iterate through all calibration images, identify corners using cv2.findChessboardCorners and draw using
        cv2.drawChessboardCorners. 
        """
        img = cv2.imread(cal_img_dir + '\\' + i)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(img, (7,5))  
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(img, corners, (11,11), (-1,-1), criteria)
        x.append(corners2)
        X.append(op)
        cv2.drawChessboardCorners(img, (7,5), corners2, ret) 
        if PLOT:
            cv2.imshow(i, img)
    
    'Task 1_B - Calibration Matrix'
    
    ret, K, d, r, t = cv2.calibrateCamera(X, x, img.shape[::-1], None, None)
    print("Camera calibration Matrix \'K\':\n", K)
    pl = K[0][0]
    print("Principle length: ", pl)
    ar = K[1][1]/K[0][0]
    print("Aspect ration: ", ar)
    pp = K[0][2]/K[1][2]
    print("Principle point: ", pp) 
    
    
    'Task 1_C - Feature Extraction'
    
    video = cv2.VideoCapture('Assignment_MV_02_video.mp4')

    while video.isOpened():
        ret,img= video.read()        
        if ret:
            new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)        
            p0 = cv2.goodFeaturesToTrack(new_img, 0, 0.1, 10)               #Identify good features to track                                 
            break   
        
        

    p0 = cv2.cornerSubPix(new_img, p0, (11, 11), (-1,-1), criteria)         #Refine feature points to sub-pixel accuracy
    print("Tracking {} features.".format(len(p0)))

    #Initialise tracks
    index = np.arange(len(p0))          #Record feature point indexes
    tracks = {}
    for i in range(len(p0)):
        tracks[index[i]] = {0:p0[i]}    #For each point, record in tracks dict (tracks = {point_index: {frame: coordinates}})

    frame = 0
    while video.isOpened():
        ret,img= video.read()           #Iterate through frames               
        if not ret:
            break

        frame += 1
        old_img = new_img
        new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)        

        #Use cv2.calcOpticalFlowPyrLK to calculate optical flow, and refine to sub-pixel accuracy
        if len(p0)>0: 
            p1, st, err  = cv2.calcOpticalFlowPyrLK(old_img, new_img, p0, None)     
            p1 = cv2.cornerSubPix(new_img, p1, (11, 11), (-1,-1), criteria)             #Refine feature points to sub-pixel accuracy
            p0 = p1[st==1].reshape(-1,1,2)            
            index = index[st.flatten()==1]
        
        #Update tracks dictionary
        for i in range(len(p0)):
            if index[i] in tracks:
                tracks[index[i]][frame] = p0[i]
            else:
                tracks[index[i]] = {frame: p0[i]}
        
    video.release()
        
    
    return p0, index, tracks, frame, K
    	 

def skew(x):
    return np.array([[0,-x[2],x[1]],
                     [x[2],0,-x[0]],
                     [-x[1],x[0],0]])



def task_2(p0, index, tracks, frames):
    
    print('-'*50+'\n\t\t\t\tTask 2\n'+'-'*50)
    
    'Task 2_A - Visualise Tracks in first and last frames'
    video = cv2.VideoCapture('Assignment_MV_02_video.mp4')
    frame = 0
    frame1 = 0
    frame2 = 30
    d = 0
    
    ts = copy.deepcopy(tracks)
    for track in ts:
        'Remove tracks that are not in both frame0 and frame30'
        if (frame1 in tracks[track]) and (frame2 in tracks[track]):
            pass
        else:
            tracks.pop(track)
            
    print("Number of points in both Frame 0 and Frame 30: ", len(tracks))
            
    while video.isOpened():
        ret,img= video.read()        
        if not ret:                                       
            break   
        frame += 1
        for i in range(len(index)):                                         
            for f in range(0,frame):
                if (f in tracks[index[i]]) and (f+1 in tracks[index[i]]):
                    cv2.line(img,
                              (int(tracks[index[i]][f][0,0]), int(tracks[index[i]][f][0,1])),
                              (int(tracks[index[i]][f+1][0,0]), int(tracks[index[i]][f+1][0,1])), 
                              (0,255,0), 1)    

        if PLOT:
            k = cv2.waitKey(0)
            if k%256 == 27:                                                     #Press escape to exit
                break
            cv2.imshow("video", img)    

    video.release()
    cv2.destroyWindow("video")   

    coords_f1 = []                              #Initialise lists for holding co-ordinates of points in Frame0 and Frame30
    coords_f2 = []
    correspondences = []
    for track in tracks:
        x1 = [tracks[track][frame1][0,0],tracks[track][frame1][0,1],1]
        x2 = [tracks[track][frame2][0,0],tracks[track][frame2][0,1],1]
        correspondences.append((np.array(x1), np.array(x2)))
        coords_f1.append((tracks[track][frame1][0,0],tracks[track][frame1][0,1]))
        coords_f2.append((tracks[track][frame2][0,0],tracks[track][frame2][0,1]))
    best_outliers = len(correspondences)+1
    best_error = 1e100
    
    'Task 2_B'
    
    coords_f1 = np.array(coords_f1)
    coords_f2 = np.array(coords_f2)
    meanf1 = 1/len(coords_f1)*np.sum(coords_f1, axis=0)
    meanf2 = 1/len(coords_f1)*np.sum(coords_f2, axis=0)
    stdevf1 = np.sqrt(1/len(coords_f1)*np.sum(np.square(coords_f1-meanf1), axis=0))
    stdevf2 = np.sqrt(1/len(coords_f2)*np.sum(np.square(coords_f2-meanf2), axis=0))


    T1 = np.array([[1/stdevf1[0], 0, -meanf1[0]/stdevf1[0]],
                  [0, 1/stdevf1[1], -meanf1[1]/stdevf1[1]],
                  [0, 0, 1]])
    T2 = np.array([[1/stdevf2[0], 0, -meanf2[0]/stdevf2[0]],
                  [0, 1/stdevf2[1], -meanf2[1]/stdevf2[1]],
                  [0, 0, 1]])
    
    correspondences_norm = []

    for cor in correspondences:
        x1 = cor[0]
        x2 = cor[1]
        y1 = np.matmul(T1, x1)
        y2 = np.matmul(T2, x2)
        correspondences_norm.append((y1, y2))
         
    
    'Task 2_C'

    
    c_xx = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 0]])

    for iteration in range(10000):        
        
        'Task 2_D'
        
        samples_in = set(random.sample(range(len(correspondences)),8))          #Get 8 random points for DLT algorithm
        samples_out = set(range(len(correspondences))).difference(samples_in)
        A = np.zeros((0,9))
        factor = {}
        
        for i in samples_in:
            y1, y2 = correspondences_norm[i]           
            ai = np.kron(y1.T,y2.T)
            A = np.append(A,[ai],axis=0)
                
        U,S,V = np.linalg.svd(A)    
        H = V[8,:].reshape(3,3).T                
        F = np.matmul(T2.T, np.matmul(H, T1))        
        count_outliers = 0
        accumulate_error = 0
        inlier_cor = []
        
        'Task 2_E'
        
        for i in samples_out:                           #For remainder of points find model equation + variance
            x1, x2 = correspondences[i]            
            g_i = np.matmul(x2.T, np.matmul(F, x1))
            a = np.matmul(x2.T, np.matmul(F, np.matmul(c_xx, np.matmul(F.T, x2))))
            b = np.matmul(x1.T, np.matmul(F.T, np.matmul(c_xx, np.matmul(F, x1))))
            sig_i = a+b
            T = np.divide(g_i**2, sig_i**2)
    
            'Task 2_F'
            if T>6.635:
                inlier_cor.append((y1, y2))
                count_outliers += 1
            else:
                accumulate_error += T
                

        if count_outliers<best_outliers:
            best_error = accumulate_error
            best_outliers = count_outliers
            best_H = H
            best_F = F
            best_s = samples_in
            best_inlier_cor = inlier_cor
        elif count_outliers==best_outliers:         #In the event of a tie, sum of test statistic over inliers wins
            if accumulate_error<best_error:
                best_error = accumulate_error
                best_outliers = count_outliers
                best_H = H
                best_F = F
                best_s = samples_in
      
        
        
    'Task 2_G'  
    samples_in = best_s
    count_inliers = 0
    samples_out =set(range(len(correspondences))).difference(samples_in)
    inliers = []
    F = best_F
    H = best_H
    for i in samples_out:
        x1,x2 = correspondences[i]            
        g_i = np.matmul(x2.T, np.matmul(F, x1))
        a = np.matmul(x2.T, np.matmul(F, np.matmul(c_xx, np.matmul(F.T, x2))))
        b = np.matmul(x1.T, np.matmul(F.T, np.matmul(c_xx, np.matmul(F, x1))))
        sig_i = a+b
        T = np.divide(g_i**2, sig_i**2)


        if T<=6.635:                     #Save inliers
            inliers.append(i)
            count_inliers += 1
    
    print("Fundamental matrix: \n", F)
    print("Number of inliers: ", count_inliers)

    
            
    'Task 2_H'
    
    video = cv2.VideoCapture('Assignment_MV_02_video.mp4')
    frame = 0
    while video.isOpened():
        ret,img= video.read()        
        if not ret:                                       
            break   
        frame += 1
        for i in range(len(index)):                                         
            for f in range(0,frame):
                try:
                    if (f in tracks[inliers[i]]) and (f+1 in tracks[inliers[i]]):
                        cv2.line(img,
                                  (int(tracks[inliers[i]][f][0,0]), int(tracks[inliers[i]][f][0,1])),
                                  (int(tracks[inliers[i]][f+1][0,0]),int(tracks[inliers[i]][f+1][0,1])), 
                                  (0,255,0), 3)
                except:                
                    if (f in tracks[index[i]]) and (f+1 in tracks[index[i]]):
                        cv2.line(img,
                                  (int(tracks[index[i]][f][0,0]),int(tracks[index[i]][f][0,1])),
                                  (int(tracks[index[i]][f+1][0,0]),int(tracks[index[i]][f+1][0,1])), 
                                  (0,0,255), 1)
                        
        if PLOT:
            k = cv2.waitKey(0)
            if k%256 == 27:                                                     #Press escape to exit
                break
            cv2.imshow("Inliers vs Outliers", img)  
            
    video.release()
    cv2.destroyWindow("Inliers vs Outliers") 
    
    video = cv2.VideoCapture('Assignment_MV_02_video.mp4')       
    frame = 0
    while video.isOpened():
        ret,img= video.read() 
        if frame == 0:
            frame_0 = img
        elif frame == 30:
            frame_30 = img
        if not ret:                                       
            break 
        frame += 1  
    
    U,S,V = np.linalg.svd(F)    
    e1 = V[2,:]
    

    U,S,V = np.linalg.svd(F.T)    
    e2 = V[2,:]
    
    print("Epipole Co-ordinates F0: ", e1/e1[2])    
    print("Epipole Co-ordinates F30: ", e2/e2[2])
    
    if PLOT:
        cv2.circle(frame_0, (int(e1[0]/e1[2]),int(e1[1]/e1[2])), 3, (0,0,255), 2)
        cv2.circle(frame_30, (int(e2[0]/e2[2]),int(e2[1]/e2[2])), 3, (0,0,255), 2)
        cv2.imshow('Epipoles F0', frame_0)
        cv2.imshow('Epipoles F30', frame_30)  
    
    return F, H, correspondences, best_inlier_cor


def task_3(F, K, H, correspondences, inlier_cor):
    
    print('-'*50+'\n\t\t\t\tTask 3\n'+'-'*50)
    
    'Task 3_A'
    E = np.matmul(K.T, np.matmul(F, K))
    print("Essential Matrix:\n", E)
    U, S, V = np.linalg.svd(E)

    print("Singular values: ", S)
    
    if np.linalg.det(U) < 0:
        U[:,2] *= -1
    if np.linalg.det(V) < 0:
        V[2,:] *= -1

 
    
    'Task 3_B'
    
    W = np.array([[0, -1, 0],
                 [1, 0, 0],
                 [0, 0, 1]])
    Z = np.array([[0, 1, 0],
                 [-1, 0, 0],
                 [0, 0, 0]])
    
    
    d = 50/(60*60)      #Distance per frame
    B = d*30            #Distance betwenn two cameras
    
    UZU = np.matmul(U, np.matmul(Z, U.T))

    R_t = np.array([[-UZU[1][2]],
                      [UZU[0][2]],
                      [UZU[1][0]]])

    R_1 = np.matmul(U, np.matmul(W, V.T))
    R_2 = np.matmul(U, np.matmul(W.T, V.T))

    
    t_1 = B*np.matmul(np.linalg.inv(R_1), R_t)
    t_2 = -B*np.matmul(np.linalg.inv(R_1), R_t)
    t_3 = B*np.matmul(np.linalg.inv(R_2), R_t)
    t_4 = -B*np.matmul(np.linalg.inv(R_2), R_t)
    
    ts = [t_1, t_2, t_3, t_4]
    
    
    'Task 3_C'
    

    print('Rts:')
    for t in ts:
        print(t)
        



    
def main():
    p0, i, t, f, k = task_1()
    F, H, c, ic  = task_2(p0, i, t, f)
    task_3(F, k, H, c, ic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 16:39:38 2021

@author: Shane
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import functools
import pandas as pd
import copy

"Set PLOT to True to view images and plots generated from the program"
PLOT = True

def task1(img):
    """
    Scale Space
    A. Download the input image file Assignment_MV_1_image.png from Canvas. Load the
        file and convert it into a single channel grey value image. Make sure the data type is
        float32 to avoid any rounding errors. [2 points]
    B. Create twelve Gaussian smoothing kernels with increasing scales σ∈{2k2⁄ | k=0,…,11}
        and plot each of these kernels as image [7 points]. Make sure that the window size is large enough to sufficiently 
        capture the characteristic of the Gaussian. Apply these kernels to the input image to create a scale

    Parameters
    ----------
    img : IMAGE
        Gray Scale Image for processing.

    Returns
    -------
    ss : DICT
        Dictionary holding scale space images referenced by sigma value.
    sigs : LIST
        Sigma values used to ref ss above.

    """
    
    'Task 1.B'
    #Gaussian Smoothing Kernels indexed by sigma
    gsk = {}
    #Scale Space dicttionary indexed by sigma
    ss = {}
    #Sigmas
    sigs = []
    
    for k in range(12):
        sigma = 2**(k/2)
        sigs.append(sigma)
        x, y = np.meshgrid(np.arange(-3*sigma, 3*sigma), 
                           np.arange(-3*sigma, 3*sigma))
        
        #Gaussian Smoothing kernels 
        gsk[sigma] = 1*np.exp(-(x**2 + y**2)/(2*sigma**2))/(2*np.pi*sigma**2)
        #Filter our grayscale image with each Gaussian Smoothing Kernel to give scale space representation
        ss[sigma] = cv2.filter2D(img, -1, gsk[sigma])
        
        #If the global PLOT is true, plot gaussian smoothing kernels and show scale space images. Uncomment cv2.imwrite to save Scale Space Images. 
        if PLOT:
            cv2.imshow('Scale Space Image, Sigma: ' + str(sigma), ss[sigma]/255)
            # cv2.imwrite(r'C:\Users\Shane\Documents\College\Machine Vision\Assignments\Assignment 1\ss ' + str(sigma) + ".png", ss[sigma])
            plt.figure()
            plt.imshow(gsk[sigma])
            plt.title("Gausian Smoothing Kernal, Sigma= "+ str(sigma))

    return ss, sigs





   
def task2(ss, sigs, img):
    """
    Feature Point Locations 
    A. Use the scale-space representation from task 1 to calculate difference of Gaussian images at all scales where this is possible. Display all resulting DoG images [5 points].
    B. Find key-points by thresholding all DoGs from subtask A. Use a threshold of T=10 and suppress non-maxima in scale-space by making sure that the key-points have no neighbours, 
        both in space as well as in scale, where the value is higher [3 points]. The resulting key-points should comprise three coordinates (x,y,σ), two spatial and the scale at which they were detected.
    C. Visualise the key-point locations and their scales in the input image by drawing a circle of radius 3𝜎 around every key-point [2 points].

    Parameters
    ----------
    ss : DICT
        Dictionary holding scale space images referenced by sigma value.
    sigs : LIST
        Sigma values used to ref ss above.
    img : IMAGE
        Image for processing in part C.

    Returns
    -------
    kps_dict : DICT
        Key Points referenced by co-ordinates x, y.

    """
    
    'Task 2.A'
    
    #Difference of Gaussians dictionary (Refernenced by i (0 - 11))
    DoG = {}

    for i in range(11):
        DoG[i] = ss[sigs[i+1]]-ss[sigs[i]]
        if PLOT:
            cv2.imshow('DoG, Sigma = ' + str(sigs[i+1]) + str(sigs[i]), DoG[i]/255)
            # cv2.imwrite(r'C:\Users\Shane\Documents\College\Machine Vision\Assignments\Assignment 1\DoG ' + str(i) + ".png", DoG[i])
            
   
    'Task 2.B' 
   
   #All key points for each scale space image or stored as a list referenced by sigma in this dict
    kps_dict = {}
    for i in range(11):
        mask = DoG[i]>10
        kps = []
        for x in range(2, DoG[i].shape[0]-2):
            for y in range(2, DoG[i].shape[1]-2):
                if mask[x, y]: 
                    #Is it biggest in neighbourhood
                    val = DoG[i][x,y]
                    still_max = True
                    if i > 0:
                        #Check Scale Space Image below key point
                        still_max = n_max_sup(DoG[i-1], val, x, y)
                    if still_max:  
                        #Check Neighbourhood on same scale space as key point
                        still_max = n_max_sup(DoG[i], val, x, y)
                    if i < 10 and still_max:
                        #Check Scale Space Image above key point
                        still_max = n_max_sup(DoG[i+1], val, x, y)
                    if still_max:
                        points = (x, y)
                        if x > 2*(9/2)*sigs[i] and y > 2*(9/2)*sigs[i] and x < (ss[sigs[0]].shape[0]-1)-2*(9/2)*sigs[i] and y < (ss[sigs[0]].shape[1]-1)-2*(9/2)*sigs[i] :
                            #Ignore key points too close to edges ahead of task3 (sigma dependant)
                            if x > 18 and y > 18 and x < (ss[sigs[0]].shape[0]-18) and y < (ss[sigs[0]].shape[1]-1)-18 :
                                #Ignore key points too close to edge ahead of task4 (taking into account smaller sigmas)
                                kps.append(points)              
                                
        kps_dict[sigs[i]] = kps         #Save all KPs for this scale space image
        
    
    'Task 2.C'
    color = (0, 0, 255)
    #Print circles around key points
    for sigma in sigs[:-1]:
        radius = round(3 * sigma)
        for y, x in kps_dict[sigma]:    
            co_ords = (x, y)
            img = cv2.circle(img, co_ords, radius, color)
    
    if PLOT:
        cv2.imshow('Task 1: Keypoints w/ circles', img)
        # cv2.imwrite(r'C:\Users\Shane\Documents\College\Machine Vision\Assignments\Assignment 1\Key Points.png', img)
        
    return kps_dict  


def n_max_sup(dog, val, x, y):
    #Returns True if max in 4x4 pixel neighbourhood 
    if ((val>=dog[x,y]) and 
        (val>dog[x-1,y-1]) and
        (val>dog[x-2,y-1]) and
        (val>dog[x-1,y-2]) and
        (val>dog[x-2,y-2]) and
        (val>dog[x,y-2]) and
        (val>dog[x-2,y]) and
        (val>dog[x-2,y+1]) and
        (val>dog[x-2,y+2]) and
        (val>dog[x-1,y+2]) and         
        (val>dog[x,y+2]) and
        (val>dog[x+2,y]) and
        (val>dog[x+1,y+2]) and
        (val>dog[x+2,y+2]) and
        (val>dog[x+2,y+1]) and
        (val>dog[x+2,y-1]) and
        (val>dog[x+1,y-2]) and
        (val>dog[x+2,y-2]) and
        (val>dog[x-1,y]) and
        (val>dog[x-1,y+1]) and
        (val>dog[x,y-1]) and
        (val>dog[x,y+1]) and
        (val>dog[x+1,y-1]) and
        (val>dog[x+1,y]) and
        (val>dog[x+1,y+1])):
            return True
    return False



def task3(ss, sigs, kps, img):
    """
    Feature point orientations
    A. Calculate derivatives of all scale-space images from task 1 
        Display the resulting derivative images 𝑔𝑥 and 𝑔𝑦 at all scales [3 points].
    B. For each key-point (𝑥,𝑦) consider the 7×7 grid of points sampled at a distance of ±92𝜎 around its location 
        and calculate the gradient lengths and gradient directions for each point on this grid [6 points]. Make sure to use the 
        appropriate scale σ and the correct gradient images 𝑔𝑥 and 𝑔𝑦. Use nearest neighbour interpolation to sample the gradient grid.
    C. Calculate a Gaussian weighting function for each of the 7×7 grid points [2 point]. Then create a 36-bin orientation histogram 
        vector h and accumulate the weighted gradient lengths wqrmqr for each grid point (q,r) where the gradient direction θqr falls into this particular bin [3 points], i.e. calculate for each 10𝑜 range −18≤𝑖<18 of potential direction the weighted contributions that fall into this range ℎ𝑖=Σwqrmqr𝑖≤ 36 𝜃𝑞𝑟 2𝜋<𝑖+1
        Use the maximum of this orientation histogram ℎ to determine the dominant orientation 
        of the key-point [1 point]. Each key-point is now characterised by its location, its scale, and its dominant orientation 
    D. Visualise the orientation of all key-points by drawing a circle with radius 3σ and a line from the key-point 
        centre to the circle radius, which indicate the orientation.

    Parameters
    ----------
    ss : DICT
        Dictionary holding scale space images referenced by sigma value.
    sigs : LIST
        Sigma values used to ref ss above.
    kps : DICT
        Key Points referenced by sigma (stored key points as list for each scale space image.
    img : IMAGE
        Image for processing in Part D.

    Returns
    -------
    d_o : DICT
        Dominant Orientations.
    dg_x : DICT
        Derivative of Gaussian X.
    dg_y : DICT
        Derivative of Gaussian Y.
    """
    
    'Task 3.A'
    #Gaussian Kernels as defined in the assignment specification
    gaussian_kernel_x = np.array([[1, 0, -1]])
    gaussian_kernel_y = np.transpose(gaussian_kernel_x)
    
    #Dictionaries to hold derivative of gaussian images - referenced by sigma
    dg_x = {}
    dg_y = {}      
            
    for i in range (11):
        sigma_d = sigs[i]  
        #Get derivative of gaussian images by filtering scale space imgs with gaussian kernels
        dg_x[sigs[i]] = cv2.filter2D(ss[sigs[i]], -1, gaussian_kernel_x)
        dg_y[sigs[i]] = cv2.filter2D(ss[sigs[i]], -1, gaussian_kernel_y)  
        
        if PLOT:
            #Show resulting images
            cv2.imshow("Deriviative of Gaussian X, Sigma = "+str(sigs[i]), dg_x[sigs[i]])
            cv2.imshow("Deriviative of Gaussian Y, Sigma = "+str(sigs[i]), dg_y[sigs[i]])
            # cv2.imwrite(r'C:\Users\Shane\Documents\College\Machine Vision\Assignments\Assignment 1\gx' + str(sigs[i])+'.png', dg_x[sigs[i]]*255)
            # cv2.imwrite(r'C:\Users\Shane\Documents\College\Machine Vision\Assignments\Assignment 1\gy' + str(sigs[i])+'.png', dg_y[sigs[i]]*255)
       
    color = (0, 0, 255)
    sigma = sigs[0]
    count = 0 
    m_qr = {}       #Gradient Lenghts, key = KP co-ordinates (x, y)
    theta_qr = {}   #Gradient directions, key = KP co-ordinates (x, y)
    w_qr = {}       #Gaussian weights, keys = KP co-ordinates (x, y)         
    d_o = {}        #Dominant Orientation, key = KP co-ordinates (x, y)
    h = {}          #Holds list of dominent orientation bins 0 - 35 
    
    #32 bins template
    bins = [] 
    for i in range(-18, 18):
        bins.append(0)
    
    'Task 3.B'
    for sigma in sigs[:-1]:
        qr=[]
        for i in range(-9, 12, 3):
            #Save q & r reference from point of interest in list
            qr.append((i/2)*sigma)     

        for x, y in kps[sigma]: 
            #Iterate through key points for a given sigma
            m_qr[x, y] = {}
            theta_qr[x, y] = {}
            w_qr[x, y] = {}
            h[x, y] = copy.deepcopy(bins)
            for q in qr:
                for r in qr:
                    #Iterate through neighbourhood and find gradient lenghts, directions and weights
                    m_qr[x, y][q,r] = np.sqrt(dg_x[sigma][round(x+q),round(y+r)]**2 + dg_y[sigma][round(x+q),round(y+r)]**2)
                    theta_qr[x, y][q, r] = np.arctan2(dg_y[sigma][round(x+q),round(y+r)], dg_x[sigma][round(x+q),round(y+r)])
                    w_qr[x, y][q, r] = 1/((9*np.pi*sigma**2)/2)*np.exp(-(q**2+r**2)/((9*sigma**2)/2))
            
            
            'Task 3.C'
            for q in qr:
                for r in qr:
                    #Iterate through neighbourhood again and find dominant orientations, by placing in bins
                    h_i = np.sum(m_qr[x, y][q,r]*w_qr[x, y][q, r])      
                    thet_b = 36*(theta_qr[x,y][q,r]-(min(theta_qr[x, y].values())))/(max(theta_qr[x, y].values())
                                                                                    -min(theta_qr[x, y].values()))
                    if thet_b == 36:
                        #Maximum bin index is 35
                        thet_b = 35
                        
                    h[x,y][math.floor(thet_b)] += h_i         #Bins are offset by 18, ranging from 0 - 35. refernced by index where 0 = -18
            
            d_o[x, y] = h[x,y].index(max(h[x,y]))*10 + 5            
        

    'Task 3.D'
    for sigma in sigs[:-1]:
        radius = round(3 * sigma)
        for x, y in kps[sigma]:   
            #Iterate through Key Points and draw circle and dominant orientation on image
            co_ords = (y, x)
            img = cv2.circle(img, co_ords, radius, color)
            if d_o[x, y] < 90:
                #Find location on circle where dominant direction line terminates using hypotenuse (know angle and radius, must find pixels R/L and U/D)
                angle = d_o[x,y]
                vert = math.sin(angle)*radius
                horz = - math.cos(angle)*radius
            elif d_o[x,y] < 180:
                angle = 90 - d_o[x,y]
                vert = math.cos(angle)*radius
                horz = math.sin(angle)*radius
            elif d_o[x,y] < 270:
                angle = 180 - d_o[x,y]
                vert = - math.sin(angle)*radius
                horz = math.cos(angle)*radius
            else:
                angle = 270 - d_o[x,y]
                vert = - math.cos(angle)*radius
                horz = - math.sin(angle)*radius
            
            end_point = (y + round(vert), x + round(horz))
            img = cv2.line(img, co_ords, end_point, color)      

    if PLOT:
        cv2.imshow('Task 3: Dominant Orientations', img)
        # cv2.imwrite(r'C:\Users\Shane\Documents\College\Machine Vision\Assignments\Assignment 1\Dominant Orientations.png', img)

        
    return d_o, dg_x, dg_y




def task4(ss, sigs, kps, d_o, dg_x, dg_y):
    """
    Feature Descriptors
    The SIFT algorithm also proposed to calculate a 128-vector that describes the local distribution of gradient directions relative 
    to the dominant direction for each key point. This descriptor can be used for feature matching or as scale and rotation invariant 
    image descriptor. Similar to the orientation calculation we look at a 16×16 grid around each key-point location (𝑥,𝑦) now, again 
    taking the scale into consideration. The 16×16 grid is sub-divided into 4×4 grids of size 4×4 each (see figure on the right).
    A. The grid coordinates relative to the key-point location for each of the 4×4 sub-grids (𝑖,𝑗)∈{−2,…,1}×{−2,…,1} covering an area 
        of ±92𝜎 are given by (𝑠,𝑡)𝑖𝑗∈{916(𝑘+12)𝜎 | 𝑘=4𝑖,..,4𝑖+3}×{916(𝑘+12)𝜎 | 𝑘=4𝑗,..,4𝑗+3}
        Calculate for each of these coordinates the Gaussian weighting function as well as the gradient lengths and gradient directions
        adjusted by the dominant direction 𝜃̂ around each key-point [8 points]. Make sure to use the appropriate scale σ and the 
        correct gradient images 𝑔𝑥 and 𝑔𝑦. Use nearest neighbour interpolation to sample the gradient grid.
    B. Now create a 8-bin orientation histogram vector hij for each of the 4×4 sub-grids (𝑖,𝑗)∈{−2,…,1}×{−2,…,1} and accumulate the 
        weighted gradient lengths wstmst for each grid point (s,t) within the sub-grid where the adjusted gradient direction 
        θst−𝜃̂ falls into this particular bin [3 points].
    C. Concatenate all these 16 histogram 8-vectors into a single 128-vector 𝑑 describing the feature at the key-point. 
        Normalise this descriptor vector dividing it by its length 𝑑√𝑑𝑇𝑑 and compute the final descriptor vector for each key-point by 
        capping the maximum value of the vector at 0.2 [1 point].

    Parameters
    ----------
    ss : DICT
        Dictionary holding scale space images referenced by sigma value.
    sigs : LIST
        Sigma values used to ref ss above.
    kps : DICT
        Key Points, holds lists of key points from a given scale (referenced by sigma).
    d_o : DICT
        Dominant Orientations.
    dg_x : DICT
        Derivative of Gaussian X images.
    dg_y : DICT
        Derivative of Gaussian Y images.

    Returns
    -------
    d : DICT
        128-Vectors describing the feature at each key point.

    """
    
    m_st = {}       #Gradient Lenghts at location st
    theta_st = {}   #Gradient Direction at st
    w_st = {}       #Weight at st
    h_ij = {}       #Keypoint Histogram
    d = {}          #128-Vectors describing the feature at each key point.
    
    #Create Bins template
    bins = [] 
    for i in range(0, 8):
        bins.append(0)

    for sigma in sigs[:-1]:
        
        ij_tmp = []
        st_tmp = []
        increments = (9/2*sigma)/4
        b = 0
        for a in range(-2, 2):
            #Array of 4x4 grid, locations with reference to Key Point
            ij_tmp.append(a*(9/2)*sigma) 
            st_tmp.append(increments*b)
            b += 1
            
        st = []   
        ij = [] 
        
        for a in range(4):
            for b in range(4):
                #Create array of 4x4 grid, locations with reference to bottom left point on original (larger) 4x4 grid
                ij.append((ij_tmp[a],ij_tmp[b]))
                st.append((st_tmp[a], st_tmp[b]))  
        
        
        'Task 4.A'        
        for x, y in kps[sigma]: 
            #Iterate through key points
            m_st[x, y] = {}
            theta_st[x, y] = {}
            w_st[x, y] = {}
            h_ij[x, y] = {}
            d[x, y] = {}
            first = True
            
            for i, j in ij:
                #Iterate through 4x4 grid
                h_ij[x, y][i, j] = copy.deepcopy(bins)
                m_st[x, y][i, j] = {}
                theta_st[x, y][i, j] = {}
                w_st[x, y][i, j] = {}  
                for s, t in st:
                    #Iterate through 4x4 grid within larger grid
                    s = s+i
                    t = t+j 
                    m_st[x, y][i, j][s,t] = np.sqrt(dg_x[sigma][round(x+s),round(y+t)]**2 + dg_y[sigma][round(x+s),round(y+t)]**2)
                    theta_st[x, y][i, j][s, t] = np.arctan2(dg_y[sigma][round(x+s),round(y+t)], dg_x[sigma][round(x+s),round(y+t)]) - d_o[x, y]        
                    w_st[x, y][i, j][s, t] = 1/((81*np.pi*sigma**2)/2)*np.exp(-(s**2+t**2)/((81*sigma**2)/2))
                    
                    'Task 4.B'
                    h_i = np.sum(m_st[x, y][i,j][s,t]*w_st[x, y][i, j][s,t])
                    thet_b = 8*(theta_st[x,y][i,j][s,t]-(min(theta_st[x, y][i,j].values())))/(max(theta_st[x, y][i,j].values())
                                                                                    -min(theta_st[x, y][i,j].values()))    
                    if thet_b == 8:
                        #Max value is 8, put in max bin (7)
                        thet_b = 7
                    
                    try:
                        #Encountered issue where due to a rounding error thet_b was = NaN throwing a value error here as we find the .floor of it
                        h_ij[x,y][i,j][math.floor(thet_b)] += h_i
                    except ValueError:
                        thet_b = 0
                        h_ij[x,y][i,j][math.floor(thet_b)] += h_i
            
                if first:
                    d[x,y] = []
                    first = False
                    
                for q in h_ij[x,y][i,j]:
                    d[x,y].append(q)
            
            'Task 4.C'
            #Normalise Vector
            d[x,y] = np.array([d[x,y]])      
            d[x,y] = ((1/np.sqrt(np.dot(d[x,y], d[x,y].T)))*d[x,y]).T
            #Capp the max value to 0.2
            d[x,y] = np.clip(d[x,y], None, 0.2)
    
    return d
    




def exec_time(func):
    """
    Generic Execution time recorder, pass in function. Records execution time using decorators

    Parameters
    ----------
    func : FUNCTION
        Function .


    """
    
    @functools.wraps(func)
    def record_exec_time(*args, **kwargs):
        start_time = time.perf_counter()
        mn = func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time
        print("Execution Time: ", execution_time)
        return mn

    return record_exec_time

@exec_time        
def main():
    """
    Main Program

    Returns
    -------
    None.

    """
    
    'Task A.A'
    input_img = cv2.imread("Assignment_MV_1_image.png")  
    img = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    img = np.array(img, dtype = 'float32') 
    scale_space, sigmas = task1(img)
    key_points = task2(scale_space, sigmas, input_img)
    dominant_orientations, derivative_of_gaussian_x, derivative_of_gaussian_y = task3(scale_space, sigmas, key_points, input_img)
    # descriptor_vector = task4(scale_space, sigmas, key_points, dominant_orientations, derivative_of_gaussian_x, derivative_of_gaussian_y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    main()
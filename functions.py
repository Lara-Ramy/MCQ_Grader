import xlwt
import os
from commonfunctions import *
import numpy as np
import pandas as pd
import cv2
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.morphology import binary_erosion, binary_dilation, binary_closing,skeletonize, thin
from skimage.measure import find_contours
from skimage.draw import rectangle
from skimage.transform import resize


# getting binary image
def getBinaryImg(img, thresh):
    binary_img = img.copy()
    binary_img[binary_img > thresh] = 1
    binary_img[binary_img <= thresh] = 0
    return binary_img

# getting closed image
def getClosedImg(img, win_size):
    closed_img = binary_dilation(binary_erosion(img, footprint=win_size), footprint=win_size)
    return closed_img

# getting contour according to aspect ratio and area
def getAspectRatioContour(img, thresh = 0.7, win_size =  np.ones((2, 2), dtype=int), lower_aspect = 6, upper_aspect = 10):
    #image preprocessing
    binary_img = getBinaryImg(img, thresh)
    closed_img = getClosedImg(binary_img, win_size)

    contour = find_contours(closed_img, 0.8)
    bounding_rects = []
    
    for rect in contour:
        x_min = rect[:, 1].min()
        x_max = rect[:, 1].max()
        y_min = rect[:, 0].min()
        y_max = rect[:, 0].max()
        
        aspect_ratio = (x_max - x_min) / (y_max - y_min) 
        area = (x_max - x_min) * (y_max - y_min) 
        if (lower_aspect < aspect_ratio < upper_aspect) and (300 < area < 9000):
            bounding_rects.append([int(x_min), int(x_max), int(y_min), int(y_max)])
    bounding_rects = sorted(bounding_rects, key = lambda item: (item[2] + 2*item[0]))
    return bounding_rects

# getting contours and sorting them by area
def getRectContour(img, thresh = 0.7, win_size =  np.ones((2, 2), dtype=int), inverse = 0):
    #image preprocessing
    binary_img = getBinaryImg(img, thresh)
    closed_img = getClosedImg(binary_img, win_size)
    
    contour = find_contours(closed_img, 0.8)
    bounding_rects = []
    area = []
    
    for rect in contour:
        x_min = rect[:, 1].min()
        x_max = rect[:, 1].max()
        y_min = rect[:, 0].min()
        y_max = rect[:, 0].max()
        
        area = (x_max - x_min)*(y_max - y_min)
        aspect_ratio = (x_max - x_min) / (y_max - y_min) 
        if aspect_ratio > 1:
            bounding_rects.append([int(area),int(x_min), int(x_max), int(y_min), int(y_max)])
    
    bounding_rects = sorted(bounding_rects, reverse = True)
    return bounding_rects


#getting contours using cv library
def getCVRectContour(img, ut = 150,lt = 100,kernel = 3, win_size =  np.ones((3, 3), dtype=int)):
    #image preprocessing
    img_blur = cv2.GaussianBlur(img, (kernel, kernel), 7)
    img_canny = cv2.Canny(img_blur,lt,ut) 
    img_dial = cv2.dilate(img_canny, win_size, iterations=2)
    img_closed = cv2.erode(img_dial, win_size, iterations=1)

    contours, heirarchy = cv2.findContours(img_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_rects = []
    
    for rect in contours:
        area = cv2.contourArea(rect)
        #print(area)
        perimeter = cv2.arcLength(rect, True)
        points = cv2.approxPolyDP(rect, 0.02*perimeter, True)
        #print(points)
        
        if (len(points) == 4) and (area > 50):
            bounding_rects.append(rect)
            
    bounding_rects = sorted(bounding_rects, key = cv2.contourArea, reverse = True)    
    
    return bounding_rects

#getting corner points for getAspectRatioContour function
def getAspectCornerPoints(contour, i,img):
    c = contour[i]

    points = np.array([[c[0], c[2]],
                      [c[1], c[2]],
                      [c[1], c[3]],
                      [c[0], c[3]]])
    #grey_img = img.copy()
    #plt.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], color = "black", linewidth = 1)
    #plt.plot([points[1][0], points[2][0]], [points[1][1], points[2][1]], color = "black", linewidth = 1)
    #plt.plot([points[2][0], points[3][0]], [points[2][1], points[3][1]], color = "black", linewidth = 1)
    #plt.plot([points[3][0], points[0][0]], [points[3][1], points[0][1]], color = "black", linewidth = 1)
    #plt.imshow(grey_img)
    #plt.show()
    return points

#getting corner points for getCVRectContour function
def getCVCornerPoints(contour):
    perimeter = cv2.arcLength(contour, True)
    points = cv2.approxPolyDP(contour, 0.02*perimeter, True)
    return points

# getting corner points for getRectContour function
def getCornerPoints(contour, i,img):
    c = contour[i][1:5]
    #print(c)
    points = np.array([[c[0], c[2]],
                      [c[1], c[2]],
                      [c[1], c[3]],
                      [c[0], c[3]]])

    grey_img = img.copy()
    #plt.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], color = "black", linewidth = 1)
    #plt.plot([points[1][0], points[2][0]], [points[1][1], points[2][1]], color = "black", linewidth = 1)
    #plt.plot([points[2][0], points[3][0]], [points[2][1], points[3][1]], color = "black", linewidth = 1)
    #plt.plot([points[3][0], points[0][0]], [points[3][1], points[0][1]], color = "black", linewidth = 1)
    #plt.imshow(grey_img)
    #plt.show()
    #show_images([grey_img])
    #print(x1)
    return points

#getordered corner points
def getOrderedCornerPoints(points):
    points = points.reshape((4,2))
    points_ordered = np.zeros((4,1,2), np.int32)
    s = points.sum(1)
    diff = np.diff(points, axis = 1)
    points_ordered[0] = points[np.argmin(s)]
    points_ordered[1] = points[np.argmin(diff)]
    points_ordered[2] = points[np.argmax(diff)]
    points_ordered[3] = points[np.argmax(s)]
    return points_ordered


#get only the ID from the Question Paper
def getIDRegion(img, index = 1):
    #image preprocessing
    i = img.copy()
    i[i > 0.6] = 1
    i[i <= 0.6] = 0
    i = i[20:250, 80:560]

    win_size = np.ones((18, 42), dtype=int)
    
    c = getRectContour(i, thresh = 0.7, win_size =  win_size)
    p = getCornerPoints(c, index,i)
    p_ordered = getOrderedCornerPoints(p)
    p2 = np.float32([[0,0], [300,0], [0,100], [300, 100]])
    p1 = np.float32(p_ordered)
    
    ID_img = cv2.getPerspectiveTransform(p1, p2)
    ID_warp = cv2.warpPerspective(i,ID_img, (300, 100))
    
    return ID_warp

# getting the region of each question ordered by number
def getQuestionRegion(img, index = 1, mcq = 4):
    #image preprocessing
    I = img.copy()
    I[I > 0.55] = 1
    I[I <= 0.55] = 0
    #different closing windows for differen number of choices
    if mcq == 2:
        win_size = np.ones((3, 15), dtype=int)
        c = getAspectRatioContour(I, thresh = 0.5, win_size =  win_size, lower_aspect = 3.5, upper_aspect = 4.5)
    elif mcq == 3:
        win_size = np.ones((4, 20), dtype=int)
        c = getAspectRatioContour(I, thresh = 0.5, win_size =  win_size, lower_aspect = 5.5, upper_aspect = 7)
    elif mcq == 4:
        win_size = np.ones((2, 20), dtype=int)
        c = getAspectRatioContour(I, thresh = 0.5, win_size =  win_size, lower_aspect = 7, upper_aspect = 10)
    elif mcq == 5:
        win_size = np.ones((4, 20), dtype=int)
        c = getAspectRatioContour(I, thresh = 0.5, win_size =  win_size, lower_aspect = 10, upper_aspect = 20)
    #print(c)
    
    p = getAspectCornerPoints(c, index, I)
    
    p_ordered = getOrderedCornerPoints(p)
    
    p2 = np.float32([[0,0], [300,0], [0,100], [300, 100]])
    p1 = np.float32(p_ordered)
    
    Q_img = cv2.getPerspectiveTransform(p1, p2)
    Q_warp = cv2.warpPerspective(I,Q_img, (300, 100))
  
    #show_images([Q_warp])
    return Q_warp

#splitting rows of the ID
def splitrows(img, rowsnum):
    rows=[]
    Rsections=(100/(rowsnum+1))
    extra=Rsections/3
    extra=int(extra)
    Rsections=int(Rsections)
    Rbegin=int(Rsections+extra)
    Rend=int(Rbegin+Rsections)
    
    for i in range (rowsnum):
        rows.insert(i,img[Rbegin:Rend][:])
        Rbegin=Rend
        Rend=Rbegin+Rsections
    return rows

#splitting columns of the ID
def splitcolumns(img):
    columns=[]
    sections=30
    begin=0
    end=begin+sections
    
    for i in range (10):
        columns.insert(i,img[:,begin:end])
        begin=end
        end=begin+sections
    return columns

#getting individual bubbles of the ID
def getbubble(img,rowsnum):
    bubbles=[]
    j=0
    k=0
    for i in range (rowsnum*10):
        IDrows = splitrows(img,rowsnum)
        IDrow = IDrows[j]
        IDcolumns = splitcolumns(IDrow)
        IDcolumn = IDcolumns[k]
        bubbles.insert(i,IDcolumn)
        k=k+1
        if (k>9):
            j=j+1
            k=0
    return bubbles

#getting the ID as an array of number
def getID(rowsnum, IDbubbles):
    countC=0
    countR=0
    errorID=0
    sumarray=[]
    avgid=[]
    errorarr=[]
    sumID=0
    i=0
    pixelval=np.zeros((rowsnum,10))
    for images in IDbubbles:
        totalpixels=cv2.countNonZero(images)
        pixelval[countR][countC]=totalpixels
        sumID=sumID+totalpixels
        countC+=1
        if(countC == 10):
            sumarray.insert(i,sumID)
            sumID=0
            countR+=1
            countC=0
            i+=1
    
    for i in range (rowsnum):
        avgid.insert(i,int(sumarray[i]/10))
        errorarr.insert(i,int(sumarray[i]/120))
    
    #print(pixelval)
    #print(avgid)
    
    for j in range (rowsnum):
        for i in range (10):
            if(pixelval[j][i]<(avgid[j]-errorarr[j])):
                errorID=errorID+1
                

    
    arrindex=[]
    index=[]
    
    for x in range (0,rowsnum):
        arr=pixelval[x]
        index=np.where(arr==np.amin(arr))
        arrindex.append(index[0][0])
        
    if errorID>(rowsnum):
        arrindex = "wrong id"

    return arrindex

#converting the array into a number
def convert(list):
    if list != "wrong id":
        s = [str(i) for i in list]
        res = int("".join(s))
    else:
        res = "wrong id"
    return(res)

#getting each bubble of the questions
def splitQcolumns(img, colsnum):
    cols=[]
    Csections=(300/(colsnum))
    Csections=int(Csections)
    Cbegin=0
    Cend=Cbegin+Csections
    
    for i in range (colsnum):
        cols.insert(i,img[:,Cbegin:Cend])
        Cbegin=Cend
        Cend=Cbegin+Csections
    return cols

ID_1 = [2,1,2,2,1,1,2,1,1,1,2,2,1,1,2]
ID_2 = [1,1,1,1,1,1,0,1]
ID_3 = [1,1]
ID_4 = [1,1,1,1,1,1,1,1,1,2,2,1,1]

#getting 1 for correct answer and 0 
def getQanswer(colsnum,columns):   
    countC=0
    countR=0
    error=0
    summ=0
    errorarr = [50]
    pixelval=np.zeros((1,colsnum))
    for images in columns:
        totalpixels=cv2.countNonZero(images)
        summ=summ+totalpixels
        pixelval[countR][countC]=totalpixels
        countC+=1
        if(countC == 10):countR+=1;countC=0
    
    avg=summ/colsnum
    
    for i in range (colsnum):
        if(pixelval[0][i]<(avg-200)):
            error=error+1
    
    arrindex=[]
    index=[]
    for x in range (0,1):
        arr=pixelval[x]
        index=np.where(arr==np.amin(arr))
        arrindex.append(index[0][0])
    
    if error>1:
        return(errorarr)
    else:
        return(arrindex)
#getting an array of the    
def getScore(mcq,quesnum,img, list2):
    score=0
    quesarr=[]
    for i in range (quesnum): 
        ques=getQuestionRegion(img, index = i,mcq = mcq)
        columns=splitQcolumns(ques,mcq)
        answer1=getQanswer(mcq,columns)
        answer=answer1[0]+1
        if(answer==list2[i]):
            score=score+1
            quesarr.insert(i,1)
        else:
            quesarr.insert(i,0)
    return quesarr, score








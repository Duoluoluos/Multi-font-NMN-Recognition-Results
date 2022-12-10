# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 19:48:06 2022

@author: 12543
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import traceback
import time
import seaborn as sns
import pretty_midi
import os

def get_roi(img,x,y,w,h):
    return img[y:y+h+1,x:x+w+1]

def display(img,x,y,w,h,save=False,show=False):
    if show== True:
        plt.figure()
        plt.imshow(img[y:y+h+1,x:x+w+1],cmap='gray')
        plt.xticks([]),plt.yticks([])
    if save==True:
        cv.imwrite('pattern library\\Font 1\\{}.png'.format(str(x)),img[y:y+h+1,x:x+w+1])


def get_patterns(filepath):
    lt=[]
    for name in os.listdir(filepath):
        gray=cv.cvtColor(cv.imread(filepath+'/'+name),cv.COLOR_BGR2GRAY)
        lt.append(gray)
    return lt

def pattern_match(img,pd,font="Font 1"):    
    patterns = get_patterns("pattern library\\" + font)
    for (gx,gy,gw,gh),obj in pd.items():
        # 用于存放识别到的数字
        digit_out = []
        img_digit = get_roi(img,gx,gy,gw,gh)
        # 二值化处理
        img_thresh = cv.threshold(
        img_digit, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]        
        source=[]
        for pattern in patterns:
            SA = np.zeros((max(img_thresh.shape[0],pattern.shape[0]),max(img_thresh.shape[1],pattern.shape[1])),np.uint8)
            H,W=img_thresh.shape
            SA[0:H,0:W] = img_thresh
            PA = cv.threshold(pattern, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
            res = cv.matchTemplate(PA, SA, cv.TM_CCOEFF_NORMED)
            max_val = cv.minMaxLoc(res)[1]
            source.append(max_val)
        digit_out.append(str(source.index(max(source))))
        pd[(gx,gy,gw,gh)]=eval(digit_out[0])
        
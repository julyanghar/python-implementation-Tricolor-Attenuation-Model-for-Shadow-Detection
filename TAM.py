import cv2
import numpy as np
import os
import math

def watershed(img:np.ndarray)->np.ndarray:
    return None

def get_Gray_pixel_value(B:float,G:float,R:float)->float:
    value = max(R,G,B)/(min(R,G,B)+1)
    if value == 0:
        value = float('inf')
    else:
        value = math.log(value)
    return value

def RGB2GRAY(img:np.ndarray)->np.ndarray:
    height = img.shape[0]
    width = img.shape[1]
    gray_img = np.zeros(shape=(height,width))
    for i in range(height):
        for j in range(width):
            gray_img[i][j] = get_Gray_pixel_value(float(img[i][j][0]),float(img[i][j][1]),float(img[i][j][2]))

    min_value = np.min(gray_img)
    for i in range(height):
        for j in range(width):
            if gray_img[i][j] == float('inf'):
                gray_img[i][j] = min_value

    result = cv2.normalize(src=gray_img, dst=gray_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                           dtype=cv2.CV_8UC1)
    return result


def get_TAMimages(img:np.ndarray)->np.ndarray:
    m = 1.31
    n = 1.19
    B,G,R = cv2.split(img)

    F_R = np.mean(R)
    F_G = np.mean(G)
    F_B = np.mean(B)
    TAMimg = np.zeros(shape=(img.shape[0],img.shape[1]))

    TAMvector = [m*F_R/F_B, n*F_G/F_B, 1]

    if(TAMvector[0] > TAMvector[1] and TAMvector[1] > TAMvector[2]):
        TAMimg = R - B
    if (TAMvector[0] > TAMvector[2] and TAMvector[2] > TAMvector[1]):
        TAMimg = R - G
    if (TAMvector[1] > TAMvector[0] and TAMvector[0] > TAMvector[2]):
        TAMimg = G - B
    if (TAMvector[1] > TAMvector[2] and TAMvector[2] > TAMvector[0]):
        TAMimg = G - R
    if (TAMvector[2] > TAMvector[0] and TAMvector[0] > TAMvector[1]):
        TAMimg = B - G
    if (TAMvector[2] > TAMvector[1] and TAMvector[1] > TAMvector[0]):
        TAMimg = B - R

    return TAMimg

def get_CombinedImage(img:np.ndarray,TAMimg:np.ndarray)->np.ndarray:
    img_gray = cv2.cvtColor(src=img,code=cv2.COLOR_RGB2GRAY) #img_gray : Y ; TAMimg : X; combinedImg : Z.
    combinedImg = np.zeros(shape=img_gray.shape)
    # initialize parameters
    Zeta_T = -1
    Zeta_T_new = 1
    T = 0   #T : T
    T_new = 0
    alpha = np.mean(img_gray)/np.mean(TAMimg)
    while True:
        combinedImg = alpha*TAMimg+img_gray
        mu = np.mean(combinedImg)
        # This for loop is used to get argmax(Zeta_T)
        for i in range(256):
            T_new = i
            G_T = -(T_new*T_new)+2*mu*T_new
            Zeta_T_temp = G_T*(np.mean(cv2.threshold(src = combinedImg, maxval = 255, thresh = T_new, type = cv2.THRESH_TOZERO)[1])-
                               np.mean(cv2.threshold(src = combinedImg, thresh = T_new,maxval = 255, type = cv2.THRESH_TOZERO_INV)[1]))
            if Zeta_T_temp > Zeta_T_new:
                T_new = i
                Zeta_T_new = Zeta_T_temp
        if Zeta_T_new <= Zeta_T:
            break
        else:
            T = T_new
            Zeta_T = Zeta_T_new

        kappa = (np.mean(cv2.threshold(src = TAMimg, thresh = T_new,maxval = 255, type = cv2.THRESH_TOZERO)[1]) -
                 np.mean(cv2.threshold(src = TAMimg, thresh = T_new,maxval = 255, type = cv2.THRESH_TOZERO_INV)[1])) / np.mean(TAMimg)
        eta = (np.mean(cv2.threshold(src = img_gray, thresh = T_new,maxval = 255, type = cv2.THRESH_TOZERO)[1]) -
                 np.mean(cv2.threshold(src = img_gray, thresh = T_new,maxval = 255, type = cv2.THRESH_TOZERO_INV)[1])) / np.mean(img_gray)
        alpha = np.exp(kappa/eta)

    # normalization
    combinedImg_result = np.zeros(shape=img_gray.shape)
    result = cv2.normalize(src=combinedImg,dst=combinedImg_result,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    return result





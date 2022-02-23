import TAM
import watershed
import numpy as np
import cv2

def get_region_TAM(img:np.ndarray,markers:np.ndarray,mark:int)->np.ndarray:
    rows = img.shape[0]
    colums = img.shape[1]
    TAM_region = np.zeros(shape=(rows,colums))
    img_gray = TAM.RGB2GRAY(img)

    m = 1.31
    n = 1.19
    B, G, R = cv2.split(img)

    for i in range(1,mark):
        coordinates = np.where(markers == i)

        mean_nonshadow = np.mean(img_gray[coordinates])
        coordinates_nonshadow = []

        for i in range(len(coordinates[0])):
            if img_gray[coordinates[0][i]][coordinates[1][i]] >= mean_nonshadow:
                coordinates_nonshadow.append((coordinates[0][i],coordinates[1][i]))

        total_num = len(coordinates_nonshadow)

        F_B = 0
        F_G = 0
        F_R = 0

        for x,y in coordinates_nonshadow:
            F_B += B[x][y]

        for x,y in coordinates_nonshadow:
            F_G += G[x][y]

        for x,y in coordinates_nonshadow:
            F_R += R[x][y]

        F_B = F_B/total_num
        F_G = F_G/total_num
        F_R = F_R/total_num

        if F_B == 0:
            TAMvector = [m * F_R, n * F_G, 0]
        else:
            TAMvector = [m * F_R / F_B, n * F_G / F_B, 1]

        if (TAMvector[0] > TAMvector[1] and TAMvector[1] > TAMvector[2]):
            TAMimg_ = R - B
        if (TAMvector[0] > TAMvector[2] and TAMvector[2] > TAMvector[1]):
            TAMimg_ = R - G
        if (TAMvector[1] > TAMvector[0] and TAMvector[0] > TAMvector[2]):
            TAMimg_ = G - B
        if (TAMvector[1] > TAMvector[2] and TAMvector[2] > TAMvector[0]):
            TAMimg_ = G - R
        if (TAMvector[2] > TAMvector[0] and TAMvector[0] > TAMvector[1]):
            TAMimg_ = B - G
        if (TAMvector[2] > TAMvector[1] and TAMvector[1] > TAMvector[0]):
            TAMimg_ = B - R

        for i in range(len(coordinates[0])):
            x = coordinates[0][i]
            y = coordinates[1][i]
            TAM_region[x][y] = TAMimg_[x][y]

    return TAM_region

def binarize_TAM_region(TAM_region:np.ndarray,markers:np.ndarray,mark:int)->np.ndarray:
    rows = TAM_region.shape[0]
    colums = TAM_region.shape[1]
    res = np.zeros(shape=(rows, colums))

    for i in range(1,mark):
        temp = TAM_region
        coordinates = np.where(markers == i)
        T = np.mean(TAM_region[coordinates])
        temp = np.where(temp > T,255,0)

        for i in range(len(coordinates[0])):
            x = coordinates[0][i]
            y = coordinates[1][i]
            res[x][y] = temp[x][y]

    return res

def rectify_shadow(img:np.ndarray,shadow_img:np.ndarray,markers:np.ndarray,mark:int)->np.ndarray:

    rectified_shadow = shadow_img

    m = 1.31
    n = 1.19
    k1 = 0.8
    k2 = 1.2
    B, G, R = cv2.split(img)
    coordinates_nonshadow_all = np.where(shadow_img == 255)
    coordinates_shadow_all = np.where(shadow_img == 0)

    for i in range(mark):
        # 获得已分割的区域的坐标
        coordinates_region = np.where(markers == i)
        label = False

        # 查找已分割区域的，且是非阴影区域的坐标点
        coordinates_nonshadow_region = find_elements_in_tuples_of_two_lists(coordinates_region,coordinates_nonshadow_all)
        coordinates_shadow_region = find_elements_in_tuples_of_two_lists(coordinates_region,coordinates_shadow_all)

        delta_B = np.mean(B[coordinates_nonshadow_region]) - np.mean(B[coordinates_shadow_region])
        delta_G = np.mean(G[coordinates_nonshadow_region]) - np.mean(G[coordinates_shadow_region])
        delta_R = np.mean(R[coordinates_nonshadow_region]) - np.mean(R[coordinates_shadow_region])

        F_R = np.mean(R[coordinates_nonshadow_region])
        F_G = np.mean(G[coordinates_nonshadow_region])
        F_B = np.mean(B[coordinates_nonshadow_region])

        L = [m*F_R/F_B,n*F_G/F_B,1]*delta_B

        lower = k1*L
        upper = k2*L

        nonshadow_shadow_difference = [delta_R,delta_G,delta_B]

        is_shadow = False

        if lower[0] < nonshadow_shadow_difference[0] < upper[0]:
            if lower[1] < nonshadow_shadow_difference[1] < upper[1]:
                if lower[2] < nonshadow_shadow_difference[2] < upper[2]:
                    is_shadow = True

        if is_shadow == True:
            rectified_shadow[coordinates_region] = 0
        else :
            rectified_shadow[coordinates_region] = 255

    return None

def find_elements_in_tuples_of_two_lists(sub_tuple:tuple,origin_tuple:tuple)->tuple:
    res = np.array(sub_tuple)
    for i in range(len(origin_tuple[0])):
        label = False
        for j in range(len(origin_tuple[0])):
            if sub_tuple[0][i] == origin_tuple[0][j]:
                if sub_tuple[1][i] == origin_tuple[1][j]:
                    label = True
                    break

        if label == False:
            np.delete(res, (0, i))
            np.delete(res, (1, i))

    return tuple(res)


def detect_shadow(img:np.ndarray)->np.ndarray:

    img_gray = TAM.RGB2GRAY(img)
    markers,mark = watershed.watershed(img_gray)
    TAM_region = get_region_TAM(img,markers,mark)
    binarized_TAM_region = binarize_TAM_region(TAM_region,markers,mark)



    return binarized_TAM_region


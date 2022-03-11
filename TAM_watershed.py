import TAM
import watershed
import numpy as np
import cv2
import os

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

        mean_nonshadow_B = np.mean(B[coordinates])
        mean_nonshadow_G = np.mean(G[coordinates])
        mean_nonshadow_R = np.mean(R[coordinates])
        coordinates_nonshadow = []

        for i in range(len(coordinates[0])):
            if B[coordinates[0][i]][coordinates[1][i]] >= mean_nonshadow_B and G[coordinates[0][i]][coordinates[1][i]] >= mean_nonshadow_G and R[coordinates[0][i]][coordinates[1][i]] >= mean_nonshadow_R:
                coordinates_nonshadow.append((coordinates[0][i],coordinates[1][i]))

        total_num = len(coordinates_nonshadow)

        if total_num == 0:
            continue

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

    # From 1 to mark-1. Value equal to zero means that point is not marked.
    for i in range(1,mark):
        # 获得已分割的区域的坐标
        coordinates_region = np.where(markers == i)

        # 查找已分割区域的，且是非阴影区域的坐标点
        coordinates_shadow_region, coordinates_nonshadow_region = find_shadow_nonshadow__pixels_in_tuples_of_two_lists(coordinates_region,shadow_img)
        # 下面一行代码可以优化

        is_shadow = False

        F_R_NS = np.mean(R[coordinates_nonshadow_region])
        if np.isnan(F_R_NS):
            F_R_NS = 0

        F_G_NS = np.mean(G[coordinates_nonshadow_region])
        if np.isnan(F_G_NS):
            F_G_NS = 0

        F_B_NS = np.mean(B[coordinates_nonshadow_region])
        if np.isnan(F_B_NS):
            F_B_NS = 0

        F_R_S = np.mean(R[coordinates_shadow_region])
        if np.isnan(F_R_S):
            F_R_S = 0

        F_G_S = np.mean(G[coordinates_shadow_region])
        if np.isnan(F_G_S):
            F_G_S = 0

        F_B_S = np.mean(B[coordinates_shadow_region])
        if np.isnan(F_B_S):
            F_B_S = 0

        delta_B = F_B_NS-F_B_S
        delta_G = F_G_NS-F_G_S
        delta_R = F_R_NS-F_R_S

        if F_B_NS == 0:
            delta_B = 0

        if delta_B == 0:
            if delta_B == 0 and delta_G == 0 and delta_R == 0:
                is_shadow = True

        else:
            L = np.array([m*F_R_NS/F_B_NS,n*F_G_NS/F_B_NS,1])
            L = L*delta_B

            lower = k1*L
            upper = k2*L

            nonshadow_shadow_difference = np.array([delta_R,delta_G,delta_B])

            if lower[0] < nonshadow_shadow_difference[0] < upper[0]:
                if lower[1] < nonshadow_shadow_difference[1] < upper[1]:
                    if lower[2] < nonshadow_shadow_difference[2] < upper[2]:
                        is_shadow = True

        if is_shadow == True:
            rectified_shadow[coordinates_region] = 0
        elif is_shadow == False :
            rectified_shadow[coordinates_region] = 255

    return rectified_shadow

def rectify_shadow_with_origin_img(origin_img:np.ndarray,shadow_img:np.ndarray,markers:np.ndarray,mark:int)->np.ndarray:

    res = np.empty(shape=shadow_img.shape)
    res.fill(255)

    B, G, R = cv2.split(origin_img)
    coordinates_shadow_all = np.where(shadow_img == 0)

    # From 1 to mark-1. Value equal to zero means that point is not marked.
    for i in range(1, mark):
        # 获得已分割的区域的坐标
        coordinates_region = np.where(markers == i)
        # 查找区域中阴影区域的坐标点
        coordinates_shadow_region = find_shadow_pixels_in_tuples_of_two_lists(coordinates_region,shadow_img)
        is_shadow = False

        F_R = np.mean(R[coordinates_region])
        if(np.isnan(F_R)):
            F_R = 0

        F_G = np.mean(G[coordinates_region])
        if (np.isnan(F_G)):
            F_G = 0

        F_B = np.mean(B[coordinates_region])
        if (np.isnan(F_B)):
            F_B = 0

        for j in range(len(coordinates_shadow_region[0])):

            F_R_ = R[coordinates_shadow_region[0][j]][coordinates_shadow_region[1][j]]
            F_G_ = G[coordinates_shadow_region[0][j]][coordinates_shadow_region[1][j]]
            F_B_ = B[coordinates_shadow_region[0][j]][coordinates_shadow_region[1][j]]

            if F_R_ < F_R:
                if F_G_ < F_G:
                    if F_B_ < F_B:
                        is_shadow = True
                        x = coordinates_shadow_region[0][j]
                        y = coordinates_shadow_region[1][j]

            if is_shadow:
                res[x][y] = 0


    return res


def find_shadow_nonshadow__pixels_in_tuples_of_two_lists(origin_tuple:tuple,shadow_img:np.ndarray)->(list,list):
    res_shadow = []
    res_nonshadow = []

    length = len(origin_tuple[0])

    for i in range(length):
        is_shadow = False

        x = origin_tuple[0][i]
        y = origin_tuple[1][i]

        if shadow_img[x][y] == 0:
            is_shadow = True

        if is_shadow:
            res_shadow.append((x,y))
        else:
            res_nonshadow.append((x, y))
    return (res_shadow, res_nonshadow)

def find_shadow_pixels_in_tuples_of_two_lists(origin_tuple:tuple,shadow_img:np.ndarray)->tuple:
    res = np.array(origin_tuple)
    colum_origin = len(origin_tuple[0])
    colum = colum_origin
    res = res.reshape((-1))
    delete_num = 0

    for i in range(colum_origin):
        is_shadow = False

        x = origin_tuple[0][i]
        y = origin_tuple[1][i]

        if shadow_img[x][y] == 0:
            is_shadow = True

        if is_shadow == False:
            res = np.delete(res, (i - delete_num))
            colum = colum - 1
            res = np.delete(res, (i + colum - delete_num))
            delete_num += 1

    res = res.reshape(2, -1)
    return tuple(res)


def find_nonshadow_pixels_in_tuples_of_two_lists(origin_tuple:tuple,shadow_img:np.ndarray)->tuple:

    res = np.array(origin_tuple)
    colum_origin = len(origin_tuple[0])
    colum = colum_origin
    res = res.reshape((-1))
    delete_num = 0

    for i in range(colum_origin):
        is_shadow = True

        x = origin_tuple[0][i]
        y = origin_tuple[1][i]

        if shadow_img[x][y] == 255:
            is_shadow = False

        if is_shadow == True:
            res = np.delete(res, (i - delete_num))
            colum = colum - 1
            res = np.delete(res, (i + colum - delete_num))
            delete_num += 1

    res = res.reshape(2, -1)
    return tuple(res)

def find_elements_in_tuples_of_two_lists(sub_tuple:tuple,origin_tuple:tuple)->tuple:
    res = np.array(sub_tuple)
    colum_origin = len(sub_tuple[0])
    colum = colum_origin
    res = res.reshape((-1))
    delete_num = 0

    for i in range(colum_origin):
        label = False
        for j in range(len(origin_tuple[0])):
            if sub_tuple[0][i] == origin_tuple[0][j]:
                if sub_tuple[1][i] == origin_tuple[1][j]:
                    label = True
                    break

        if label == False:
            res = np.delete(res, (i-delete_num))
            colum = colum-1
            res = np.delete(res, (i+colum-delete_num))
            delete_num += 1

    res = res.reshape(2,-1)
    return tuple(res)


def detect_shadow(img:np.ndarray)->np.ndarray:

    img_gray = TAM.RGB2GRAY(img)
    markers,mark = watershed.watershed(img_gray)
    TAM_region = get_region_TAM(img,markers,mark)
    shadow_img = binarize_TAM_region(TAM_region,markers,mark)
    rectifeid_shadow = rectify_shadow(img=img,shadow_img=shadow_img,markers=markers,mark=mark)
    final_res = rectify_shadow_with_origin_img(origin_img=img,shadow_img=rectifeid_shadow,markers=markers,mark=mark)

    return final_res


def get_all_pics(img:np.ndarray,file:str)->np.ndarray:


    img_gray = cv2.imread("./Grayscale/" + file, flags = 0)
    if type(img_gray) != np.ndarray:
        img_gray = TAM.RGB2GRAY(img)
        cv2.imwrite(filename="./Grayscale/" + file, img=img_gray)

    print(file)
    img_segemented, markers,mark = watershed.use_opencv_watershed(img, img_gray)
    cv2.imwrite(filename="./watershed/" + file, img=img_segemented)

    TAM_region = cv2.imread("./TAM_region_img/" + file, flags = 0)
    if type(TAM_region) != np.ndarray:
        TAM_region = get_region_TAM(img,markers,mark)
        cv2.imwrite(filename="./TAM_region_img/" + file, img=TAM_region)

    combined_img = cv2.imread("./combinedimages/" + file, flags=0)
    if type(combined_img) != np.ndarray:
        combined_img = TAM.get_CombinedImage(img, TAM_region)
        cv2.imwrite(filename="./combinedimages/" + file, img=combined_img)

    shadow_img = binarize_TAM_region(combined_img,markers,mark)
    cv2.imwrite(filename="./shadow_img/" + file, img=shadow_img)

    rectifeid_shadow = rectify_shadow(img=img,shadow_img=shadow_img,markers=markers,mark=mark)
    cv2.imwrite(filename="./rectified_shadow/" + file, img=rectifeid_shadow)

    final_res = rectify_shadow_with_origin_img(origin_img=img,shadow_img=rectifeid_shadow,markers=markers,mark=mark)
    cv2.imwrite(filename="./result/" + file, img=final_res)


    return None


def test_TAM_RGB2GRAY():

    for root, dirs, files in os.walk("./images"):
        for file in files:
            filepath = "./images/" + file
            img = cv2.imread(filepath)
            img_gray = TAM.RGB2GRAY(img)
            cv2.imwrite(filename="./Grayscale/" + file, img=img_gray)

    return None

def test_TAM_watershed():

    for root, dirs, files in os.walk("./Grayscale"):
        for file in files:
            filepath = "./Grayscale/" + file
            img_gray = cv2.imread(filepath)
            img_segemented, markers,mark = watershed.use_opencv_watershed(img_gray)
            cv2.imwrite(filename="./watershed/" + file, img=img_segemented)

    return None

def test_get_region_TAM():

    for root, dirs, files in os.walk("./Grayscale"):
        for file in files:
            filepath = "./Grayscale/" + file
            img = cv2.imread('./images/'+file)
            img_gray = cv2.imread(filepath)
            _, markers,mark = watershed.use_opencv_watershed(img_gray)
            TAM_region = get_region_TAM(img, markers, mark)
            cv2.imwrite(filename="./TAM_region_img/" + file, img=TAM_region)

    return None
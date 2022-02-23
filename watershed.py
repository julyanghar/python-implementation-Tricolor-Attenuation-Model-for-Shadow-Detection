import collections
import numpy as np
import cv2
import skimage.color as color
global SHED
SHED = -1

def findLocalMinimum(img: np.ndarray)->(np.ndarray,int) :

    img = img.squeeze()
    num_row = img.shape[0]
    num_colum = img.shape[1]

    # 初始化markers
    markers = np.zeros((num_row,num_colum),dtype=int)
    # 需要把最外围的一圈像素点设置marker为-1，否则可能会越界访问
    for i in range(num_row):
        markers[i][num_colum-1] = SHED
        markers[i][0] = SHED

    for i in range(num_colum):
        markers[0][i] = SHED
        markers[num_row-1][i] = SHED

    # mark代表颜色，不同的值代表不同的颜色(区域)，mark为0代表暂未上色
    mark = 0
    mark_deque = collections.deque()
    for i in range(1,num_row-1):
        for j in range(1,num_colum-1):
            val = img[i][j]
            # 找极小值点
            if(markers[i][j] == 0 and val <= img[i-1][j] and val <= img[i+1][j] and val <= img[i][j-1] and val <= img[i][j+1]):
                mark += 1
                markers[i][j] = mark
                mark_deque.append((i,j))

            while (len(mark_deque)!=0):
                x,y = mark_deque.pop()
                # If values are equal and the neighborhood is not marked yet, put it in the deque
                if(val == img[x-1][y] and markers[x-1][y] == 0):
                    markers[x-1][y] = mark
                    mark_deque.append((x-1,y))

                if (val == img[x + 1][y] and markers[x + 1][y] == 0):
                    markers[x + 1][y] = mark
                    mark_deque.append((x + 1, y))

                if (val == img[x][y-1] and markers[x][y-1] == 0):
                    markers[x][y-1] = mark
                    mark_deque.append((x, y-1))

                if (val == img[x][y + 1] and markers[x][y + 1] == 0):
                    markers[x][y + 1] = mark
                    mark_deque.append((x, y + 1))

    return markers,mark

def watershed(img: np.ndarray) -> (np.ndarray,int):
    # convert to gray scale
    if(len(img.shape)==3):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    num_row = img_gray.shape[0]
    num_colum = img_gray.shape[1]
    markers,mark = findLocalMinimum(img_gray)

    # 获取梯度图像：
    img_grad = cv2.GaussianBlur(img_gray,ksize=(3,3),sigmaX=0)
    grad_x = cv2.Sobel(img_gray,  -1, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, -1, 0, 1, ksize=3)
    img_grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)

    img_gray = img_grad

    # 初始化，首先找到极小值区域附近的点，这些点具有如下特征：
    # 1：本身未被标记
    # 2：四邻域内存在被标记的像素点
    unmarked_pixels_deques = collections.deque()

    for i in range(256):
        unmarked_pixels_deques.append(collections.deque())
    for i in range(1,num_row-1):
        for j in range(1,num_colum-1):
            if markers[i][j] == 0 and (markers[i-1][j] > 0 or markers[i+1][j] > 0 or markers[i][j-1] > 0 or markers[i][j+1] > 0):
                # 若本身未被标记，且四邻域内存在被标记的像素点，记下该像素点的位置，压入队列中
                unmarked_pixels_deques[img_gray[i][j]].append((i,j))

    # 先淹没地势最低的点，也就是像素值最小的点

    # 目前正在淹没的像素点的像素值
    pixel_level = 0
    # 当前像素点的坐标
    while True:
        for i in range(257):
            if i == 256:
                pixel_level = 256
                break
            if len(unmarked_pixels_deques[i]) > 0 :
                pixel_level = i
                break
        if pixel_level == 256:
            break

        while len(unmarked_pixels_deques[pixel_level]) > 0:
            x,y = unmarked_pixels_deques[pixel_level].pop()
    #         首先查看该像素点附近有几种颜色
            check_list = []
            if markers[x-1][y] > 0:
                check_list.append(markers[x-1][y])

            if markers[x+1][y] > 0:
                check_list.append(markers[x+1][y])

            if markers[x][y-1] > 0:
                check_list.append(markers[x][y-1])

            if markers[x][y+1] > 0:
                check_list.append(markers[x][y+1])

            check_set = set(check_list)
            if len(check_set) > 1:
                markers[x][y] = SHED
            elif len(check_set) == 1:
                markers[x][y] = check_list[0]

#     检查每一个邻域像素点，若该邻域像素点未被标记，其四邻域内有被标记的像素点,并且其不存在队列中，则将其压入队列中

            if markers[x][y] > 0:
                x_,y_ = x - 1,y
                value_ = img_gray[x_][y_]
                if markers[x_][y_] == 0 and unmarked_pixels_deques[value_].count((x_,y_)) == 0 :
                    unmarked_pixels_deques[value_].append((x_,y_))
                    # 若队列中有更暗的像素值，转而遍历那些像素值
                    if value_ < pixel_level:
                        pixel_level = value_

                x_, y_ = x + 1, y
                value_ = img_gray[x_][y_]
                if markers[x_][y_] == 0 and unmarked_pixels_deques[value_].count((x_, y_)) == 0 :
                    unmarked_pixels_deques[value_].append((x_, y_))
                    if value_ < pixel_level:
                        pixel_level = value_

                x_, y_ = x, y - 1
                value_ = img_gray[x_][y_]
                if markers[x_][y_] == 0 and unmarked_pixels_deques[value_].count((x_, y_)) == 0 :
                    unmarked_pixels_deques[value_].append((x_, y_))
                    if value_ < pixel_level:
                        pixel_level = value_

                x_, y_ = x, y + 1
                value_ = img_gray[x_][y_]
                if markers[x_][y_] == 0 and unmarked_pixels_deques[value_].count((x_, y_)) == 0 :
                    unmarked_pixels_deques[value_].append((x_, y_))
                    if value_ < pixel_level:
                        pixel_level = value_

    return markers,mark

def check_markers(img_gray:np.ndarray,markers:np.ndarray):
    row = img_gray.shape[0]
    colum = img_gray.shape[1]

    for i in range(1,row-1):
        for j in range(1,colum-1):
            if markers[i][j] > 0 :
                if markers[i - 1][j] > 0:
                    if img_gray[i][j] == img_gray[i-1][j]:
                        if markers[i][j] != markers[i-1][j]:
                            print("error")
                    elif img_gray[i][j] != img_gray[i-1][j]:
                        if markers[i][j] == markers[i - 1][j]:
                            print("error")

                if markers[i + 1][j] > 0:

                    if img_gray[i][j] == img_gray[i+1][j]:
                        if markers[i][j] != markers[i+1][j]:
                            print("error")
                    elif img_gray[i][j] != img_gray[i+1][j]:
                        if markers[i][j] == markers[i+1][j]:
                            print("error")

                if markers[i][j-1] > 0:

                    if img_gray[i][j] == img_gray[i][j-1]:
                        if markers[i][j] != markers[i][j-1]:
                            print("error")
                    elif img_gray[i][j] != img_gray[i][j-1]:
                        if markers[i][j] == markers[i][j-1]:
                            print("error")

                if markers[i][j+1] > 0:

                    if img_gray[i][j] == img_gray[i][j+1]:
                        if markers[i][j] != markers[i][j+1]:
                            print("error")
                    elif img_gray[i][j] != img_gray[i][j+1]:
                        if markers[i][j] == markers[i][j+1]:
                            print("error")

def colorMask(img:np.ndarray,markers:np.ndarray)->np.ndarray:

    return None


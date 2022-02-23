import os
import TAM
import cv2
import numpy as np

def test_RGB2GRAY():
    for root, dirs, files in os.walk("./images"):
        for file in files:
            filepath = "./images/" + file
            img = cv2.imread(filepath)
            res = TAM.RGB2GRAY(img)
            cv2.imwrite(filename="./Grayscale/Grayscale_" + file, img=res)
    return None

def test_get_pixels_with_tuple():
    img = np.array([
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
        [13,14,15,16]
    ])


    # coordinates = np.where(img > 10)
    coordinates = ([0,1,1],[0,0,1])

    list = img[coordinates]
    print(list)

def test_array_to_tuple():
    img = np.array([1,2,3,4,5])
    res = tuple(img)
    print(res)


test_array_to_tuple()
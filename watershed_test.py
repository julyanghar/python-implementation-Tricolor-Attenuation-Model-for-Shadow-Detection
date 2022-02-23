# test tuple
import watershed
import TAM_watershed
import numpy as np
import cv2
from skimage import data, util, filters, color
from skimage.segmentation import watershed as wts

def test_tuple():
    tup = (1,2)
    x,y = tup
    print(x)
    print(y)

# test deque
def test_deque():
    import collections
    de = collections.deque()
    de.append((0,1))
    de.append((0,2))
    element = de.pop()
    length = len(de)
    print(length)

def test_modify_img(img:np.ndarray)->np.ndarray:
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray[129][144] = 0
    return img_gray

def test_set():
    list = [255,255,256]
    myset = set(list)
    print(len(myset))






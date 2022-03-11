import TAM
import TAM_watershed
import watershed
import cv2
import os


# main函数
for root,dirs,files in os.walk("./images"):
    for file in files:
        filepath = "./images/" + file
        img = cv2.imread(filepath, flags = 1)
        TAM_watershed.get_all_pics(img,file)

# TAM_watershed.test_get_region_TAM()
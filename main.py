import TAM
import TAM_watershed
import watershed
import cv2
import os

# file = '0a.jpg'
# filepath = "./images/" + file
# img = cv2.imread(filepath)
# res = TAM_watershed.detect_shadow(img)
# cv2.imwrite(filename="./result/" + file, img=res)

for root,dirs,files in os.walk("./images"):
    for file in files:
        filepath = "./images/" + file
        img = cv2.imread(filepath)
        res = TAM_watershed.detect_shadow(img)
        cv2.imwrite(filename="./result/" + file, img=res)
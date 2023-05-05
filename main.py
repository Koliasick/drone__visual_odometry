from datasets import ImagesDataset
import cv2
import os
import pathes


dataset = ImagesDataset("C:\\Users\\Admin\\Desktop\\drone__visual_odometry\\Dataset")

for data in dataset:
    img = cv2.imread(data[0])
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(data)
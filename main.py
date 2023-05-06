from datasets import ImagesDataset
import cv2
import os
import paths


dataset = ImagesDataset("C:\\Users\\Admin\\Desktop\\drone__visual_odometry\\Dataset")

print(dataset[0]["zoomed_in"]["non_intersecting_images"])
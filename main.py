from datasets import ImagesDataset
import cv2
from dataloaders import get_simple_data_loader


dataset = ImagesDataset("C:\\Users\\Admin\\Desktop\\drone__visual_odometry\\Dataset")

data_loader = get_simple_data_loader(dataset)

for i_batch, sample_batched in enumerate(data_loader):
    print(sample_batched["drone_image"][0].numpy())
    cv2.namedWindow('Image')
    cv2.namedWindow('Image2')
    cv2.imshow('Image', sample_batched["drone_image"][0].numpy())
    cv2.imshow('Image2', sample_batched["satellite_image"][0].numpy())
    cv2.waitKey(0)
    cv2.destroyAllWindows()



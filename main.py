from datasets import ImagesDataset
import cv2
from dataloaders import get_simple_data_loader
from augmentations import ZoomAndShiftTransform, ReplaceSatelliteImageTransform, MirrorTransform, CustomPILToTensorTransform


dataset = ImagesDataset("C:\\Users\\Admin\\Desktop\\drone__visual_odometry\\Dataset",
                        transforms=[
                            ReplaceSatelliteImageTransform(0.33),
                            MirrorTransform(0.25, 0.25),
                            ZoomAndShiftTransform((1.0, 2.0)),
                            CustomPILToTensorTransform()
                        ])

data_loader = get_simple_data_loader(dataset)

for i_batch, sample_batched in enumerate(data_loader):
    print(sample_batched)


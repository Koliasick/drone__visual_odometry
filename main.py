from datasets import ImagesDataset
from dataloaders import get_simple_data_loader
from augmentations import ZoomAndShiftTransform, ReplaceSatelliteImageTransform, MirrorTransform, CustomPILToTensorTransform
from torchvision import transforms


dataset = ImagesDataset("C:\\Users\\Admin\\Desktop\\drone__visual_odometry\\Dataset",
                        transforms=[
                            ReplaceSatelliteImageTransform(0.33),
                            MirrorTransform(0.25, 0.25),
                            ZoomAndShiftTransform((1.0, 2.0)),
                            CustomPILToTensorTransform()
                        ])

data_loader = get_simple_data_loader(dataset)

for i_batch, sample_batched in enumerate(data_loader):

    if i_batch == 0:

        to_image_transform = transforms.ToPILImage()
        drone_image = to_image_transform(sample_batched["drone_image"][0])
        satellite_image = to_image_transform(sample_batched["satellite_image"][0])

        print(sample_batched["satellite_image_contains_drone_image"][0])
        drone_image.show()
        satellite_image.show()


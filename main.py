from PIL import ImageDraw
from datasets import ImagesDataset
from dataloaders import get_simple_data_loader
from augmentations import ZoomAndShiftTransform, ReplaceSatelliteImageTransform, MirrorTransform, CustomPILToTensorTransform, ResizeImages
import torch
from networks import ResNetModified, DroneSatelliteModelAttempt2
from losses import inside_image_loss
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt


dataset = ImagesDataset("C:\\Users\\Admin\\Desktop\\drone__visual_odometry\\Dataset",
                        transforms=[
                            ReplaceSatelliteImageTransform(0.33),
                            MirrorTransform(0.25, 0.25),
                            ZoomAndShiftTransform((1.0, 1.9)),
                            ResizeImages((600, 600)),
                            CustomPILToTensorTransform()
                        ])

data_loader = get_simple_data_loader(dataset)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DroneSatelliteModelAttempt2().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()

    cumulative_loss = 0

    # Iterate over the training data in batches
    for i_batch, sample_batched in enumerate(data_loader):

        ##########################

        for i in range(32):
            if(sample_batched["satellite_image_contains_drone_image"][i] == 1):
                drone_img = to_pil_image(sample_batched["drone_image"][i])
                satellite_image = to_pil_image(sample_batched["satellite_image"][i])

                point = sample_batched["drone_on_satellite_coordinates"]["x"][i], sample_batched["drone_on_satellite_coordinates"]["y"][i]

                draw = ImageDraw.Draw(satellite_image)
                draw.ellipse((point[0]-2, point[1]-2, point[0]+2, point[1]+2), fill='red')

                fig, (ax1, ax2) = plt.subplots(1, 2)

                ax1.imshow(satellite_image)
                ax2.imshow(drone_img)

                plt.show()

        #########################

        input_data = torch.cat([sample_batched["drone_image"], sample_batched["satellite_image"]], dim=1)
        input_data = input_data.type(torch.float)

        target_data = torch.stack([
            sample_batched["satellite_image_contains_drone_image"],
            sample_batched["drone_on_satellite_coordinates"]["x"],
            sample_batched["drone_on_satellite_coordinates"]["y"]], dim=1)
        target_data = target_data.type(torch.float)

        input_data, target_data = input_data.to(device), target_data.to(device)

        optimizer.zero_grad()

        output_data = model(input_data)

        loss = inside_image_loss(output_data, target_data)

        loss.backward()

        optimizer.step()

        cumulative_loss += loss

        print(f"Processed batch: {i_batch}, Loss: {loss}")

    # Print the loss for this epoch
    print('Epoch [{}/{}], loss: {}'
          .format(epoch+1, num_epochs, cumulative_loss/i_batch))

torch.save(model.state_dict(), "Initial save")

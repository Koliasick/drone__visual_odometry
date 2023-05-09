from datasets import ImagesDataset
from dataloaders import get_simple_data_loader
from augmentations import ZoomAndShiftTransform, ReplaceSatelliteImageTransform, MirrorTransform, CustomPILToTensorTransform, ResizeImages
from torchvision import transforms
import torch
from networks import ResNetModified
from losses import inside_image_loss


dataset = ImagesDataset("C:\\Users\\Admin\\Desktop\\drone__visual_odometry\\Dataset",
                        transforms=[
                            ReplaceSatelliteImageTransform(0.33),
                            MirrorTransform(0.25, 0.25),
                            ZoomAndShiftTransform((1.0, 2.0)),
                            ResizeImages((600, 600)),
                            CustomPILToTensorTransform()
                        ])

data_loader = get_simple_data_loader(dataset)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNetModified().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.007)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()

    cumulative_loss = 0

    # Iterate over the training data in batches
    for i_batch, sample_batched in enumerate(data_loader):

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

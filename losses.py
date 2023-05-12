import torch.nn.functional as F
import torch


def inside_image_loss(pred, target):
    # pred: a tensor of shape (batch_size, 3) containing the predicted values
    # target: a tensor of shape (batch_size, 3) containing the target values

    # Extract the predicted and target values for whether image1 is inside image2
    pred_inside = pred[:, 0]
    target_inside = target[:, 0]

    # Extract the predicted and target values for the x and y coordinates of image1 inside image2
    pred_x = pred[:, 1]
    pred_y = pred[:, 2]
    target_x = target[:, 1]
    target_y = target[:, 2]

    # Calculate the RMSE loss for the "inside" prediction
    loss_inside = torch.sqrt(F.mse_loss(pred_inside, target_inside))

    # Calculate the RMSE loss for the x and y coordinates, and make it so when target_inside is 0 loss_x and loss_y is 0
    loss_x = torch.sqrt(F.mse_loss(pred_x / 600 * target_inside, target_x / 600 * target_inside))
    loss_y = torch.sqrt(F.mse_loss(pred_y / 600 * target_inside, target_y / 600 * target_inside))

    # Calculate the total loss as the sum of the individual losses
    loss = loss_inside + loss_x + loss_y

    return loss
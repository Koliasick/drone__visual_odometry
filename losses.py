import torch.nn.functional as F


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

    # Calculate the binary cross-entropy loss for the "inside" prediction
    loss_inside = F.mse_loss(pred_inside, target_inside) * 1000

    # Calculate the mean squared error loss for the x and y coordinates
    loss_x = F.l1_loss(pred_x, target_x * pred_inside)
    loss_y = F.l1_loss(pred_y, target_y * pred_inside)

    # Calculate the total loss as the sum of the individual losses
    loss = loss_inside + loss_x + loss_y

    return loss
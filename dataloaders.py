import torch

def get_simple_data_loader(dataset):
    return torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)


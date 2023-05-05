import torch
import pandas as pd
import pathes
import os


class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, transforms=None):
        self.transforms = transforms
        self.items = []

        part_one = self._get_part_of_dataset(base_path, pathes.part_one_path)
        part_two = self._get_part_of_dataset(base_path, pathes.part_two_path)
        part_three = self._get_part_of_dataset(base_path, pathes.part_three_path)
        part_four = self._get_part_of_dataset(base_path, pathes.part_four_path)

        self.items.extend(part_one)
        self.items.extend(part_two)
        self.items.extend(part_three)
        self.items.extend(part_four)

    def _get_part_of_dataset(self, base_path, dataset_part):

        zoomed_in_path = os.path.join(base_path, dataset_part, pathes.satellite_zoomed_in_path)
        zoomed_out_path = os.path.join(base_path, dataset_part, pathes.satellite_zoomed_out_path)

        zoomed_in_csv_path = os.path.join(zoomed_in_path, pathes.csv_file_name)
        # Filenames are the same for zoomed and non-zoomed images so we dont need to read both files
        #zoomed_out_csv_path = os.path.join(zoomed_out_path, pathes.csv_file_name)

        result = []

        # Zoomed in images
        df = pd.read_csv(zoomed_in_csv_path)
        for index, row in df.iterrows():
            drone_image_path = os.path.join(base_path, dataset_part, pathes.drone_images_path, row["name"])
            zoomed_in_satellite_image_path = os.path.join(zoomed_in_path, pathes.satellite_images_path, row["name"])
            zoomed_out_satellite_image_path = os.path.join(zoomed_out_path, pathes.satellite_images_path, row["name"])
            result.append(
                (drone_image_path,
                 zoomed_in_satellite_image_path,
                 zoomed_out_satellite_image_path,
                 pathes.intersection_images_part_1_2_path
                 if dataset_part == pathes.part_one_path or dataset_part == pathes.part_two_path
                 else pathes.intersection_images_part_3_4_path))

        return result

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

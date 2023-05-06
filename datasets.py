import torch
import pandas as pd
import paths
import os


class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, transforms=None):
        self.transforms = transforms
        self.items = []

        part_one = self._get_part_of_dataset(base_path, paths.part_one_path)
        part_two = self._get_part_of_dataset(base_path, paths.part_two_path)
        part_three = self._get_part_of_dataset(base_path, paths.part_three_path)
        part_four = self._get_part_of_dataset(base_path, paths.part_four_path)

        self.items.extend(part_one)
        self.items.extend(part_two)
        self.items.extend(part_three)
        self.items.extend(part_four)

    def _get_non_intersecting_images_path(self, base_non_intersecting_path, satellite_zoom_level_path, part_path):

        result = os.listdir(
            os.path.join(
                base_non_intersecting_path,
                satellite_zoom_level_path,
                part_path
            ))

        result = [
            os.path.join(
                base_non_intersecting_path,
                satellite_zoom_level_path,
                part_path,
                name
            ) for name in result]

        return result

    def _get_non_intersecting_images_paths(self, base_path):
        non_intersecting_path = os.path.join(base_path, paths.intersection_images_path)

        non_intersecting_images_zoomed_in_part_1_2_file_names = self._get_non_intersecting_images_path(
            non_intersecting_path,
            paths.satellite_zoomed_in_path,
            paths.intersection_images_part_1_2_path
        )

        non_intersecting_images_zoomed_in_part_3_4_file_names = self._get_non_intersecting_images_path(
            non_intersecting_path,
            paths.satellite_zoomed_in_path,
            paths.intersection_images_part_3_4_path
        )

        non_intersecting_images_zoomed_out_part_1_2_file_names = self._get_non_intersecting_images_path(
            non_intersecting_path,
            paths.satellite_zoomed_out_path,
            paths.intersection_images_part_1_2_path
        )

        non_intersecting_images_zoomed_out_part_3_4_file_names = self._get_non_intersecting_images_path(
            non_intersecting_path,
            paths.satellite_zoomed_out_path,
            paths.intersection_images_part_3_4_path
        )

        return non_intersecting_images_zoomed_in_part_1_2_file_names, \
            non_intersecting_images_zoomed_in_part_3_4_file_names, \
            non_intersecting_images_zoomed_out_part_1_2_file_names, \
            non_intersecting_images_zoomed_out_part_3_4_file_names


    def _get_part_of_dataset(self, base_path, dataset_part):
        zoomed_in_path = os.path.join(base_path, dataset_part, paths.satellite_zoomed_in_path)
        zoomed_out_path = os.path.join(base_path, dataset_part, paths.satellite_zoomed_out_path)

        zoomed_in_csv_path = os.path.join(zoomed_in_path, paths.csv_file_name)
        # Filenames are the same for zoomed and non-zoomed images, so we don't need to read both files
        # zoomed_out_csv_path = os.path.join(zoomed_out_path, pathes.csv_file_name)

        non_intersecting_images_zoomed_in_part_1_2_file_names, \
            non_intersecting_images_zoomed_in_part_3_4_file_names, \
            non_intersecting_images_zoomed_out_part_1_2_file_names, \
            non_intersecting_images_zoomed_out_part_3_4_file_names = self._get_non_intersecting_images_paths(base_path)

        result = []

        df = pd.read_csv(zoomed_in_csv_path)
        for index, row in df.iterrows():
            drone_image_path = os.path.join(base_path, dataset_part, paths.drone_images_path, row["name"])
            zoomed_in_satellite_image_path = os.path.join(zoomed_in_path, paths.satellite_images_path, row["name"])
            zoomed_out_satellite_image_path = os.path.join(zoomed_out_path, paths.satellite_images_path, row["name"])
            result.append(
                {
                    "drone_image": drone_image_path,
                    "zoomed_in": {
                        "image": zoomed_in_satellite_image_path,
                        "non_intersecting_images": non_intersecting_images_zoomed_in_part_1_2_file_names
                            if dataset_part == paths.part_one_path or dataset_part == paths.part_two_path
                            else non_intersecting_images_zoomed_in_part_3_4_file_names
                    },
                    "zoomed_out": {
                        "image": zoomed_out_satellite_image_path,
                        "non_intersecting_images": non_intersecting_images_zoomed_out_part_1_2_file_names
                            if dataset_part == paths.part_one_path or dataset_part == paths.part_two_path
                            else non_intersecting_images_zoomed_out_part_3_4_file_names
                    }
                })

        return result

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # Apply transforms and other augmentation here

        # Should return
        #   1. Drone image
        #   2. Satellite image
        #   3. Coordinates of center of drone image on satellite image
        return self.items[idx]

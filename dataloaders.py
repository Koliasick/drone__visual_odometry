import pathes
import random


dataset_part = random.choices([pathes.part_one_path, pathes.part_two_path, pathes.part_three_path, pathes.part_four_path])
zoom_level = random.choices([pathes.satellite_zoomed_in_path, pathes.satellite_zoomed_out_path])
contains_drone_image = random.choices([True, True, False])


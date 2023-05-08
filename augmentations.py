import cv2
import numpy as np
import random
from PIL import Image


def zoomAndOffset(image, point, zoom_factor):
    point_on_original_image_x = point[0]
    point_on_original_image_y = point[1]

    width, height = image.size
    center = (width // 2, height // 2)

    point_on_original_image_to_center_offset_x = center[0] - point_on_original_image_x
    point_on_original_image_to_center_offset_y = center[1] - point_on_original_image_y

    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)

    width_delta = width - new_width
    height_delta = height - new_height

    x_shift_offset = random.randrange(int(-width_delta / 2), int(width_delta / 2))
    y_shift_offset = random.randrange(int(-height_delta / 2), int(height_delta / 2))

    new_center_x = center[0] + x_shift_offset
    new_center_y = center[1] + y_shift_offset

    zoomed = image.crop((
        new_center_x - new_width // 2,
        new_center_y - new_height // 2,
        new_center_x + new_width // 2,
        new_center_y + new_height // 2
    ))

    zoomed_image_center = (zoomed.size[0] // 2, zoomed.size[1] // 2)

    # Go to the point that was center on original image and then go to the point that we need to find
    point_on_zoomed_image_x = zoomed_image_center[0] - x_shift_offset - point_on_original_image_to_center_offset_x
    point_on_zoomed_image_y = zoomed_image_center[1] - y_shift_offset - point_on_original_image_to_center_offset_y

    return zoomed, (point_on_zoomed_image_x, point_on_zoomed_image_y)


def resize(image, point, new_size):
    width, height = image.size

    scale_x = width / new_size[0]
    scale_y = height / new_size[1]

    new_point = (point[0] / scale_x, point[1] / scale_y)

    resized = image.resize(new_size)

    return resized, new_point


class ZoomAndShiftTransform:
    def __init__(self, zoom_range=(1.0, 2.0)):
        self.zoom_range = zoom_range

    def __call__(self, satellite_img, point, drone_img, non_intersecting_satellite_images, satellite_img_contains_drone_img):

        zoomed_image, point_on_zoomed_image = zoomAndOffset(satellite_img, point, random.randrange(self.zoom_range))

        resized_image, point_on_resized_image = resize(zoomed_image, point_on_zoomed_image, (1280, 1180))

        return resized_image, point_on_resized_image, drone_img, satellite_img_contains_drone_img


class ReplaceSatelliteImageTransform:
    def __init__(self, replacement_chance):
        self.replacement_chance = replacement_chance

    def __call__(self, satellite_img, point, drone_img, non_intersecting_satellite_images, satellite_img_contains_drone_img):

        if random.random() < self.replacement_chance:
            replaced_image = Image.open(random.sample(non_intersecting_satellite_images, 1)[0])
            return replaced_image, point, drone_img, False
        else:
            return satellite_img, point, drone_img, satellite_img_contains_drone_img

class MirrorTransform:
    def __init__(self, vertical_flip_chance, horizontal_flip_chance):
        self.vertical_flip_chance = vertical_flip_chance
        self.horizontal_flip_chance = horizontal_flip_chance

    def __call__(self, satellite_img, point, drone_img, non_intersecting_satellite_images, satellite_img_contains_drone_img):

        resulting_satellite_img = satellite_img
        resulting_drone_img = drone_img
        resulting_point = point

        if random.random() < self.vertical_flip_chance:
            

        if random.random() < self.horizontal_flip_chance:


        return resulting_satellite_img, resulting_point, resulting_drone_img, satellite_img_contains_drone_img
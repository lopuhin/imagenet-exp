import random
import math

from albumentations import ImageOnlyTransform
from albumentations.augmentations import functional as F
import cv2
import numpy as np


def wrap_image_transform(transform):
    return lambda x: transform(image=x)['image']


class RandomResizedCrop(ImageOnlyTransform):
    """
    Port of RandomResizedCrop from torchvision to albumentations interface.
    Crop the given image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size: int,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 interpolation=cv2.INTER_LINEAR,
                 always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.size = size
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    def get_params_dependent_on_targets(self, params):
        return {}

    @staticmethod
    def _get_params(img: np.ndarray, scale, ratio):
        """Get parameters for a random sized crop.

        Args:
            img: Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w):
            i: Upper pixel coordinate.
            j: Left pixel coordinate.
            h: Height of the cropped image.
            w: Width of the cropped image.
        """
        img_h, img_w, _ = img.shape
        area = img_w * img_h
        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_w and h <= img_h:
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)
                return i, j, h, w

        # Fallback
        w = min(img_w, img_h)
        i = (img_h - w) // 2
        j = (img_w - w) // 2
        return i, j, w, w

    def apply(self, img: np.ndarray, **params):
        h_start, w_start, crop_height, crop_width = self._get_params(
            img, self.scale, self.ratio)
        crop = img[h_start: h_start + crop_height,
                   w_start: w_start + crop_width]
        return F.resize(crop, self.size, self.size, self.interpolation)

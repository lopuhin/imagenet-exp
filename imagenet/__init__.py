import re
import warnings

from PIL import Image


if not re.match(r'^\d+\.\d+\.\d+\.post', Image.PILLOW_VERSION):
    warnings.warn(
        'Pillow-SIMD is not installed, data loading will be much slower.')

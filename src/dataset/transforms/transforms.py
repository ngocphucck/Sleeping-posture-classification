import numpy as np
import cv2


class HistogramEqualization(object):
    def __call__(self, image):
        return cv2.equalizeHist(image)


class ColorConvert(object):
    def __init__(self, color_space='RGB'):
        self.color_space = color_space

    def __call__(self, image):
        image = image.convert(self.color_space)

        return image


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = image.resize(self.size)

        return image


class Cutout(object):
    def __init__(self, mask_size, prob, cutout_inside, mask_color=(0, 0, 0)):
        self.mask_size = mask_size
        self.prob = prob
        self.cutout_inside = cutout_inside
        self.mask_color = mask_color

    def __call__(self, image):
        image = np.asarray(image).copy()
        mask_size_half = self.mask_size // 2
        offset = 1 if mask_size_half % 2 == 0 else 0

        if np.random.random() > self.prob:
            return image

        height, width = image.shape[:2]

        if self.cutout_inside:
            cxmin, cxmax = mask_size_half, width + offset - mask_size_half
            cymin, cymax = mask_size_half, height + offset - mask_size_half
        else:
            cxmin, cxmax = 0, width + offset
            cymin, cymax = 0, height + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)

        xmin = max(0, cx - mask_size_half)
        xmax = min(width, cx + mask_size_half)
        ymin = max(0, cy - mask_size_half)
        ymax = min(height, cy + mask_size_half)

        image[xmin:xmax, ymin:ymax] = self.mask_color

        return image

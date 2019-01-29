# Adapted from: https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/augmentations/augmentations.py

import math
import numbers
import random
import numpy as np

from PIL import Image, ImageOps

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask, ins, depth):
        img, mask, ins, depth = Image.fromarray(img, mode='RGB'), Image.fromarray(mask, mode='L'), Image.fromarray(ins, mode='I'), Image.fromarray(depth, mode='F')
        assert img.size == mask.size
        assert img.size == ins.size
        assert img.size == depth.size

        for a in self.augmentations:
            img, mask, ins, depth = a(img, mask, ins, depth)

        return np.array(img), np.array(mask, dtype=np.uint8), np.array(ins, dtype=np.uint64), np.array(depth, dtype=np.float32)


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask, ins, depth):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)
            ins = ImageOps.expand(ins, border=self.padding, fill=0)
            depth = ImageOps.expand(depth, border=self.padding, fill=0)

        assert img.size == mask.size
        assert img.size == ins.size
        assert img.size == depth.size

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask, ins, depth
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST), ins.resize((tw,th), Image.NEAREST), depth.resize((tw, th), Image.NEAREST)

        _sysrand = random.SystemRandom()
        x1 = _sysrand.randint(0, w - tw)
        y1 = _sysrand.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)), ins.crop((x1, y1, x1 + tw, y1 + th)),  depth.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, ins, depth):
        _sysrand = random.SystemRandom()
        if _sysrand.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), ins.transpose(Image.FLIP_LEFT_RIGHT), depth.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask, ins, depth


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask, ins=None):
        assert img.size == mask.size
        assert img.size == ins.size
        assert img.size == depth.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST), ins.resize(self.size, Image.NEAREST), depth.resize(self.size, Image.NEAREST)


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask, ins, depth):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask, ins, depth
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST), ins.resize((ow, oh), Image.NEAREST), depth.resuze((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST), ins.resize((ow,oh), Image.NEAREST), depth.reszie((ow, oh), Image.NEAREST)


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        _sysrand = random.SystemRandom()
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            
            target_area = _sysrand.uniform(0.45, 1.0) * area
            aspect_ratio = _sysrand.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if _sysrand.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = _sysrand.randint(0, img.size[0] - w)
                y1 = _sysrand.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), Image.BILINEAR), mask.resize((self.size, self.size),
                                                                                       Image.NEAREST)

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask, ins, depth):
        _sysrand = random.SystemRandom()
        rotate_degree = _sysrand.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST), ins.rotate(rotate_degree, Image.NEAREST), depth.rotate(rotate_degree, Image.NEAREST)


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size
        _sysrand = random.SystemRandom()
        
        w = int(_sysrand.uniform(0.5, 2) * img.size[0])
        h = int(_sysrand.uniform(0.5, 2) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return self.crop(*self.scale(img, mask))

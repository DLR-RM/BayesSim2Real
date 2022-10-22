"""
Coior Augmentation: Autocontrast, Brightness, Color, Contrast, Equalize, Posterize, Solarize, SolarizeAdd

Geometric Augmentation: Rotate_BBox, ShearX_BBox, ShearY_BBox, TranslateX_BBox, TranslateY_BBox

Mask Augmentation: Cutout

Color Augmentation based on BBoxes: Equalize_Only_BBoxes, Solarize_Only_BBoxes

Geometric Augmentation based on BBoxes: Rotate_Only_BBoxes, ShearX_Only_BBoxes, ShearY_Only_BBoxes,
                                        TranslateX_Only_BBoxes, TranslateY_Only_BBoxes, Flip_Only_BBoxes

Mask Augmentation based on BBoxes: BBox_Cutout, Cutout_Only_BBoxes

"""
import torch, random
from src.core.datasets import functional
import torchvision.transforms.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from skimage.util import random_noise

from PIL import Image, ImageOps

import numpy as np


### Basic Augmentation
class Compose:
    """
    Composes several transforms together.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bboxs):
        for t in self.transforms:
            image, bboxs = t(image, bboxs)
        return image, bboxs


class ToTensor:
    """
    Converts a PIL Image or numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    Only applied to image, not bboxes.
    """
    def __call__(self, image, bboxs):
        return F.to_tensor(image), bboxs


class Normalize(torch.nn.Module):
    """
    Normalize a tensor image with mean and standard deviation.
    Only applied to image, not bboxes.
    """
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, image, bboxs):
        return F.normalize(image, self.mean, self.std, self.inplace), bboxs



class Brightness(torch.nn.Module):
    """
    Adjust image brightness using magnitude.
    Only applied to image, not bboxes.
    """
    def __init__(self, p, magnitude, minus=True):
        super().__init__()
        self.p = p
        self.magnitude = magnitude
        self.minus = minus

    def forward(self, image, bboxs):
        if self.minus and (torch.rand(1) < 0.5): self.magnitude *= -1
        if torch.rand(1) < self.p:
            brightness_image = functional.brightness(image, 1+self.magnitude)
            return brightness_image, bboxs
        else:
            return image, bboxs


class Color(torch.nn.Module):
    """
    Adjust image color balance using magnitude.
    Only applied to image, not bboxes.
    """
    def __init__(self, p, magnitude, minus=True):
        super().__init__()
        self.p = p
        self.magnitude = magnitude
        self.minus = minus

    def forward(self, image, bboxs):
        if self.minus and (torch.rand(1) < 0.5): self.magnitude *= -1
        if torch.rand(1) < self.p:
            color_image = functional.color(image, 1+self.magnitude)
            return color_image, bboxs
        else:
            return image, bboxs


class Contrast(torch.nn.Module):
    """
    Adjust image contrast using magnitude.
    Only applied to image, not bboxes.
    """
    def __init__(self, p, magnitude, minus=True):
        super().__init__()
        self.p = p
        self.magnitude = magnitude
        self.minus = minus

    def forward(self, image, bboxs):
        if self.minus and (torch.rand(1) < 0.5): self.magnitude *= -1
        if torch.rand(1) < self.p:
            contrast_image = functional.contrast(image, 1+self.magnitude)
            return contrast_image, bboxs
        else:
            return image, bboxs


class Equalize(torch.nn.Module):
    """
    Equalize the histogram of the given image.
    Only applied to image, not bboxes.
    """
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, image, bboxs):
        if torch.rand(1) < self.p:
            equalize_image = ImageOps.equalize(image)
            return equalize_image, bboxs
        else:
            return image, bboxs


class Posterize(torch.nn.Module):
    """
    Posterize the image by reducing the number of bits for each color channel.
    Only applied to image, not bboxes.
    """
    def __init__(self, p, bits):
        super().__init__()
        self.p = p
        self.bits = int(bits)

    def forward(self, image, bboxs):
        if torch.rand(1) < self.p:
            posterize_image = ImageOps.posterize(image, self.bits)
            return posterize_image, bboxs
        else:
            return image, bboxs


class Solarize(torch.nn.Module):
    """
    Solarize the image by inverting all pixel values above a threshold.
    Only applied to image, not bboxes.
    """
    def __init__(self, p, threshold):
        super().__init__()
        self.p = p
        self.threshold = int(threshold)

    def forward(self, image, bboxs):
        if torch.rand(1) < self.p:
            solarize_image = ImageOps.solarize(image, self.threshold)
            return solarize_image, bboxs
        else:
            return image, bboxs


class SolarizeAdd(torch.nn.Module):
    """
    Solarize the image by added image below a threshold.
    Add addition amount to image and then clip the pixel value to 0~255 or 0~1.
    Parameter addition must be integer.
    Only applied to image, not bboxes.
    """
    def __init__(self, p, addition, threshold=128, minus=True):
        super().__init__()
        self.p = p
        self.addition = int(addition)
        self.threshold = int(threshold)
        self.minus = minus

    def forward(self, image, bboxs):
        if self.minus and (torch.rand(1) < 0.5): self.addition *= -1
        if torch.rand(1) < self.p:
            solarize_add_image = functional.solarize_add(image, self.addition, self.threshold)
            return solarize_add_image, bboxs
        else:
            return image, bboxs


class GaussianNoise(torch.nn.Module):
    """
    Blurr the image by added image below a threshold.
    Add addition amount to image and then clip the pixel value to 0~255 or 0~1.
    Parameter addition must be integer.
    Only applied to image, not bboxes.
    """
    def __init__(self, p, std, mean=0.):
        super().__init__()
        self.p = p
        self.std = std
        self.mean = mean

    def forward(self, image, bboxs):
        if torch.rand(1) < self.p:
            trans = T.ToTensor()

            image = trans(image)[0]

            gauss_image = torch.tensor(random_noise(image, mode='gaussian', mean=self.mean, var=self.std, clip=True))

            trans = T.ToPILImage()
            gauss_image = trans(gauss_image)

            return gauss_image, bboxs
        else:
            return image, bboxs


class Horizontal_Flip(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, image, bboxs):
        if torch.rand(1) < self.p:
            # Flip image
            flipped_image = F.hflip(image)

            # Flip boxes
            flipped_bboxs = bboxs
            flipped_bboxs[:, 0] = image.width - bboxs[:, 0] - 1
            flipped_bboxs[:, 2] = image.width - bboxs[:, 2] - 1
            flipped_bboxs = flipped_bboxs[:, [2, 1, 0, 3]]

            return flipped_image, flipped_bboxs
        else:
            return image, bboxs

### Geometric Augmentation
class Crop_And_Resize_X(torch.nn.Module):
    """
    Crop the image, resize it and change
    the boxes accordingly on the X-axis.
    Level is the number of pixel that will be cropped.
    Both applied to image and bboxes
    """
    def __init__(self, p, pixels, replace=128):
        super().__init__()
        self.p = p
        self.pixels = pixels
        self.replace = replace

    def forward(self, image, bboxs):
        if torch.rand(1) < self.p:

            trans = T.Resize([image.size[1], image.size[0]], interpolation=Image.BILINEAR)

            cropped_image = F.crop(image, 0, self.pixels, image.size[1], image.size[0] - self.pixels)
            cropped_image = trans(cropped_image)
            if bboxs is None:
                return cropped_image, bboxs
            else:
                cropped_bbox = functional.crop_and_resize_x(image, bboxs, self.pixels)
                return cropped_image, cropped_bbox
        else:
            return image, bboxs


class Crop_And_Resize_Y(torch.nn.Module):
    """
    Crop the image, resize it and change
    the boxes accordingly on the X-axis.
    Level is the number of pixel that will be cropped.
    Both applied to image and bboxes
    """
    def __init__(self, p, pixels, replace=128):
        super().__init__()
        self.p = p
        self.pixels = pixels
        self.replace = replace

    def forward(self, image, bboxs):
        if torch.rand(1) < self.p:
            trans = T.Resize([image.size[1], image.size[0]], interpolation=Image.BILINEAR)

            cropped_image = F.crop(image, self.pixels, 0, image.size[1] - self.pixels, image.size[0])
            cropped_image = trans(cropped_image)
            if bboxs is None:
                return cropped_image, bboxs
            else:
                cropped_bbox = functional.crop_and_resize_y(image, bboxs, self.pixels)
                return cropped_image, cropped_bbox
        else:
            return image, bboxs


class ShearX_BBox(torch.nn.Module):
    """
    Shear image and change bboxes on X-axis.
    The pixel values filled in will be of the value replace.
    Level is usually between -0.3~0.3.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Both applied to image and bboxes.
    """
    def __init__(self, p, level, replace=128, minus=True):
        super().__init__()
        self.p = p
        self.level = level
        self.replace = replace
        self.minus = minus

    def forward(self, image, bboxs):
        if self.minus and (torch.rand(1) < 0.5): self.level *= -1
        if torch.rand(1) < self.p:
            shear_image = image.transform(image.size, Image.AFFINE, (1, self.level, 0, 0, 1, 0), fillcolor=(self.replace, self.replace, self.replace))
            if bboxs is None:
                return shear_image, bboxs
            else:
                shear_bbox = functional.shear_with_bboxes(image, bboxs, self.level, self.replace, shift_horizontal=True)
                return shear_image, shear_bbox
        else:
            return image, bboxs


class ShearY_BBox(torch.nn.Module):
    """
    Shear image and change bboxes on Y-axis.
    The pixel values filled in will be of the value replace.
    Level is usually between -0.3~0.3.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Both applied to image and bboxes.
    """
    def __init__(self, p, level, replace=128, minus=True):
        super().__init__()
        self.p = p
        self.level = level
        self.replace = replace
        self.minus = minus

    def forward(self, image, bboxs):
        if self.minus and (torch.rand(1) < 0.5): self.level *= -1
        if torch.rand(1) < self.p:
            shear_image = image.transform(image.size, Image.AFFINE, (1, 0, 0, self.level, 1, 0), fillcolor=(self.replace, self.replace, self.replace))
            if bboxs is None:
                return shear_image, bboxs
            else:
                shear_bbox = functional.shear_with_bboxes(image, bboxs, self.level, self.replace, shift_horizontal=False)
                return shear_image, shear_bbox
        else:
            return image, bboxs


### Color Augmentation based on BBoxes
class Equalize_Only_BBoxes(torch.nn.Module):
    """
    Apply equalize to each bboxes in the image with probability.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Only applied to image not bboxes.
    """
    def __init__(self, p):
        super().__init__()
        self.p = p/3

    def forward(self, image, bboxs):
        if bboxs is None:
            return image, bboxs
        else:
            equalize_image = functional.equalize_only_bboxes(image, bboxs, self.p)
            return equalize_image, bboxs


class Solarize_Only_BBoxes(torch.nn.Module):
    """
    Apply solarize to each bboxes in the image with probability.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Only applied to image not bboxes.
    """
    def __init__(self, p, threshold):
        super().__init__()
        self.p = p/3
        self.threshold = int(threshold)

    def forward(self, image, bboxs):
        if bboxs is None:
            return image, bboxs
        else:
            solarize_image = functional.solarize_only_bboxes(image, bboxs, self.p, self.threshold)
            return solarize_image, bboxs


### Geometric Augmentation based on BBoxes
class Rotate_Only_BBoxes(torch.nn.Module):
    """
    Apply rotation to each bboxes in the image with probability.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Only applied to image not bboxes.
    """
    def __init__(self, p, degrees, replace=128, minus=True):
        super().__init__()
        self.p = p/3
        self.degrees = degrees
        self.replace = replace
        self.minus = minus

    def forward(self, image, bboxs):
        if self.minus and (torch.rand(1) < 0.5): self.degrees *= -1
        if bboxs is None:
            return image, bboxs
        else:
            rotate_image = functional.rotate_only_bboxes(image, bboxs, self.p, self.degrees, self.replace)
            return rotate_image, bboxs


class ShearX_Only_BBoxes(torch.nn.Module):
    """
    Apply shear to each bboxes in the image with probability only on X-axis.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Only applied to image not bboxes.
    """
    def __init__(self, p, level, replace=128, minus=True):
        super().__init__()
        self.p = p/3
        self.level = level
        self.replace = replace
        self.minus = minus

    def forward(self, image, bboxs):
        if self.minus and (torch.rand(1) < 0.5): self.level *= -1
        if bboxs is None:
            return image, bboxs
        else:
            shear_image = functional.shear_only_bboxes(image, bboxs, self.p, self.level, self.replace, shift_horizontal=True)
            return shear_image, bboxs


class ShearY_Only_BBoxes(torch.nn.Module):
    """
    Apply shear to each bboxes in the image with probability only on Y-axis.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Only applied to image not bboxes.
    """
    def __init__(self, p, level, replace=128, minus=True):
        super().__init__()
        self.p = p/3
        self.level = level
        self.replace = replace
        self.minus = minus

    def forward(self, image, bboxs):
        if self.minus and (torch.rand(1) < 0.5): self.level *= -1
        if bboxs is None:
            return image, bboxs
        else:
            shear_image = functional.shear_only_bboxes(image, bboxs, self.p, self.level, self.replace, shift_horizontal=False)
            return shear_image, bboxs


class Flip_Only_BBoxes(torch.nn.Module):
    """
    Apply horizontal flip to each bboxes in the image with probability.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Only applied to image not bboxes.
    """
    def __init__(self, p):
        super().__init__()
        self.p = p/3

    def forward(self, image, bboxs):
        if bboxs is None:
            return image, bboxs
        else:
            flip_image = functional.flip_only_bboxes(image, bboxs, self.p)
            return flip_image, bboxs

M = 10

color_range = torch.arange(2., 9., (8-2) / M).tolist()
shear_range = torch.arange(0, 0.6+1e-8, (0.5-0)/M).tolist()
translate_range = torch.arange(0, 250+1e-8, (250-0)/M).tolist()
gauss_range = torch.arange(0, .7, .7/M).tolist()


Mag = {'Brightness' : color_range, 'Color' : color_range, 'Contrast' : color_range, 
       'Posterize' : torch.arange(1, 4+1e-8, (4-1)/M).tolist()[::-1], 'Sharpness' : color_range,
       'Solarize' : torch.arange(120, 256+1e-8, (256-120)/M).tolist()[::-1], 'SolarizeAdd' : torch.arange(0, 200+1e-8, (200-0)/M).tolist(),

       'GaussianNoise' : gauss_range,
       
       'ShearX_BBox' : shear_range, 'ShearY_BBox' : shear_range,

       'ShearX_Only_BBoxes' : shear_range, 'ShearY_Only_BBoxes' : shear_range,

       'Crop_And_Resize_X' : translate_range, 'Crop_And_Resize_Y' : translate_range,
       
       'Solarize_Only_BBoxes' : torch.arange(120, 256+1e-8, (256-120)/M).tolist()[::-1]
      }


Fun = {'Brightness' : Brightness, 'Color' : Color, 'Contrast' : Contrast, 'Equalize' : Equalize,
       'Posterize' : Posterize, 'Solarize' : Solarize, 'SolarizeAdd' : SolarizeAdd,

       'GaussianNoise' : GaussianNoise, 'Horizontal_Flip' : Horizontal_Flip,

       'ShearX_BBox' : ShearX_BBox, 'ShearY_BBox' : ShearY_BBox,

       'ShearX_Only_BBoxes' : ShearX_Only_BBoxes, 'ShearY_Only_BBoxes' : ShearY_Only_BBoxes,
           
       'Flip_Only_BBoxes' : Flip_Only_BBoxes, 'Crop_And_Resize_X' : Crop_And_Resize_X, 'Crop_And_Resize_Y': Crop_And_Resize_Y,
       
       'Equalize_Only_BBoxes' : Equalize_Only_BBoxes, 'Solarize_Only_BBoxes' : Solarize_Only_BBoxes,
      }


class Policy(torch.nn.Module):
    def __init__(self, policy, pre_transform, post_transform, mag):
        super().__init__()
        self.pre_transform = pre_transform
        self.post_transform = post_transform

        if policy == 'policy_v3':
            self.policy = policy_v3(mag)
        elif policy == 'policy_v0':
            self.policy = policy_v0(mag)
        elif policy == 'policy_v001':
            self.policy = policy_v001(mag)
        else:
            raise AssertionError

    def forward(self, image, bboxs):
        policy_idx = random.randint(0, len(self.policy)-1)
        policy_transform = self.pre_transform + self.policy[policy_idx] + self.post_transform
        policy_transform = Compose(policy_transform)
        image, bboxs = policy_transform(image, bboxs)
        return image, bboxs
    
    
def SubPolicy(f1, p1, m1, f2, p2, m2):
    subpolicy = []
    if f1 in ['Equalize', 'Equalize_Only_BBoxes', 'Flip_Only_BBoxes', 'Horizontal_Flip']: subpolicy.append(Fun[f1](p1))
    else: subpolicy.append(Fun[f1](p1, Mag[f1][m1]))
    
    if f2 in ['Equalize', 'Equalize_Only_BBoxes', 'Flip_Only_BBoxes', 'Horizontal_Flip']: subpolicy.append(Fun[f2](p2))
    else: subpolicy.append(Fun[f2](p2, Mag[f2][m2]))
        
    return subpolicy


def SubPolicy3(f1, p1, m1, f2, p2, m2, f3, p3, m3):
    subpolicy = []
    if f1 in ['AutoContrast', 'Equalize', 'Equalize_Only_BBoxes', 'Flip_Only_BBoxes', 'Horizontal_Flip']: subpolicy.append(Fun[f1](p1))
    else: subpolicy.append(Fun[f1](p1, Mag[f1][m1]))
    
    if f2 in ['AutoContrast', 'Equalize', 'Equalize_Only_BBoxes', 'Flip_Only_BBoxes', 'Horizontal_Flip']: subpolicy.append(Fun[f2](p2))
    else: subpolicy.append(Fun[f2](p2, Mag[f2][m2]))
        
    if f3 in ['AutoContrast', 'Equalize', 'Equalize_Only_BBoxes', 'Flip_Only_BBoxes', 'Horizontal_Flip']: subpolicy.append(Fun[f3](p3))
    else: subpolicy.append(Fun[f3](p3, Mag[f3][m3]))
        
    return subpolicy  



def policy_v3(mag):
    policy = [SubPolicy('Crop_And_Resize_X', 1, mag,           'GaussianNoise', 1, mag),
              SubPolicy('Posterize', 0.8, mag,                 'Contrast', 1.0, mag),
              SubPolicy('Equalize_Only_BBoxes', 0.6, None,     'Color', 1.0, mag),
              SubPolicy('Contrast', 0.5, mag,                  'ShearY_BBox', 0.8, mag),
              SubPolicy('Equalize', 0.8, None,                 'GaussianNoise', 0.7, mag),
              SubPolicy('Horizontal_Flip', 1., None,           'Crop_And_Resize_Y', 0.2, mag),
              SubPolicy('Equalize', 1.0, None,                 'Crop_And_Resize_X', 1.0, mag),
              SubPolicy('Posterize', 0.6, mag,                 'Color', 0.7, mag),
              SubPolicy('Color', 0.6, mag,                     'Solarize_Only_BBoxes', 1.0, mag),
              SubPolicy('Equalize', 0.4, None,                 'Solarize', 0.8, mag),
              SubPolicy('Brightness', 1.0, mag,                'Solarize', 0.8, mag),
              SubPolicy('Contrast', 0.3, mag,                  'ShearY_BBox', 0.8, mag),
              SubPolicy('Color', 0.6, mag,                     'Contrast', 0.2, mag),
              SubPolicy('Solarize_Only_BBoxes', 1.0, mag,      'Contrast', 1.0, mag),
              SubPolicy('SolarizeAdd', 0.8, mag,               'Equalize', 0.8, None)]
    return policy

def policy_v0(mag):
    policy = [
        SubPolicy3('Crop_And_Resize_X', 0.4, mag,       'GaussianNoise', 0.6, mag,          'Color', 0.8, mag),
        SubPolicy3('Crop_And_Resize_Y', 0.7, mag,       'SolarizeAdd', 0.3, mag,            'Contrast', 0.4, mag),
        SubPolicy3('Horizontal_Flip', 0.5, None,        'Equalize', 0.8, None,              'Solarize', 1.0, mag),
        SubPolicy3('Horizontal_Flip', 0.5, None,        'Posterize', 0.7, mag,              'Color', 0.6, mag),
        SubPolicy3('ShearY_BBox', 0.8, mag,             'Contrast', 0.2, mag,               'Equalize_Only_BBoxes', 0.8, None),
        SubPolicy3('ShearX_Only_BBoxes', 0.3, mag,      'Color', 0.4, mag,                  'Solarize', 0.8, mag),
        SubPolicy3('ShearY_Only_BBoxes', 0.2, mag,      'Contrast', 0.5, mag,               'Brightness', 0.9, mag),
        SubPolicy3('Horizontal_Flip', 0.5, None,        'Solarize', 1.0, mag,               'GaussianNoise', 1, mag),
        SubPolicy3('Crop_And_Resize_Y', 0.2, mag,       'Brightness', 1.0, mag,             'Color', 0.7, mag),
        SubPolicy3('Flip_Only_BBoxes', 0.4, mag,        'SolarizeAdd', 0.8, mag,            'GaussianNoise', 0.1, mag),
        SubPolicy3('ShearX_BBox', 0.2, mag,             'Color', 0.7, mag,                  'Solarize', 0.4, mag),
        SubPolicy3('Crop_And_Resize_X', 0.4, mag,       'Solarize', 0.4, mag,               'Color', 0.8, mag),
        SubPolicy3('ShearY_BBox', 0.8, mag,             'Posterize', 0.7, mag,              'Contrast', 1.0, mag)
    ]
    return policy

def policy_v001(mag):
    policy = [
              SubPolicy('Horizontal_Flip', 1., None,           'Crop_And_Resize_Y', 0.2, mag),
    ]
    return policy
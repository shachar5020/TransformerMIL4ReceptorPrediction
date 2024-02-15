import random

import numpy as np
import torch
from PIL import Image
from skimage.util import random_noise
from torchvision import transforms


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = torch.randint(low=0, high=h, size=(1,)).numpy()[0]
            x = torch.randint(low=0, high=w, size=(1,)).numpy()[0]
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img


class MyRotation:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)


class MyCropTransform:
    """crop the image at upper left."""

    def __init__(self, tile_size):
        self.tile_size = tile_size

    def __call__(self, x):
        x = transforms.functional.crop(img=x, top=x.size[0] - self.tile_size, left=x.size[1] - self.tile_size,
                                       height=self.tile_size, width=self.tile_size)
        return x


class MyGaussianNoiseTransform:
    """add gaussian noise."""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        stdev = self.sigma[0] + (self.sigma[1] - self.sigma[0]) * np.random.rand()
        x_arr = np.asarray(x)

        # random_noise() method will convert image in [0, 255] to [0, 1.0],
        # inherently it use np.random.normal() to create normal distribution
        # and adds the generated noised back to image
        noise_img = random_noise(x_arr, mode='gaussian', var=stdev ** 2)
        noise_img = (255 * noise_img).astype(np.uint8)

        x = Image.fromarray(noise_img)
        return x


class MyMeanPixelRegularization:
    """replace patch with single pixel value"""

    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        if np.random.rand() < self.p:
            x = torch.zeros_like(x) + torch.tensor([[[0.87316266]], [[0.79902739]], [[0.84941472]]])
        return x


def define_transformations(transform_type, train, tile_size, color_param=0.1, norm_type='standard'):
    MEAN = {'TCGA': [58.2069073 / 255, 96.22645279 / 255, 70.26442606 / 255],
            'HEROHE': [224.46091564 / 255, 190.67338568 / 255, 218.47883547 / 255],
            'standard': [0.8998, 0.8253, 0.9357],
            'Imagenet': [0.485, 0.456, 0.406],
            'Amir': [0.9357, 0.8253, 0.8998]
            }

    STD = {'TCGA': [40.40400300279664 / 255, 58.90625962739444 / 255, 45.09334057330417 / 255],
           'HEROHE': [np.sqrt(1110.25292532) / 255, np.sqrt(2950.9804851) / 255, np.sqrt(1027.10911208) / 255],
           'standard': [0.1125, 0.1751, 0.0787],
           'Imagenet': [0.229, 0.224, 0.225],
           'Amir': [0.0787, 0.1751, 0.1125]
           }

    # Setting the transformation:
    if transform_type == 'aug_receptornet':
        final_transform = transforms.Compose([transforms.Normalize(
            mean=(MEAN[norm_type][0], MEAN[norm_type][1], MEAN[norm_type][2]),
            std=(STD[norm_type][0], STD[norm_type][1], STD[norm_type][2]))])
    else:
        final_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(
                                                  mean=(MEAN[norm_type][0], MEAN[norm_type][1], MEAN[norm_type][2]),
                                                  std=(STD[norm_type][0], STD[norm_type][1], STD[norm_type][2]))
                                              ])
    scale_factor = 0.2
    if transform_type != 'none' and train:
        if transform_type == 'flip':
            transform1 = \
                transforms.Compose([transforms.RandomVerticalFlip(),
                                    transforms.RandomHorizontalFlip()])
        elif transform_type == 'rvf':  # rotate, vertical flip
            transform1 = \
                transforms.Compose([MyRotation(angles=[0, 90, 180, 270]),
                                    transforms.RandomVerticalFlip()])
        elif transform_type in ['cbnfrsc', 'cbnfrs']:  # color, blur, noise, flip, rotate, scale, +-cutout
            transform1 = \
                transforms.Compose([
                    transforms.ColorJitter(brightness=(0.85, 1.15), contrast=(0.75, 1.25),
                                           saturation=0.1, hue=(-0.1, 0.1)),
                    transforms.GaussianBlur(3, sigma=(1e-7, 1e-1)),
                    MyGaussianNoiseTransform(sigma=(0, 0.05)),
                    transforms.RandomVerticalFlip(),
                    MyRotation(angles=[0, 90, 180, 270]),
                    transforms.RandomAffine(degrees=0, scale=(1, 1 + scale_factor)),
                ])
        elif transform_type in ['pcbnfrsc', 'pcbnfrs']:  # param color, blur, noise, flip, rotate, scale, +-cutout
            transform1 = \
                transforms.Compose([
                    transforms.ColorJitter(brightness=color_param, contrast=color_param * 2,
                                           saturation=color_param, hue=color_param),
                    transforms.GaussianBlur(3, sigma=(1e-7, 1e-1)),
                    MyGaussianNoiseTransform(sigma=(0, 0.05)),
                    transforms.RandomVerticalFlip(),
                    MyRotation(angles=[0, 90, 180, 270]),
                    transforms.RandomAffine(degrees=0, scale=(1, 1 + scale_factor)),
                ])

        elif transform_type == 'aug_receptornet':
            transform1 = \
                transforms.Compose([
                    transforms.ColorJitter(brightness=64.0 / 255, contrast=0.75, saturation=0.25, hue=0.04),
                    transforms.RandomHorizontalFlip(),
                    MyRotation(angles=[0, 90, 180, 270]),
                    transforms.ToTensor(),
                    Cutout(n_holes=1, length=100),
                    MyMeanPixelRegularization(p=0.75)
                ])

        elif transform_type == 'cbnfr':  # color, blur, noise, flip, rotate
            transform1 = \
                transforms.Compose([
                    transforms.ColorJitter(brightness=(0.85, 1.15), contrast=(0.75, 1.25),
                                           saturation=0.1, hue=(-0.1, 0.1)),
                    transforms.GaussianBlur(3, sigma=(1e-7, 1e-1)),
                    MyGaussianNoiseTransform(sigma=(0, 0.05)),
                    transforms.RandomVerticalFlip(),
                    MyRotation(angles=[0, 90, 180, 270]),
                ])
        elif transform_type in ['bnfrsc', 'bnfrs']:  # blur, noise, flip, rotate, scale, +-cutout
            transform1 = \
                transforms.Compose([
                    transforms.GaussianBlur(3, sigma=(1e-7, 1e-1)),
                    MyGaussianNoiseTransform(sigma=(0, 0.05)),
                    transforms.RandomVerticalFlip(),
                    MyRotation(angles=[0, 90, 180, 270]),
                    transforms.RandomAffine(degrees=0, scale=(1, 1 + scale_factor)),
                ])
        elif transform_type == 'frs':  # flip, rotate, scale
            transform1 = \
                transforms.Compose([
                    transforms.RandomVerticalFlip(),
                    MyRotation(angles=[0, 90, 180, 270]),
                    transforms.RandomAffine(degrees=0, scale=(1, 1 + scale_factor)),
                ])
        else:  # custom transform
            return transform_type

        transform = transforms.Compose([transform1, final_transform])
    else:
        transform = final_transform

    if transform_type in ['cbnfrsc', 'bnfrsc', 'c_0_05_bnfrsc', 'pcbnfrsc']:
        transform.transforms.append(Cutout(n_holes=1, length=100))

    return transform

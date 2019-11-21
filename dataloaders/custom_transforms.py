import torch
import random
import numpy as np
from torchvision import transforms

from PIL import Image, ImageOps, ImageFilter

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        region_prop = sample['rp']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask,
                'rp': region_prop}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        region_prop = sample['rp']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)
        region_prop = np.array(region_prop).astype(np.float32)
        region_prop = region_prop[np.newaxis,:,:]

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        region_prop = torch.from_numpy(region_prop).float()

        return {'image': img,
                'label': mask,
                'rp': region_prop}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rp = sample['rp']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            rp = rp.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask,
                'rp': rp}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

class FixedCrop(object):

    # Crop according to given dimensions where,
    # x1,x2 width crop
    # y1,y2 height crop

    def __init__(self,x1,x2,y1,y2):
        self.x1,self.x2,self.y1,self.y2 = x1,x2,y1,y2

    def __call__(self, sample):
        img = sample['image']
        img = img[self.y1:self.y2,self.x1:self.x2,:3]
        label = sample['label']
        label = label[self.y1:self.y2, self.x1:self.x2]
        region_prop = sample['rp']
        region_prop = region_prop[self.y1:self.y2, self.x1:self.x2]

        return {'image': Image.fromarray(img),
                'label': Image.fromarray(label),
                'rp': Image.fromarray(region_prop)}

class ColorJitter(object):

    """Add color jitter to input image"""

    def __init__(self,jitter=0.1):
        self.jitter = jitter

    def __call__(self,sample):
        img = sample['image']
        label = sample['label']
        transform_rgb = transforms.ColorJitter(self.jitter,self.jitter,self.jitter, 0)

        return {'image': transform_rgb(img),
                'label': label}

class RandomCrop(object):
    ## Makes sliding windows style crops: Aasheesh
    def __init__(self,crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img=np.asarray(sample['image'])
        mask=np.asarray(sample['label'])
        h,w,_=img.shape
        assert h == self.crop_size[0], "Input image height incorrect"
        crop_w=np.random.randint(0,w-self.crop_size[1])
        return {'image': Image.fromarray(img[0:self.crop_size[0],crop_w:crop_w+self.crop_size[1],:]),
                'label': Image.fromarray(mask[0:self.crop_size[0],crop_w:crop_w+self.crop_size[1]])}

class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}
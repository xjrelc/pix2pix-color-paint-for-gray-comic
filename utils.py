# -*- coding: utf-8 -*-

import skimage
import imageio
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
from PIL import ImageFilter

class DataLoader(object):
    
    def __init__(self, dataset_path=r'.\datasets\CombinedImages',scale=4, crop_size=96, name=None):
        self.image_height = 256
        self.image_width = 256
        self.dataset_path = dataset_path
        self.__scale = 4
        self.__crop_size = 96
        self.scale = scale
        self.crop_size = crop_size
        self.name = name
        pass

    def imread(self, path):
        return imageio.imread(path, pilmode='RGB').astype(np.float)
    
    def find_images(self, path):
        result = []
        for filename in os.listdir(path):
            _, ext = os.path.splitext(filename.lower())
            if ext == ".jpg" or ext == ".png":
                result.append(os.path.join(path, filename))
                pass
            pass
        result.sort()
        return result
    
    def load_data(self, batch_size=1, for_testing=False):
        search_result = self.find_images(self.dataset_path)
        batch_images = np.random.choice(search_result, size=batch_size)
        images_A = []
        images_B = []
        for image_path in batch_images:
            combined_image = self.imread(image_path)
            h, w, c = combined_image.shape
            nW = int(w/2)
            image_A, image_B = combined_image[:, :nW, :], combined_image[:, nW:, :]
            image_A = skimage.transform.resize(image_A, (self.image_height, self.image_width))
            image_B = skimage.transform.resize(image_B, (self.image_height, self.image_width))
            if not for_testing and np.random.random() < 0.5:
                # 数据增强，左右翻转
                image_A = np.fliplr(image_A)
                image_B = np.fliplr(image_B)
                pass
            images_A.append(image_A)
            images_B.append(image_B)
            pass
        images_A = np.array(images_A)/127.5 - 1.
        images_B = np.array(images_B)/127.5 - 1.
        return images_A, images_B

    def load_batch(self, batch_size=1, for_testing=False):
        search_result = self.find_images(self.dataset_path)
        self.n_complete_batches = int(len(search_result) / batch_size)
        for i in range(self.n_complete_batches):
            batch = search_result[i*batch_size:(i+1)*batch_size]
            images_A, images_B = [], []
            for image_path in batch:
                combined_image = self.imread(image_path)
                h, w, c = combined_image.shape
                nW = int(w/2)
                image_A = combined_image[:, :nW, :]
                image_B = combined_image[:, nW:, :]
                image_A = skimage.transform.resize(image_A, (self.image_height, self.image_width))
                image_B = skimage.transform.resize(image_B, (self.image_height, self.image_width))
                if not for_testing and np.random.random() > 0.5:
                    # 数据增强，左右翻转
                    image_A = np.fliplr(image_A)
                    image_B = np.fliplr(image_B)
                    pass
                images_A.append(image_A)
                images_B.append(image_B)
                pass
            images_A = np.array(images_A)/127.5 - 1.
            images_B = np.array(images_B)/127.5 - 1.
            yield images_A, images_B
     
    
    @property
    def scale(self):
        return self.__scale
    
    @scale.setter
    def scale(self, value):
        if not isinstance(value, int):
            raise ValueError("scale must be int")
        elif value <= 0:
            raise ValueError("scale must > 0")
        else:
            self.__scale = value
            pass
        pass
    
    @property
    def crop_size(self):
        return self.__crop_size
    
    @crop_size.setter
    def crop_size(self, value):
        if not isinstance(value, int):
            raise ValueError("crop size must be int")
        elif value <= 0:
            raise ValueError("crop size must > 0")
        else:
            self.__crop_size = value
            pass
        pass
    
    def imread(self, path):
        return Image.open(path)
    
    def resize(self, image, size):
        resamples = [Image.NEAREST, Image.BILINEAR, Image.HAMMING, \
                     Image.BICUBIC, Image.LANCZOS]
        resample = random.choice(resamples)
        return image.resize(size, resample=resample)
    
    def gaussianblur(self, image, radius=2):
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def medianfilter(self, image, size=3):
        return image.filter(ImageFilter.MedianFilter(size=size))
    
    def downsampling(self, image):
        resize = (image.size[0]//self.scale, image.size[1]//self.scale)
        hidden_scale = random.uniform(1, 3)
        hidden_resize = (int(resize[0]/hidden_scale), int(resize[1]/hidden_scale))
        radius = random.uniform(1, 3)
        image = self.gaussianblur(image, radius)
        image = self.resize(image, hidden_resize)
        image = self.resize(image, resize)
        return image
    
    def search(self, setpath):
        results = []
        files = os.listdir(setpath)
        for file in files:
            path = os.path.join(setpath, file)
            results.append(path)
            pass
        return results
    
    def rotate(self, lr, hr):
        angle = random.choice([0, 90, 180, 270])
        lr = lr.rotate(angle, expand=True)
        hr = hr.rotate(angle, expand=True)
        return lr, hr
    
    def flip(self, lr, hr):
        mode = random.choice([0, 1, 2, 3])
        if mode == 0:
            pass
        elif mode == 1:
            lr = lr.transpose(Image.FLIP_LEFT_RIGHT)
            hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
            pass
        elif mode == 2:
            lr = lr.transpose(Image.FLIP_TOP_BOTTOM)
            hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
            pass
        elif mode == 3:
            lr = lr.transpose(Image.FLIP_LEFT_RIGHT)
            hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
            lr = lr.transpose(Image.FLIP_TOP_BOTTOM)
            hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
            pass
        return lr, hr
    
    def crop(self, lr, hr):
        hr_crop_size = self.crop_size
        lr_crop_size = hr_crop_size//self.scale
        lr_w = np.random.randint(lr.size[0] - lr_crop_size + 1)
        lr_h = np.random.randint(lr.size[1] - lr_crop_size + 1)
        hr_w = lr_w * self.scale
        hr_h = lr_h * self.scale
        lr = lr.crop([lr_w, lr_h, lr_w+lr_crop_size, lr_h+lr_crop_size])
        hr = hr.crop([hr_w, hr_h, hr_w+hr_crop_size, hr_h+hr_crop_size])
        return lr, hr
    
    def pair(self, fp):
        hr = self.imread(fp)
        lr = self.downsampling(hr)
        lr, hr = self.rotate(lr, hr)
        lr, hr = self.flip(lr, hr)
        lr, hr = self.crop(lr, hr)
        lr =  np.asarray(lr)
        hr =  np.asarray(hr)
        return lr, hr
    
    def batches(self, setpath="datasets/train", batch_size=16, complete_batch_only=False):
        images = self.search(setpath)
        sizes = []
        for image in images:
            array = plt.imread(image)
            sizes.append(array.shape[0])
            sizes.append(array.shape[1])
            pass
        crop_size_max = min(sizes)
        crop_size = min(crop_size_max, self.crop_size)
        if self.crop_size != crop_size:
            self.crop_size = crop_size
            print("Info: crop size adjusted to " + str(self.crop_size) + ".")
            pass
        np.random.shuffle(images)
        n_complete_batches = int(len(images)/batch_size)
        self.n_batches = int(len(images) / batch_size)
        have_res_batch = (len(images)/batch_size) > n_complete_batches
        if have_res_batch and complete_batch_only==False:
            self.n_batches += 1
            pass
        for i in range(n_complete_batches):
            batch = images[i*batch_size:(i+1)*batch_size]
            lrs, hrs = [], []
            for image in batch:
                lr, hr = self.pair(image)
                lrs.append(lr)
                hrs.append(hr)
                pass
            lrs = np.array(lrs)
            hrs = np.array(hrs)
            yield lrs, hrs
        if self.n_batches > n_complete_batches:
            batch = images[n_complete_batches*batch_size:]
            lrs, hrs = [], []
            for image in batch:
                lr, hr = self.pair(image)
                lrs.append(lr)
                hrs.append(hr)
                pass
            lrs = np.array(lrs)
            hrs = np.array(hrs)
            yield lrs, hrs
        pass
        
    pass
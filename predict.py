import skimage
import imageio
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model,load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pix2pix import Pix2Pix
from PIL import Image
import tensorflow as tf
from PIL import Image
from keras import backend as K
from keras.losses import mean_absolute_error, mean_squared_error
import random
from WDSR import wdsr_b
from utils import DataLoader
import cv2

data_loader = DataLoader()

def combine(bottom_pic,top_pic,alpha,beta,gamma,save_pth):
    bottom = cv2.imread(bottom_pic)
    top = cv2.imread(top_pic)
    h, w, _ = bottom.shape
    img2 = cv2.resize(top, (w,h), interpolation=cv2.INTER_AREA)
    overlapping = cv2.addWeighted(bottom, alpha, img2, beta, gamma)
    cv2.imwrite(save_pth, overlapping)

def predict_single_image(pix2pix,wdsr, image_path, save_path):
    pix2pix.generator.load_weights('./weights/generator_weights.h5')
    wdsr.load_weights('./weights/wdsr-b-32-x4.h5')
    image_B = imageio.imread(image_path, pilmode='RGB').astype(np.float)
    image_B = skimage.transform.resize(image_B, (pix2pix.nW, pix2pix.nH))
    images_B = []
    images_B.append(image_B)
    images_B = np.array(images_B)/127.5 - 1.
    generates_A = pix2pix.generator.predict(images_B)
    generate_A = generates_A[0]
    generate_A = np.uint8((np.array(generate_A) * 0.5 + 0.5) * 255)
    generate_A = Image.fromarray(generate_A)
    generated_image = Image.new('RGB', (pix2pix.nW, pix2pix.nH))
    generated_image.paste(generate_A, (0, 0, pix2pix.nW, pix2pix.nH))
    lr = np.asarray(generated_image)
    x = np.array([lr])
    y = wdsr.predict(x)
    y = np.clip(y, 0, 255)
    y = y.astype('uint8')
    sr = Image.fromarray(y[0])
    sr.save(save_path)
    combine(image_path,save_path,0.5,0.5,0,save_path)
    pass

gan = Pix2Pix()
wdsr = wdsr_b(scale=4, num_res_blocks=32)

predict_single_image(gan,wdsr, '1.jpg', 'test_1.jpg')
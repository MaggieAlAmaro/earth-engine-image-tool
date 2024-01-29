
#import rasterio
import matplotlib.pyplot as plt
import random
import numpy as np
import argparse
from PIL import Image
from itertools import product
import os, sys, math
# import albumentations
# from albumentations import RandomRotate90

from Image import MyImage, RGBAImage, GrayscaleImage

from utils import processDirOrFile, newFilename, makeOutputFolder
from process import Processor



# @ Functionality adapted from Marian Stefanescu, Towards Data Science, Medium
# https://towardsdatascience.com/measuring-enhancing-image-quality-attributes-234b0f250e10
# Credit to Darel Rex Finley, https://alienryderflex.com/hsp.html
class BrightnessSort(Processor):
    def __init__(self, config_args) -> None:
        self.outputDir = makeOutputFolder('brightness')
        self.dark = os.path.join(self.outputDir,'dark')
        self.bright = os.path.join(self.outputDir,'bright')
        os.mkdir(self.dark)
        os.mkdir(self.bright)
        #Very dark, dark, Normal, Bright, Very Bright



    def pixel_brightness(self, rgbVector) -> float:
        r, g, b = rgbVector
        return math.sqrt(0.299 * r ** 2 + 0.587 * g ** 2 + 0.114 * b ** 2)
    
    def image_brightness(self, data, size):
        nr_of_pixels = size[0]* size[1]
        brightness = 0
        for i in range(size[0]):
            for j in range(size[1]):
                brightness += self.pixel_brightness(data[i][j])
                # func = np.vectorize(self.pixel_brightness)
        return brightness/ nr_of_pixels

    
    def sortToDir(self, data, size):
        brightness = self.image_brightness(data, size)
        print("Bright value: ", brightness)
        return self.dark if brightness < 215 else self.bright

        
    def process(self, image: MyImage):
        # only for rgb or rgba images
        assert(len(image.shape) > 2)
        resultOutDir = self.sortToDir(image.data[:2], image.shape)
        fn = newFilename(image.image_filename, suffix=".png", outdir=resultOutDir)
        image.image.save(fn)


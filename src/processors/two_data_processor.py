#import rasterio
import matplotlib.pyplot as plt
import random
import numpy as np
import argparse
from PIL import Image
from itertools import product
import os, sys, math
from abc import ABC, abstractmethod
# import albumentations
# from albumentations import RandomRotate90

from utils import processDirOrFile, newFilename, makeOutputFolder

from src.processors.process import Processor

from Image import *


def zipData(data_path1, data_path2):
    img1WPath = [os.path.join(data_path1, file) for file in os.listdir(data_path1)]
    img2WPath = [os.path.join(data_path2, file) for file in os.listdir(data_path2)]
    return zip(img1WPath,img2WPath)


class Subtract(Processor):
    def __init__(self, config_args) -> None:
        super().__init__()
        self.mean = []
        self.data_target = config_args['data_target']
        # self.outputDir = makeOutputFolder('subtract')


    def subtractImages(self, zippedData):
        for image1_path, image2_path in zippedData:
            img1 = Image.open(image1_path)
            img2 = Image.open(image2_path)
            data1 = np.array(img1)
            data2 = np.array(img2)
            #result = abs(data1 - data2)
            #img = Image.fromarray(result)
            mse = (np.square(data1 - data2)).mean()
            #print("MSE:", mse)
            self.mean.append(mse)

        
    def process(self, data_path: str):
        self.outputDir = data_path
        if(os.path.isdir(data_path) and os.path.isdir(self.data_target)):
            self.subtractImages(zipData(data_path,self.data_target))
            print("Mean of MSE:", np.array(self.mean).mean())
        else:
            raise Exception("Can only compare directories.")
    



class Compare(Processor):
    def __init__(self, config_args):
        super().__init__()
        self.data_target = config_args['data_target']
        self.exitOnDifferent = config_args['exitOnDifferent']
        self.areAllSame = True

    def compareImage(self, image1, image2):
        img1 = Image.open(image1)
        img2 = Image.open(image2)
        data1 = np.array(img1)
        data2 = np.array(img2)
        if(data1.shape != data2.shape):
            print("The image shapes don't match for {i1} and {i2}.".format(i1=image1,i2=image2))
            return False      
        elif((data1 == data2).all()):
            # print("The images are EQUAL.")
            return True
        else:
            print("The images {i1} and {i2} are NOT equal.".format(i1=image1,i2=image2))
            return False

    def compareImages(self, zippedData):
        for image1, image2 in zippedData:
            same = self.compareImage(image1, image2)
            if not same: self.areAllSame = False 
            if not same and self.exitOnDifferent:
                return

    def process(self, data_path: str):
        self.outputDir = data_path
        if(os.path.isdir(data_path) and os.path.isdir(self.data_target)):
            self.compareImages(zipData(data_path,self.data_target))
            if self.areAllSame: print("All images are EQUAL.")
        else:
            raise Exception("Can only compare directories.")
        
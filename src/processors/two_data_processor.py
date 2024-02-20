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

from src.utils import processDirOrFile, newFilename, makeOutputFolder

from src.processors.process import Processor

from Image import *

def sortNumerically(fileName):
    return int(os.path.basename(fileName).split('.').split('_')[1])


def zipData(data_path1, data_path2):
    img1WPath = [os.path.join(data_path1, file) for file in os.listdir(data_path1)]
    img2WPath = [os.path.join(data_path2, file) for file in os.listdir(data_path2)]
    # img1WPath = sorted(img1WPath, key=sortNumerically)
    # img2WPath = sorted(img2WPath, key=sortNumerically)
    return zip(img1WPath,img2WPath)


class Subtract(Processor):
    def __init__(self, config_args) -> None:
        super().__init__()
        self.mean = []
        self.data_target = config_args['data_target']
        self.channels_to_compare = config_args['channels_to_compare']
        self.nbr_channels = config_args['nbr_channels']
        # self.outputDir = makeOutputFolder('subtract')


    def subtractImages(self, zippedData):
        for image1_path, image2_path in zippedData:
            img1 = Image.open(image1_path)
            img2 = Image.open(image2_path)
            data1 = np.array(img1)
            data2 = np.array(img2)
            if(self.nbr_channels==1):
                mse = (np.square(data1-data2)).mean()
            else:
                data1 = np.transpose(data1, (2,0,1))
                data2 = np.transpose(data2, (2,0,1))
                mse = 0
                for channel in self.channels_to_compare:
                    mse += (np.square(data1[channel]-data2[channel])).mean()
                mse = mse/len(self.channels_to_compare)

            #TODO: this here below
            #np.concatenate(([data1[ch] for ch in self.channels_to_compare]),axis=1)

            # r,g,b,a1 = img1.split()
            # r,g,b,a2 = img2.split()
            # a1= np.array(a1)
            # a2= np.array(a2)
            # print(data1[3])
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

    def compareData(data1, data2):
        if(data1.shape != data2.shape):
            print("The image shapes don't match")
            return False      
        elif((data1 == data2).all()):
            print("The images are EQUAL.")
            return True
        else:
            print("The images are NOT equal.")
            return False

    def compareImage(image1, image2):
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
            same = Compare.compareImage(image1, image2)
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
        
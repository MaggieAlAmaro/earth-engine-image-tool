import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.merge import merge
import numpy as np
import argparse
from PIL import Image
from itertools import product
import os, sys, math, typing


from utils import processDirOrFile

def getParser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs, formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")
    mergeParser = subparsers.add_parser("compare", help="check if 2 images are the same")
    mergeParser.add_argument('image1', type=str, help='Name of image or directory to merge with.')    
    mergeParser.add_argument('image2', type=str, help='Name of image or directory to merge with.')  

    mergeParser = subparsers.add_parser("size-check", help="check if 2 images are the same")
    mergeParser.add_argument('image', type=str, help='Name of image or directory to merge with.')    
    mergeParser.add_argument('size', type=str, help='Name of image or directory to merge with.')    
    #return mergeParser, seperateParser
    return parser


# ONLY CHECKS IF VALUES ARE LESS THAN EXPECTED VALUE
def sizeCheck(image,  **kwargs):
    size = kwargs.get('size')
    img = Image.open(image)
    if type(size) is tuple:
        if(img.size[0] < size[0] or img.size[1] < size[1]):
            print("Image " + image + " is less then expected size")
            print(img.size)
    elif type(size) is int:
        if(img.size[0] < size or img.size[1] < size):
            print("Image " + image + " is less then expected size")
            print(img.size)

    # with rasterio.open(f) as img:
    #     bandNbr = [band for band in range(1,img.count + 1)]
    #     data = img.read(bandNbr)
    #     print(data.shape == expectedSize)
             


def compare(img1, img2, **kwargs):
    img1 = Image.open(img1)
    img2 = Image.open(img2)
    img1 = np.array(img1)
    img2 = np.array(img2)
    if(img1.shape != img2.shape):
        print("The image shapws don't match:" + str(img1.shape) + "and" + str(img2.shape))
        return False      
    elif((img1 == img2).all()):
        print("The images are EQUAL.")
        return True
    else:
        print("The images are NOT equal.")
        return False

    
if __name__ == '__main__':
    parser = getParser()
    args = parser.parse_args()
    print(args)
    if args.command == 'compare':
        processDirOrFile(compare, target=args.image1, destination=args.image2)
    if args.command == 'size-check':
        processDirOrFile(sizeCheck, target=args.image, size=args.size)

        
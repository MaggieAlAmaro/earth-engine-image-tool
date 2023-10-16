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

    mergeParser = subparsers.add_parser("sizeCheck", help="check if 2 images are the same")
    mergeParser.add_argument('image', type=str, help='Name of image or directory to merge with.')    
    mergeParser.add_argument('size', type=int, help='Name of image or directory to merge with.')    
    mergeParser.add_argument('type', type=str, help='Name of image or directory to merge with.', choices={'exact','less'})    
    #return mergeParser, seperateParser
    return parser


# ONLY CHECKS IF VALUES ARE LESS THAN EXPECTED VALUE
def sizeCheck(image,  **kwargs):
    size = kwargs.get('size')
    t = kwargs.get('type', 'exact')
    img = Image.open(image)
    
    if t == 'less':
        if type(size) is tuple:
            expr = img.size[0] < size[0] or img.size[1] < size[1]
        elif type(size) is int:
            expr = img.size[0] < size or img.size[1] < size
    elif t == 'exact':
        if type(size) is tuple:
            expr = img.size[0] != size[0] or img.size[1] != size[1]
        elif type(size) is int:
            expr = img.size[0] != size or img.size[1] != size
    if expr:
        print("Image " + image + " is NOT of expected size")
        print(img.size)

    # with rasterio.open(f) as img:
    #     bandNbr = [band for band in range(1,img.count + 1)]
    #     data = img.read(bandNbr)
    #     print(data.shape == expectedSize)
             


def compare(image1, image2, **kwargs):
    img1 = Image.open(image1)
    img2 = Image.open(image2)
    img1 = np.array(img1)
    img2 = np.array(img2)
    if(img1.shape != img2.shape):
        print("The image shapes don't match:" + str(img1.shape) + "and" + str(img2.shape))
        print(image1)
        print(image2)
        return False      
    elif((img1 == img2).all()):
        #print("The images are EQUAL.")
        return True
    else:
        print("The images are NOT equal.")
        return False

def compareDir(target, dest, **kwargs):
    if(os.path.isdir(target) and os.path.isdir(dest)):
        for target_file, destination_file in zip(os.listdir(target), os.listdir(dest)):
            target_f = os.path.join(target, target_file)
            destination_f = os.path.join(dest, destination_file)
            if os.path.isfile(target_f) and os.path.isfile(destination_f):
                if not compare(target_f,destination_f):
                    return False
            
    else:
        return False
    print("All images are EQUAL.")
    return True

    
if __name__ == '__main__':
    parser = getParser()
    args = parser.parse_args()
    print(args)
    if args.command == 'compare':
        compareDir(target=args.image1, dest=args.image2)
        #processDirOrFile(compare, target=args.image1, destination=args.image2)
    if args.command == 'sizeCheck':
        processDirOrFile(sizeCheck, target=args.image, size=args.size, type=args.type)

        
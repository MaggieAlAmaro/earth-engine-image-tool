import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.merge import merge
import numpy as np
import argparse
from PIL import Image
from itertools import product
import os, sys, math


from utils import processDirOrFile

def getParser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs, formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")
    mergeParser = subparsers.add_parser("compare", help="check if 2 images are the same")
    mergeParser.add_argument('rgb_name', type=str, help='Name of image or directory to merge with.')    
    mergeParser.add_argument('a_name', type=str, help='Name of image or directory to merge with.')    
    mergeParser.add_argument('-o','--out_dir', type=str, help='Output Directory')
    #return mergeParser, seperateParser
    return parser


def sizeCheck(imageDir, expectedSize):
    if(os.path.isdir(imageDir)):
        for filename in os.listdir(imageDir):
            f = os.path.join(imageDir, filename)
            # check if file
            if os.path.isfile(f):
                if(f.split(".")[-1] == 'png'):
                    img = Image.open(f)
                    if(img.size[0] < 1024 or img.size[1] < 1024  ):
                        print(img.size, img)
                elif(f.split(".")[-1] == 'tif' or f.split(".")[-1] == 'tiff'):
                    with rasterio.open(f) as img:
                        bandNbr = [band for band in range(1,img.count + 1)]
                        data = img.read(bandNbr)
                        print(data.shape == expectedSize)
             

    
if __name__ == '__main__':
    parser = getParser()
    args = parser.parse_args()
    print(args)
    if args.command == 'compare':
        processDirOrFile(mergeRGBA, rgb=args.rgb_name, a=args.a_name)

        
    if(len(args.compare) == 2):
        if(args.compare[0].split(".")[-1] != 'png' or args.compare[1].split(".")[-1] != 'png'):
            print("Only PNG is supported!")
        else:
            img1 = Image.open(args.compare[0])
            img2 = Image.open(args.compare[1])
            img1 = np.array(img1)
            img2 = np.array(img2)
            if(img1.shape != img2.shape):
                print("The image sizes don't match:" + str(img1.shape) + "and" + str(img2.shape))
            else:
                print(img2.shape)
                if((img1 == img2).all()):
                    print("The images are EQUAL.")
                else:
                    print("The images are NOT equal.")
    else:
        print("Must specify only 2 files!")
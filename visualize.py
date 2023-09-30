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

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs, formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")
    openParser = subparsers.add_parser("open", help="check if 2 images are the same")
    openParser.add_argument('name', type=str, help='Name of image or directory to merge with.')
    openParser.add_argument('--plot', type=bool, help='If true plots using matplotlib, if false opens with standard image')
    #return mergeParser, seperateParser
    return parser


def open(img, **kwargs):


#
    img = Image.open(img)

    print(np.array(img))
    img.show()
    print(img.size)
    
        # with rasterio.open(args.name) as img:
        #     bandNbr = [band for band in range(1,img.count + 1)]
        #     data = img.read(bandNbr)
        #     print(data.shape)
        #     #if grayscale
        #     if (len(bandNbr) == 1):
        #         show(data,cmap='gray',vmin=0, vmax=255)
        #         #show(data,cmap='gray') #if min-max are not set it will use the min and max values found in the array to set a pixel range... im assuming
        #     else:
        #         show(data)

#     if (args.name is not None):
#         if(args.name.split(".")[-1] == 'png'):
#             img = Image.open(args.name)
#             print(img.size)
#             print(img.getpixel((30,200)))
#             print(np.array(img))
#             img.show()


# #
#         elif(args.name.split(".")[-1] == 'tif' or args.name.split(".")[-1] == 'tiff'):
#             with rasterio.open(args.name) as img:
#                 bandNbr = [band for band in range(1,img.count + 1)]
#                 data = img.read(bandNbr)
#                 print(data.shape)
#                 #if grayscale
#                 if (len(bandNbr) == 1):
#                     show(data,cmap='gray',vmin=0, vmax=255)
#                     #show(data,cmap='gray') #if min-max are not set it will use the min and max values found in the array to set a pixel range... im assuming
#                 else:
#                     show(data)
                    

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if args.command == 'open':
        open(args.name, plot=args.plot) #Dont process diirirrr

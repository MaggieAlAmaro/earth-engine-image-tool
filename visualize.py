import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import numpy as np
from PIL import Image
import argparse
from itertools import product
#import rasterio
import os
import math
import rasterio

plt.style.use('dark_background')

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs, formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")
    openParser = subparsers.add_parser("open", help="Open image")
    openParser.add_argument('name', type=str, help='Name of image to open')
    openParser.add_argument('--plot', type=bool, default=None,
                             help='If true plots using matplotlib, if false opens with standard image viewer')

    gridParser = subparsers.add_parser("grid", help="Display set of images in a grid",formatter_class=argparse.RawTextHelpFormatter)
    gridParser.add_argument('imgdir', type=str, 
                            help="""Name of Image directory.  
This folder EITHER: 
    - Contains images. Will be displayed in the appropriate grid size.
    - Contains folders containing images. The images in a folder will be displayed in a row.
        All folders MUST contain the same amount of images.
    """)
    return parser



# def open(img, **kwargs):
#     img = Image.open(img)
#     print("Shape: ", img.size)
#     print("Mode: ", img.mode)
#     img.show()
def open(img, **kwargs):
    imgPIL = Image.open(img)
    a = imgPIL.convert('L')
    a.save("rgb.png")

    imgPIL.show()
    print("Shape: ", imgPIL.size)
    print("Mode: ", imgPIL.mode)
    rasImg = rasterio.open(img)
    print(rasImg.bounds)
    print(rasImg.scales)
    # rasImg.read(1)
    # scales, statistics[1,2,3], window, window_bounds,bounds
    # img.show()

    # with rasterio.open(args.name) as img:
    #     bandNbr = [band for band in range(1,img.count + 1)]
    #     data = img.read(bandNbr)
    #     show(data,cmap='gray',vmin=0, vmax=255)

                    

#TODO get adequate grid shape via the number of images. The current solution wont work for prime numbers for example
def makeSquareImageGrid(array):
    n, h, w, c = array.shape
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n/ncols)

    result = []
    for j in range(nrows):
        row = array[j*ncols]
        for i in range(1, ncols):
            if j*ncols + i >= n:
                break
            row = np.c_['1',row, array[j*ncols + i]]
        if j==nrows-1: #and ncols != nrows:        
            extra = (nrows * ncols) - n
            print(row.shape)
            extraArr = np.full((h, w*extra, c),125, dtype=np.uint16)
            print(extraArr.shape)
            row = row = np.c_['1',row, extraArr]
            print(row.shape)

        if len(result) != 0:
            result = np.r_['0',result, row]
        else:
            result = row
    return result


def makeSimpleGrid(array, ncols, order='H'):
    n, h, w, c = array.shape
    # print(np.transpose(result, axes=(0,1,2)).shape)
    nrows = n//ncols
    assert n == nrows*ncols
    if order == 'H':
        return (array.reshape((nrows, ncols, h, w, c))
                .swapaxes(1,2)
                .reshape(h*nrows, w*ncols, c))
    elif order == 'V':
        return (array.reshape((ncols,nrows, h, w, c),order='F')
                .swapaxes(1,2)
                .reshape( w*ncols,h*nrows, c))
    else:
        raise Exception("Wrong Order, Choose V or H (vertical, horizontal)")
    



def openImageList(logdir):
    imgData = []    
    if type(logdir) is list:
        for img in logdir:
            image = Image.open(img)
            imgData.append(np.array(image.convert('RGB')))
    imgData = np.array(imgData)
    return imgData

def openImageDirectory(logdir, mode='RGB'):
    imgData = []
    dirOrFiles = os.listdir(logdir)
    # dirOrFiles.sort()
    if (os.path.isfile(os.path.join(logdir,file)) for file in dirOrFiles):
        for img in dirOrFiles:
            img = Image.open(os.path.join(logdir,img))
            imgData.append(convertToArrayWithSameChannels(img,mode))
    imgData = np.array(imgData)
    # grid = makeSimpleGrid(imgData, size)
    return imgData


def convertToArrayWithSameChannels(image: Image.Image, mode):
    data = np.array(image.convert(mode))
    if(len(data.shape) == 2):
        return data.reshape(data.shape[0],data.shape[1],1)
    else: return data

def sortNumerically(fileName):
    return int(fileName.split('.')[0])

def openDirsForComparison(logdir, mode='RGB'):
    imgData = []
    dirs = os.listdir(logdir)
    size = 0

    if all(os.path.isdir(os.path.join(logdir,content)) for content in dirs):
        size = len(os.listdir(os.path.join(logdir, dirs[0])))
        for dir in dirs:
            print("In dir:", dir)
            dirPath = os.path.join(logdir,dir)
            imgInDir = os.listdir(dirPath)
            if size != len(imgInDir):
                print("All folders must have the same amount of elements.")
                return 
            imgInDir = sorted(imgInDir, key=sortNumerically)
            dirImgs = [Image.open(os.path.join(dirPath,img)) for img in imgInDir]
            dirImgsArr = [convertToArrayWithSameChannels(img,mode) for img in dirImgs]
            imgData.extend(dirImgsArr)
    else:
        print("Not all content of directory is a folder")
        return
    imgData = np.array(imgData)
    return imgData, size
      


if __name__ == '__main__':
    # parser = get_parser()
    # args = parser.parse_args()


    # if args.command == 'open':
    #     open(args.name, plot=args.plot)        
    # elif args.command == 'grid':
    
    # data = openImageDirectory('results\\LDM_4Channel_v2\\18', 'RGBA')
    # grid = makeSquareImageGrid(data)

    open('data\\5426.tif')

    #TODO: catch if images are different size. must all be same size
    # data, size = openDirsForComparison('data\\test', 'RGB')
    # grid = makeSimpleGrid(data, size, order='V')


    # plt.figure(figsize=(data.shape[1]/2, data.shape[2]/2))
    # plt.get_current_fig_manager().window.showMaximized()
    # plt.imshow(grid)
    # plt.tight_layout()
    # plt.show()


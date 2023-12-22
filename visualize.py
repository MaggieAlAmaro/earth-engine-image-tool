import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from PIL import Image
import argparse
from itertools import product
#import rasterio
import os
import math

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



def open(img, **kwargs):
    img = Image.open(img)
    print("Shape: ", img.size)
    print("Mode: ", img.mode)
    img.show()

    # with rasterio.open(args.name) as img:
    #     bandNbr = [band for band in range(1,img.count + 1)]
    #     data = img.read(bandNbr)
    #     show(data,cmap='gray',vmin=0, vmax=255)

                    

#TODO get adequate grid shape via the number of images. The current solution wont work for prime numbers for example
def makeImageGrid(array, ncols=1):
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
    #print(result.shape)
    # nrows = n//ncols
    # assert n == nrows*ncols
    # result = (array.reshape(nrows, ncols, h, w, c)
    #           .swapaxes(1,2)
    #           .reshape(h*nrows, w*ncols, c))
    return result



def grid(logdir, rows):
    imgData = []
    if type(logdir) is list:
        for img in logdir:
            image = Image.open(img)
            imgData.append(np.array(image.convert('RGB')))

    elif(os.path.isdir(logdir)):
        dirOrFiles = os.listdir(logdir)
        dirOrFiles.sort()
        #dirs inside logdir
        if all(os.path.isdir(os.path.join(logdir,content)) for content in dirOrFiles):
            for dir in dirOrFiles:
                print(dir)
                dirPath = os.path.join(logdir,dir)
                imgInDir = os.listdir(dirPath)
                dirImgs = [Image.open(os.path.join(dirPath,img)) for img in imgInDir]
                dirImgsArr = [np.array(img.convert('RGB')) for img in dirImgs] #!!!!
                imgData.extend(dirImgsArr)

        #logdir
        elif (os.path.isfile(os.path.join(logdir,file)) for file in dirOrFiles):
            for img in dirOrFiles:
                img = Image.open(os.path.join(logdir,img))
                imgData.append(np.array(img.convert('RGB')))

    imgData = np.array(imgData)
    grid = makeImageGrid(imgData,rows)
    plt.imshow(grid)
    plt.show()

      


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if args.command == 'open':
        open(args.name, plot=args.plot)        
    elif args.command == 'grid':
        grid(args.imgdir,2)

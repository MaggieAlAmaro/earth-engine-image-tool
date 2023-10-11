import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from PIL import Image
import argparse
from itertools import product
import os

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


def open(img):
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
    nrows = n//ncols
    assert n == nrows*ncols
    result = (array.reshape(nrows, ncols, h, w, c)
              .swapaxes(1,2)
              .reshape(h*nrows, w*ncols, c))
    return result



def grid(logdir):
    imgData = []
    print(logdir)
    if(os.path.isdir(logdir)):
        dirOrFiles = os.listdir(logdir)
        #dirs inside logdir
        if all(os.path.isdir(os.path.join(logdir,content)) for content in dirOrFiles):
            for dir in dirOrFiles:
                dirPath = os.path.join(logdir,dir)
                imgInDir = os.listdir(dirPath)
                dirImgs = [Image.open(os.path.join(dirPath,img)) for img in imgInDir]
                dirImgsArr = [np.array(img.convert('RGB')) for img in dirImgs]
                imgData.extend(dirImgsArr)

        #logdir
        elif (os.path.isfile(os.path.join(logdir,file)) for file in dirOrFiles):
            for img in dirOrFiles:
                img = Image.open(os.path.join(logdir,img))
                imgData.append(np.array(img.convert('RGB')))

        imgData = np.array(imgData)
        grid = makeImageGrid(imgData,2)
        plt.imshow(grid)
        plt.show()

      


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if args.command == 'open':
        open(args.name, plot=args.plot)        
    elif args.command == 'grid':
        grid(args.imgdir)

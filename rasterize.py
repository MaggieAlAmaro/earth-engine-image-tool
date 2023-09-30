import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.merge import merge
import numpy as np
import argparse
from PIL import Image
from itertools import product
import os, sys, math

plt.style.use('dark_background')

def changeBitDepth(image):
    img = rasterio.open(image)
    data = img.read()
    min= -10
    max= 6500
    normalizedData = (data - min)/(max-min)
    rescaledData = normalizedData * (255 - 0)
    
    rescaledData = rescaledData.astype('uint8')
    newImg = Image.fromarray(rescaledData[0])
    print(rescaledData.shape)
    print(rescaledData)
    newImg.save("srtm16bit-8bit--10.png")
    

def treatNumpy(image: str):
    img = rasterio.open(image)
    imageData = img.read()
    min= -10
    max= 6500
    newMin = np.cbrt(min)
    newMax = np.cbrt(max)

    bitDepth = 255


    # range scaling explained: https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
    #normalize funct: newX = x - min/ max - min
    #newval = np.sqrt(imageData)
    normalizedData = (np.cbrt(imageData) - newMin)/(newMax-newMin)
    rescaledData = normalizedData * (bitDepth - 0)
    rescaledData = rescaledData.astype('uint8')
    newImg = Image.fromarray(rescaledData[0])
    newImg.save("srtm16bit-8bit-cqrt-10.png")




# given that the SRTM image has a range estimate (according to earth engine) of:
# [-10, 6500], which I will take as [0,6500] just to make it easier (no sqrt of negative nbrs) -- after try with cube root
# TODO: Use ee.reducer.max() and .min() to find the actual value limits of the SRTM image.
# SOLUTIONS: because it cant calculate the max over the whole image, do max/min convolution over it until getting true values
def treat(x):
    min= 0
    max= 6500
    newMin = math.sqrt(min)
    newMax = math.sqrt(max)



    # range scaling explained: https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
    #normalize funct: newX = x - min/ max - min
    newval = math.sqrt(x)
    newX = (newval - newMin)/(newMax-newMin)

def tile(imageDir, tileSize):
    if(os.path.isdir(imageDir)):
        for filename in os.listdir(imageDir):
            f = os.path.join(imageDir, filename)
            # check if file
            if os.path.isfile(f):
                if(f.split(".")[-1] == 'png'):
                    img = Image.open(f)
                    w, h = img.size
                    if(h%tileSize != 0 or w%tileSize != 0):
                        print("Tile size must be devisable by the size of the image.")
                        return
                    for i, j in product(range(0, h, tileSize), range(0, w, tileSize)):
                        newFileName = "256_"+ str(i) + "_" + str(j) +"_" + f.split(os.sep)[-1]
                        out = os.path.join("256", newFileName)
                        box = (j, i, j+tileSize, i+tileSize)
                        img.crop(box).save(out)



def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "fileOrDir",
        type=str,
        nargs="?",
        action='store_const',
        help="Name of image or directory to work with.",
        default=openImg
        const
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        nargs="?",
        help="name of raster image",
    )
    parser.add_argument(
        "-m",
        "--merge",
        nargs='+',
        type=str,
        help="merge 2 tif images",
    )
    parser.add_argument(
        "-cm",
        "--channelMerge",
        nargs='+',
        type=str,
        help="merge an rgb and a 1 channel image",
    )
    parser.add_argument(
        "-c",
        "--compare",
        nargs='+',
        type=str,
        help="compares 2 images and evaluates if they are the same",
    )
    parser.add_argument(
        "-t",
        "--toType",
        nargs='+',
        type=str,
        help="convert img To Png",
    )
    parser.add_argument(
        "-s",
        "--separate",
        const=True,
        nargs="?",
        type=str,
        help="Separate RGB and A from RGBA image into 2 seperate images",
    )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()






             




    else:
        #img = Image.open("srtm16bit.tif")
        #img = img.convert(mode='I;16')
        #img.save("test.png")
        print()
        #img = rasterio.open("srtm16bit.tif")#sys.argv[1])
        #data = img.read()
        #newdata = changeBitDepth("srtm16bit.tif")
        #print(np.min(newdata), newdata.dtype)
        #newImg = Image.fromarray(newdata[0])
        #newImg.save("theEnd.png")



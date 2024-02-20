import json, pickle
import numpy as np
import argparse
from PIL import Image
import PIL
import matplotlib.pyplot as plt

from src.utils import newFilename, makeOutputFolder
# from processors.batch_processor import Histogram

def getParser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs, formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")
    openParser = subparsers.add_parser("open", help="Convert to png")   
    openParser.add_argument('pickle_filename', type=str, help='Name of image')  
    openParser.add_argument('-min', type=int, help='Name of image',default= -32768)#-10)

import rasterio
from itertools import product
def openInfo(filename):
    img = rasterio.open(filename)
    print("Coordinate bounds:", img.bounds)
    print("Crs:", img.crs)
    print()
    print("Shape:", img.width, img.height)
    print("Bands:", img.count)
    # print("For alpha channel:")
    outputDir = makeOutputFolder('tile')
    imga= Image.open(filename)
    print("Mode:", imga.mode)

    # a = img.read(1)
    # print(a[20000,100000])
    # an = a[22000:25024,100000:104024]
    # print(an)
    # a = Image.fromarray(an,'L')
    # fn = newFilename(filename, suffix=f"_{2}_{3}.png", outdir=outputDir)
    # a.save(fn)

    width = int(172801 / int(172801/1024))
    height = int(68683 / int(68683/1024))
    for i, j in product(range(0, 172801, width), range(0, 68683, height)):
        box = (j, i, j+width, i+height)
        fn = newFilename(filename, suffix=f"_{j}_{i}.png", outdir=outputDir)
        a = imga.crop(box)

        e = a.convert('L')
        e.save(fn) 
    print("==============================================")
    print()



def openImage(filename):
    with Image.open(filename) as f:
        f.show()

        
def openImagePixelData(filename):
    with open(filename, 'rb') as f:
        pixels: np.array = pickle.load(f)
        return pixels
    
# def openHistogramPickle(filename) -> Histogram:
#     with open(filename, 'rb') as f:
#         hist: Histogram = pickle.load(f)
#         return hist




if __name__ == '__main__':
    # hist = openHistogramPickle("output\\GRAYSCALE_TREATED\histogram-2024-01-24-23-39-57\\histogram.pickle")
    # hist.plotDistribution()
    # hist.fitDistributions()

    # openImage("C:\\Users\\Margarida\\Downloads\\EuropeRGB-20240130T180729Z-003\\3420.tif")
    PIL.Image.MAX_IMAGE_PIXELS = 11868491085
    openInfo("C:\\Users\\Margarida\\Downloads\\7.5_arc_second_compressedGeoTIFF\\GlobalTerrainClassification_Iwahashi_etal_2018.tif")
    # openImage("C:\\Users\\Margarida\\Downloads\\7.5_arc_second_compressedGeoTIFF\\GlobalTerrainClassification_Iwahashi_etal_2018.tif")
    # parser = getParser()
    # args = parser.parse_args()
    # print(args)
    # if args.command == 'open':
    #     processDirOrFile(blah)
    # with open("info.json", "r") as f:
    #     data = json.loads(f)
    #     print(data)

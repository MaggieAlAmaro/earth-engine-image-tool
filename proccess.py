import time
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.merge import merge
import numpy as np
import argparse
from PIL import Image
from itertools import product
import os, sys, math

from utils import processDirOrFile, newFilename, makeOutputFolder



def getParser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs, formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")
    mergeParser = subparsers.add_parser("convert", help="Convert to png")   
    mergeParser.add_argument('image_filename', type=str, help='Name of image')    
    # mergeParser.add_argument('-o','--out_filename', type=str, help='Output Directory')


    seperateParser = subparsers.add_parser("scale", help="Scale image values according to function.")
    seperateParser.add_argument('image_filename', type=str, help='Name of image')    
    seperateParser.add_argument('type', type=str, help='Type of scaling', choices=["sqrt","cbrt","log"])
    seperateParser.add_argument('-min', type=int, help='Type of scaling', default=-10)
    seperateParser.add_argument('-max', type=int, help='Type of scaling',default=6501)
    
    cropParser = subparsers.add_parser("crop", help="Crop image to size")
    cropParser.add_argument('image_filename', type=str, help='Name of image')    
    cropParser.add_argument('size', type=int,nargs='+', help='Tuple with height and width of crop')

    
    tileParser = subparsers.add_parser("tile", help="Tile the image into smaller images with the given size.")
    tileParser.add_argument('image_filename', type=str, help='Name of image')    
    tileParser.add_argument('size', type=int, help='Size of tiles')

    
    depthParser = subparsers.add_parser("to8bit", help="Convert image to 8bit")
    depthParser.add_argument('image_filename', type=str, help='Name of image')   
    depthParser.add_argument('-min', type=int, help='Type of scaling', default=-10)
    depthParser.add_argument('-max', type=int, help='Type of scaling',default=6501)
    return parser


#if 16bit (unsigned int), [0, 65455]
#if 16bit (signed int), [-32768, 32767]
def to8bit(image,  **kwargs):
    min = kwargs.get('min',-10)
    max = kwargs.get('max',6500)
    outDir = kwargs.get('outDir','.')
    # img = Image.open(image)
    # img = np.array(img)
    img = rasterio.open(image)

    a = img.read()
    print(img.bounds)
    normalizedData = (a- min)/(max-min)
    rescaledData = normalizedData * (255 - 0)  # 8bit -> [0,255]
    rescaledData = rescaledData.astype('uint8')
    newImg = Image.fromarray(rescaledData[0])
    #fn = newFilename(image, suffix="8bit.tif", outdir=outDir)
    #newImg.save(fn)
    




# given that the SRTM image has a range estimate (according to earth engine) of:
# [-10, 6500], which I will take as [0,6500] just to make it easier (no sqrt of negative nbrs) -- after try with cube root
# TODO: Use ee.reducer.max() and .min() to find the actual value limits of the SRTM image.
# SOLUTIONS: because it cant calculate the max over the whole image, do max/min convolution over it until getting true values
def scale(image: str,  **kwargs):
    img = Image.open(image)
    img = np.array(img)
    min = kwargs.get('min',-10)
    max = kwargs.get('max',6500)
    typeT = kwargs.get('type')
    normalizedData = (img - min)/(max-min)          # Normalize [0,1]
    scalingFunction = None
    if (typeT == 'cbrt'):
        scalingFunction = np.cbrt
    elif (typeT == 'sqrt'):
        scalingFunction = np.sqrt
    elif (typeT == 'log'):
        scalingFunction = np.log
    
    # Scale to range defined by scaling function
    #scaledData = (scalingFunction(normalizedData) - scalingFunction(0)) / (scalingFunction(1) - scalingFunction(0)) 
    scaledData = (normalizedData * (scalingFunction(max) - scalingFunction(min))) + scalingFunction(min)

    print(scaledData)
    scaledNormData = (scaledData - scalingFunction(min))/(scalingFunction(max)-scalingFunction(min))          # Normalize [0,1]

    bit8ScaledData = scaledNormData * (255 - 0) #TODO: check on this formula
    bit8ScaledData = bit8ScaledData.astype('uint8')
    newImg = Image.fromarray(bit8ScaledData)
    newImg.save("srtm16bit-8bit-cqrt-10AAAAA.png")



    # newMin = np.cbrt(min)
    # newMax = np.cbrt(max)
    # normalizedData = (np.cbrt(img) - newMin)/(newMax-newMin) #normalize funct: newX = x - min/ max - min
    # rescaledData = normalizedData * (255 - 0)
    # rescaledData = rescaledData.astype('uint8')
    # newImg = Image.fromarray(rescaledData[0])
    # newImg.save("srtm16bit-8bit-cqrt-10.png")



# Crops excess from right and top
def crop(image, size, **kwargs):
    img = Image.open(image)
    if(img.size[0] > size or img.size[1] > size):
        y0Clip = img.size[0] - size  # clip excess top rows
        croppedImg = img.crop((0, y0Clip, size, img.size[0]))
        newFileName = newFilename(image, outdir="output\\crops")
        croppedImg.save(newFileName)

    elif(img.size[0] < size or img.size[1] < size):
        print("IMAGE SIZE IS LOWER THAN CLIP SIZE.")



def isLineBlack(vector, tolerance = 0):
    return np.sum(vector) <= (0 + tolerance)

def checkSequenceForBlackLines(imgData, axis, idx, size):  
    blackLineCount = 0  
    for i in range(idx,idx+size):
        if(axis == 0):
            if isLineBlack(imgData[i,:]):
                blackLineCount += 1
        if(axis == 1):
            if isLineBlack(imgData[:,i]):
                blackLineCount += 1
    return blackLineCount

def removeBlackLines(image, **kwargs):
    size = kwargs.get('size')
    img = Image.open(image)
    print("Original size:",img.size)
    data = np.array(img)
    if(data.shape == size):
        print("Shape is already the expected shape.")
        return

    
    rowExtra = data.shape[0] - size[0]
    colExtra = data.shape[1] - size[1]


    row0Clip = checkSequenceForBlackLines(data,0,0,rowExtra)
    col0Clip = checkSequenceForBlackLines(data,1,0,colExtra)
    row1Clip = checkSequenceForBlackLines(data,0,data.shape[0]-rowExtra,rowExtra)
    col1Clip = checkSequenceForBlackLines(data,1,data.shape[1]-colExtra,colExtra)

    print("Line Crop Measures:",(col0Clip, row0Clip, data.shape[1] - col1Clip, data.shape[0] - row1Clip))
    croppedImg = img.crop((col0Clip, row0Clip, data.shape[1] - col1Clip, data.shape[0] - row1Clip))
    print("New size:",croppedImg.size)
    newFileName = newFilename(image, suffix="cropped.png",outdir="..\\tif")
    croppedImg.save(newFileName)



#TODO THIS
def convertPNG(oldImage, newFileName=None, destination=None):
    if(newFileName is None):
        f_type = oldImage.split(".")[-1]
        newFileName = oldImage.split(os.sep)[-1]
        newFileName = newFileName.rstrip(("."+f_type)) + ".png"

    img = Image.open(oldImage)
    channels = img.getbands()
    if(len(channels) == 1):
        img = img.convert(mode='L')
    elif(len(channels) == 3):
        img = img.convert(mode='RGB')
    elif(len(channels) == 4):
        img = img.convert(mode='RGBA')
    else: 
        print("Image format not recognized!")
        return None
    
    
    if(destination != None):
        if not os.path.exists(destination):
            os.mkdir(destination)
        newFileName = os.path.join(destination, newFileName)
    img.save(newFileName)
    print("New file created: " + newFileName)



    if(args.merge is not None):
        if(len(args.merge) == 2):
            print("nothere")
            file0 = rasterio.open(args.merge[0])
            file1 = rasterio.open(args.merge[1])    
            result, transform = merge([file0,file1])
            #show(file0)
            #show(file1)
            # found that the 257th pixel is the same as the first pixel of the next image, so the files merge on the 257th pixel
            #print(file0.read(1)[:,256]==file1.read(1)[:,0])
            plt.show(result)
            print(result.data.shape)
        else:
            print("Must specify only 2 files!")

            


def tile(image, tileSize):
    img = Image.open(image)
    w, h = img.size
    if(h%tileSize != 0 or w%tileSize != 0):
        print("Tile size must be devisable by the size of the image.")
        return
    for i, j in product(range(0, h, tileSize), range(0, w, tileSize)):
        
        #!!!!!!!!!!!outputDir = makeOutputFolder('separate')!!!
        fn = newFilename(image,suffix=f"{i}_{j}.png",outdir="tiles")
        box = (j, i, j+tileSize, i+tileSize)
        img.crop(box).save(fn)

    #OTHER CROPSSSS
    # for i, filename in enumerate(os.listdir(t)):
    #     if (filename.endswith(".png")):
    #         im = Image.open(t+os.sep+filename)
    #         left = 0
    #         top = 0
    #         right = 256
    #         bottom = 256


    #         im1 = im.crop((left, top, right, bottom))
    #         newsize = (256, 256)
    #         im1 = im1.save(newDir+os.sep+filename.split(".")[0]+"_1.png")


    #         left = 256
    #         top = 0
    #         right = 512
    #         bottom = 256

    #         im2 = im.crop((left, top, right, bottom))
    #         newsize = (256, 256)
    #         im2 = im2.save(newDir+os.sep+filename.split(".")[0]+"_2.png")



    #         left = 0
    #         top = 256
    #         right = 256
    #         bottom = 512

    #         im3 = im.crop((left, top, right, bottom))
    #         newsize = (256, 256)
    #         im3 = im3.save(newDir+os.sep+filename.split(".")[0]+"_3.png")


    #         left = 256
    #         top = 256
    #         right = 512
    #         bottom = 512

    #         im4 = im.crop((left, top, right, bottom))
    #         newsize = (256, 256)
    #         im4 = im4.save(newDir+os.sep+filename.split(".")[0]+"_4.png")



if __name__ == '__main__':
    parser = getParser()
    args = parser.parse_args()
    print(args)
    if args.command == 'convert':
        processDirOrFile(convertPNG, target=args.image_filename)
    elif args.command == 'to8bit':
        outDir = makeOutputFolder('8bit')
        processDirOrFile(to8bit, target=args.image_filename, min=args.min, max=args.max, outDir=outDir )
    elif args.command == 'scale':
        processDirOrFile(scale, target=args.image_filename, type=args.type, min=args.min, max=args.max)
    elif args.command == 'crop':
        processDirOrFile(removeBlackLines, target=args.image_filename, size=args.size)
    elif args.command == 'tile':
        outDir = makeOutputFolder('tile')
        processDirOrFile(tile, target=args.image_filename, size=args.size)

        
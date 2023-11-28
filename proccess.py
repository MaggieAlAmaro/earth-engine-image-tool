import time
import rasterio
import matplotlib.pyplot as plt
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


    scaleParser = subparsers.add_parser("scale", help="Scale image values according to function.")
    scaleParser.add_argument('image_filename', type=str, help='Name of image')    
    scaleParser.add_argument('type', type=str, help='Type of scaling', choices=["sqrt","cbrt","log","iExp"])
    scaleParser.add_argument('-min', type=int, help='Type of scaling', default=-415)
    scaleParser.add_argument('-max', type=int, help='Type of scaling',default=8729)

    

    scaleParser = subparsers.add_parser("rescale", help="Scale image values according to function.")
    scaleParser.add_argument('image_filename', type=str, help='Name of image')    
    scaleParser.add_argument('type', type=str, help='Type of scaling', choices=["sqrt","cbrt","log","iExp"])
    scaleParser.add_argument('-min', type=int, help='Type of scaling', default=-415)
    scaleParser.add_argument('-max', type=int, help='Type of scaling',default=8729)
    
    cropParser = subparsers.add_parser("crop", help="Crop image to size")
    cropParser.add_argument('image_filename', type=str, help='Name of image')    
    cropParser.add_argument('size', type=int, nargs='+', help='Tuple with height and width of crop')
    
    removeLinesParser = subparsers.add_parser("removeLines", help="Remove black lines from image")
    removeLinesParser.add_argument('image_filename', type=str, help='Name of image')    
    removeLinesParser.add_argument('size', type=int,nargs='+', help='Tuple with height and width of crop')

    
    tileParser = subparsers.add_parser("tile", help="Tile the image into smaller images with the given size.")
    tileParser.add_argument('image_filename', type=str, help='Name of image')    
    tileParser.add_argument('size', type=int, help='Size of tiles')

    
    depthParser = subparsers.add_parser("to8bit", help="Convert image to 8bit")
    depthParser.add_argument('image_filename', type=str, help='Name of image')   
    depthParser.add_argument('-min', type=int, help='Type of scaling', default=-415)
    depthParser.add_argument('-max', type=int, help='Type of scaling',default=8729)

    
    stdParser = subparsers.add_parser("stdDev", help="Convert image to 8bit")
    stdParser.add_argument('image_filename', type=str, help='Name of image')   
    stdParser.add_argument('-limit', type=float, help='Type of scaling', default=4.5)
    return parser


def to16bit(image,  **kwargs):    
    outDir = kwargs.get('outDir','.')
    img = Image.open(image)
    data = np.array(img)
    img = img.convert(mode='I;16')
    fn = newFilename(image, suffix=".png", outdir=outDir)
    img.save(fn)

#if 16bit (unsigned int), [0, 65455]
#if 16bit (signed int), [-32768, 32767]
def to8bit(image,  **kwargs):
    outDir = kwargs.get('outDir','.')
    min = kwargs.get('min')
    max = kwargs.get('max')
    img = Image.open(image)
    data = np.array(img)
    data = data - min                               # make positive
    max = max - min
    min = 0
    # img = rasterio.open(image)
    # data = img.read()
    #print(img.bounds)
    normalizedData = (data- min)/(max-min)
    rescaledData = normalizedData * (255 - 0)  # 8bit -> [0,255]
    rescaledData = rescaledData.astype('uint8')
    newImg = Image.fromarray(rescaledData) # makes the mode L
    # fn = newFilename(image, suffix="_8bit.png", outdir=outDir)
    fn = newFilename(image, suffix=".png", outdir=outDir)
    newImg.save(fn)
    


def myInverseExponential(arr, mean=0.11549015741807088):
    return -np.exp(-(1/mean)*arr) + 1


def myReverseInverseExponential(arr, mean=0.11549015741807088):
    return -mean * np.log(-arr + 1)

# given that the SRTM image has a range estimate (according to earth engine) of:
# [-10, 6500]
# TODO: Use ee.reducer.max() and .min() to find the actual value limits of the SRTM image.
# SOLUTIONS: because it cant calculate the max over the whole image, do max/min convolution over it until getting true values
def scale(image: str,  **kwargs):
    img = Image.open(image)
    data = np.array(img)
    outDir = kwargs.get('outDir','.')
    min = kwargs.get('min', -415)   #-10
    max = kwargs.get('max', 8729)   #6500
    data = data + abs(min)                               # make positive
    max = max + abs(min)  
    min = 0
    typeT = kwargs.get('type')

    normalizedData = (data - min)/(max-min)          # Normalize [0,1]
    scalingFunction = None
    if (typeT == 'cbrt'):
        scalingFunction = np.cbrt
    elif (typeT == 'sqrt'):
        scalingFunction = np.sqrt
    elif (typeT == 'log'):
        scalingFunction = np.log # DOESNT WORK - UNFIDFINED FOR 0
    elif (typeT == 'iExp'):
        scalingFunction = np.vectorize(myInverseExponential)
    
    scaledNormData = scalingFunction(normalizedData)

    # Scale to range defined by scaling function
    # scaledNormData = (scalingFunction(data) - scalingFunction(min))/(scalingFunction(max)-scalingFunction(min))          # Normalize [0,1]

    # newMin = np.cbrt(min)
    # newMax = np.cbrt(max)
    # normalizedData = (np.cbrt(img) - newMin)/(newMax-newMin) #normalize funct: newX = x - min/ max - min

    bit8ScaledData = scaledNormData * (255 - 0)
    bit8ScaledData = bit8ScaledData.astype('uint8')
    
    # fn = newFilename(image, suffix=("_scaled_"+str(typeT)+".png"), outdir=outDir)
    fn = newFilename(image, suffix=(".png"), outdir=outDir)
    newImg = Image.fromarray(bit8ScaledData)
    newImg.save(fn)

def rescale(image: str,  **kwargs):
    img = Image.open(image)
    data = np.array(img)
    outDir = kwargs.get('outDir','.')
    minOriginal = kwargs.get('min')   #-415
    maxOriginal = kwargs.get('max')   #8729
    typeT = kwargs.get('type')
    
    if(img.mode == 'L' or img.mode == 'RGB'):
        min = 0
        max = 255
    else: # probably u16bit but best not guess
        return

    normalizedData = (data - min)/(max-min)          # Normalize [0,1]
    scalingFunction = None
    if (typeT == 'cbrt'):
        scalingFunction = np
    elif (typeT == 'sqrt'):
        scalingFunction = np.square
    elif (typeT == 'iExp'):
        scalingFunction = np.vectorize(myReverseInverseExponential)
    
    rescaledData = scalingFunction(normalizedData)#* (maxOriginal - minOriginal) + minOriginal

    #rescaledDatanormalized = (data - min)/(max-min)
    bit8ScaledData = rescaledData * (255 - 0)
    bit8ScaledData = bit8ScaledData.astype('uint8')
    
    # fn = newFilename(image, suffix=("_scaled_"+str(typeT)+".png"), outdir=outDir)
    fn = newFilename(image, suffix=(".png"), outdir=outDir)
    newImg = Image.fromarray(bit8ScaledData)
    newImg.save(fn)





# Crops excess from right and top
def crop(image, **kwargs):
    img = Image.open(image)
    size = kwargs.get('size')
    outDir = kwargs.get('outDir','.')

    
    if len(size) == 2:
        expr = img.size[0] > size[0] or img.size[1] > size[1]
    elif len(size) == 1:
        expr = img.size[0] > size[0] or img.size[1] > size[0]

    if expr:
        y0Clip = img.size[0] - size[0]  # clip excess top rows
        xClip = size[1] if len(size) == 2 else size[0]
        croppedImg = img.crop((0, y0Clip, xClip, img.size[0]))
        newFileName = newFilename(image, suffix=".png",outdir=outDir)
        croppedImg.save(newFileName)

    
    if len(size) == 2:
        expr = img.size[0] < size[0] or img.size[1] < size[1]
    elif len(size) == 1:
        expr = img.size[0] < size[0] or img.size[1] <size[0]
    elif expr:
        print("IMAGE SIZE IS LOWER THAN CLIP SIZE.")


#After turning from 16bit to 8bit black lines have the value of 6
def isLineBlack(vector, tolerance = 0, value = 0):
    return (vector == value).all()    
    # return np.sum(vector) <= (0 + tolerance)

def checkSequenceForBlackLines(imgData, axis, idx, size, value = 0):  
    blackLineCount = 0  
    for i in range(idx,idx+size):
        if(axis == 0):
            if isLineBlack(imgData[i,:],value=value): 
                blackLineCount += 1
        if(axis == 1):
            if isLineBlack(imgData[:,i],value=value):
                blackLineCount += 1
    return blackLineCount



count = 0
def countBlackLines(image, **kwargs):
    size = kwargs.get('size')
    outDir = kwargs.get('outDir','.')
    img = Image.open(image)
    data = np.array(img)
    # print("Original size:", data[0,:])
    if(data.shape == size):
        print("Shape is already the expected shape.")
        return

    
    rowExtra = data.shape[0] - size[0]
    colExtra = data.shape[1] - size[1]

    blackLineValue = 6 #39  # 0 if clamp is at -10,6500; 6 if clamp is at -160,6500


    row0Clip = checkSequenceForBlackLines(data,0,0,rowExtra,blackLineValue)
    col0Clip = checkSequenceForBlackLines(data,1,0,colExtra,blackLineValue)
    row1Clip = checkSequenceForBlackLines(data,0,data.shape[0]-rowExtra,rowExtra,blackLineValue)
    col1Clip = checkSequenceForBlackLines(data,1,data.shape[1]-colExtra,colExtra,blackLineValue)

    # print("Line Crop Measures:",(col0Clip, row0Clip, data.shape[1] - col1Clip, data.shape[0] - row1Clip))

    if((data.shape[0] - row1Clip) - row0Clip < size[0]):
        row0Clip = 0
    if((data.shape[1] - col1Clip) - col0Clip < size[1]):
        col0Clip = 0

    global count
    a = col0Clip + row1Clip +row0Clip +col1Clip
    pixelOverlap = (col0Clip + col1Clip) * (row0Clip +row1Clip)
    count += (a*size[0] - pixelOverlap)
    

def removeBlackLines(image, **kwargs):
    size = kwargs.get('size')
    outDir = kwargs.get('outDir','.')
    img = Image.open(image)
    # print("Original size:",img.mode)
    data = np.array(img)
    # print("Original size:", data[0,:])
    if(data.shape == size):
        print("Shape is already the expected shape.")
        return

    
    rowExtra = data.shape[0] - size[0]
    colExtra = data.shape[1] - size[1]

    blackLineValue = 0 #6 #39  # 0 if clamp is at -10,6500; 6 if clamp is at -160,6500


    row0Clip = checkSequenceForBlackLines(data,0,0,rowExtra,blackLineValue)
    col0Clip = checkSequenceForBlackLines(data,1,0,colExtra,blackLineValue)
    row1Clip = checkSequenceForBlackLines(data,0,data.shape[0]-rowExtra,rowExtra,blackLineValue)
    col1Clip = checkSequenceForBlackLines(data,1,data.shape[1]-colExtra,colExtra,blackLineValue)

    # print("Line Crop Measures:",(col0Clip, row0Clip, data.shape[1] - col1Clip, data.shape[0] - row1Clip))
    print((col0Clip, row0Clip, data.shape[1] - col1Clip, data.shape[0] - row1Clip))

    if((data.shape[0] - row1Clip) - row0Clip < size[0]):
        row0Clip = 0
    if((data.shape[1] - col1Clip) - col0Clip < size[1]):
        col0Clip = 0

        
    print((col0Clip, row0Clip, data.shape[1] - col1Clip, data.shape[0] - row1Clip))
    
    croppedImg = img.crop((col0Clip, row0Clip, data.shape[1] - col1Clip, data.shape[0] - row1Clip))
    if(croppedImg.size[0] < size[0] or croppedImg.size[1] < size[1]):
        print("REMOVED TOO MANY LINES")
    # print("New size:",croppedImg.size)
    newFileName = newFilename(image, suffix=".png",outdir=outDir)
    croppedImg.save(newFileName)



def convert(image, **kwargs):
    outDir = kwargs.get('outDir','.')
    newFileName = newFilename(image, suffix=".png",outdir=outDir)
    img = Image.open(image)
    channels = img.getbands()
    if(len(channels) == 1):
        img = img.convert(mode='L')
    elif(len(channels) == 3):
        img = img.convert(mode='RGB')
    elif(len(channels) == 4):
        img = img.convert(mode='RGBA')
    else: 
        print("Image format not recognized!")
        return 
    img.save(newFileName)



def merge():
        file0 = rasterio.open(args.merge[0])
        file1 = rasterio.open(args.merge[1])    
        result, transform = merge([file0,file1])
        # found that the 257th pixel is the same as the first pixel 
        # of the next image, so the files merge on the 257th pixel
        #print(file0.read(1)[:,256]==file1.read(1)[:,0])
        plt.show(result)


def stdDev(image,**kwargs):
    limit = kwargs.get('limit')
    aDir = kwargs.get('above')
    bDir = kwargs.get('below')

    img = Image.open(image)
    data = np.array(img)
    stdD = np.std(data)
    
    out = bDir if stdD < limit else aDir
    newFileName = newFilename(image, suffix=".png",outdir=out)
    img.save(newFileName)



def tile(image, **kwargs):
    out = kwargs.get('outDir')
    size = kwargs.get('size')
    img = Image.open(image)
    w, h = img.size
    if h%size != 0 or w%size != 0:
        print("Tile size must be devisable by the size of the image.")
        return
    for i, j in product(range(0, h, size), range(0, w, size)):
        fn = newFilename(image, suffix=f"_{i}_{j}.png", outdir=out)
        box = (j, i, j+size, i+size)
        img.crop(box).save(fn)


if __name__ == '__main__':
    parser = getParser()
    args = parser.parse_args()
    print(args)
    if args.command == 'convert':
        outDir = makeOutputFolder('convert')
        processDirOrFile(convert, target=args.image_filename,outDir=outDir)
    elif args.command == 'to8bit':
        outDir = makeOutputFolder('8bit')
        processDirOrFile(to8bit, target=args.image_filename, min=args.min, max=args.max, outDir=outDir )
    elif args.command == 'scale':
        outDir = makeOutputFolder(str(args.type)+'_scale')
        processDirOrFile(scale, target=args.image_filename, type=args.type, min=args.min, max=args.max, outDir=outDir)
    elif args.command == 'rescale':
        outDir = makeOutputFolder(str(args.type)+'_rescale')
        processDirOrFile(rescale, target=args.image_filename, type=args.type, min=args.min, max=args.max, outDir=outDir)
    elif args.command == 'crop':
        outDir = makeOutputFolder('crop')
        processDirOrFile(crop, target=args.image_filename, size=args.size,outDir=outDir)
    elif args.command == 'removeLines':
        outDir = makeOutputFolder('remove_lines')
        processDirOrFile(removeBlackLines, target=args.image_filename, size=args.size, outDir=outDir)
        # processDirOrFile(countBlackLines, target=args.image_filename, size=args.size, outDir=outDir)
        # print(count)
    elif args.command == 'tile':
        outDir = makeOutputFolder('tile')
        processDirOrFile(tile, target=args.image_filename, size=args.size, outDir=outDir)
    elif args.command == 'stdDev':
        outDir = makeOutputFolder('stdDev')
        aDir = os.path.join(outDir,'above')
        bDir = os.path.join(outDir,'below')
        os.mkdir(bDir)
        os.mkdir(aDir)
        processDirOrFile(stdDev, target=args.image_filename, limit=args.limit, above=aDir,below=bDir)

        
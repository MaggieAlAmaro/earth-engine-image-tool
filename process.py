#import rasterio
import matplotlib.pyplot as plt
import random
import numpy as np
import argparse
from PIL import Image
from itertools import product
import os, sys, math
from abc import ABC, abstractmethod
import albumentations
from albumentations import RandomRotate90

from Image import MyImage, RGBAImage, GrayscaleImage

from utils import processDirOrFile, newFilename, makeOutputFolder



def getParser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('image_filename', type=str, help='Name of image')    
    subparsers = parser.add_subparsers(dest="command",required=True)
    mergeParser = subparsers.add_parser("convert", help="Convert to png")   
    mergeParser.add_argument('image_filename', type=str, help='Name of image')    
    # mergeParser.add_argument('-o','--out_filename', type=str, help='Output Directory')


    scaleParser = subparsers.add_parser("scale", help="Scale image values according to function.")
    scaleParser.add_argument('type', type=str, help='Type of scaling', choices=["sqrt","cbrt","log","iExp"])
    scaleParser.add_argument('-min', type=int, help='Type of scaling', default=-415)
    scaleParser.add_argument('-max', type=int, help='Type of scaling',default=8729)

    

    scaleParser = subparsers.add_parser("rescale", help="Scale image values according to function.")
    scaleParser.add_argument('type', type=str, help='Type of scaling', choices=["sqrt","cbrt","log","iExp"])
    scaleParser.add_argument('-min', type=int, help='Type of scaling', default=-415)
    scaleParser.add_argument('-max', type=int, help='Type of scaling',default=8729)
    
    cropParser = subparsers.add_parser("crop", help="Crop image to size")
    cropParser.add_argument('size', type=int, nargs='+', help='Tuple with height and width of crop')
    
    removeLinesParser = subparsers.add_parser("removeLines", help="Remove black lines from image")
    removeLinesParser.add_argument('size', type=int,nargs='+', help='Tuple with height and width of crop')

    
    tileParser = subparsers.add_parser("tile", help="Tile the image into smaller images with the given size.")
    tileParser.add_argument('size', type=int, help='Size of tiles')

    
    augParser = subparsers.add_parser("augment", help="Augment Data Randomly")

    
    depthParser = subparsers.add_parser("8bit", help="Convert image to 8bit")
    depthParser.add_argument('-min', type=int, help='Type of scaling', default=-415)
    depthParser.add_argument('-max', type=int, help='Type of scaling',default=8729)

    
    stdParser = subparsers.add_parser("stdDev", help="Convert image to 8bit")
    stdParser.add_argument('-limit', type=float, help='Type of scaling', default=4.5)
    return parser


class Processor(ABC):
    @abstractmethod
    def process(self, image: MyImage or str):
        pass
    


class Crop(Processor):
    def __init__(self, config_args):
        self.outSize = config_args['size']
        self.outputDir = makeOutputFolder('crop')

    def crop(self, size):  
        if size[0] < self.outSize[0] or size[1] < self.outSize[1]:
            raise Exception("IMAGE SIZE IS LOWER THAN CLIP SIZE.")
        return (0, 0, self.outSize[1], self.outSize[0])

    def process(self, image: MyImage):
            try:
                cropBounds = self.crop(image.size)
            except:
                return
            croppedImg = image.image.crop(cropBounds)
            newFileName = newFilename(image.image_filename, suffix=".png", outdir=self.outputDir)
            croppedImg.save(newFileName)


class Scale(Processor):
    def __init__(self, config_args):
        self.typeT = config_args['type']
        self.min = config_args['min']
        self.max = config_args['max']
        self.outputDir = makeOutputFolder(self.typeT+'_scale')

    def myInverseExponential(self, arr, mean=0.11549015741807088):
        return -np.exp(-(1/mean)*arr) + 1


    def scale(self, data):
        #Push data to > 0
        data = data + abs(self.min)                               
        self.max = self.max + abs(self.min)  
        self.min = 0

        #Normalize [0,1]
        normalizedData = (data - self.min)/(self.max - self.min)          
        scalingFunction = None
        if (self.typeT == 'cbrt'):
            scalingFunction = np.cbrt
        elif (self.typeT == 'sqrt'):
            scalingFunction = np.sqrt
        elif (self.typeT == 'iExp'):
            scalingFunction = np.vectorize(self.myInverseExponential)
        
        scaledNormData = scalingFunction(normalizedData)

        bit8ScaledData = scaledNormData * (255 - 0)
        bit8ScaledData = bit8ScaledData.astype('uint8')
        return bit8ScaledData

    def process(self, image: MyImage):
        result = self.scale(image.data)
        updatedImage = image.post_process_step(processedData=result)
        fn = newFilename(image.image_filename, suffix=".png", outdir=self.outputDir)
        updatedImage.save(fn)
        

class StdDevSort(Processor):
    def __init__(self, config_args):
        # self.aDir = config_args['limit']
        # self.bDir = kwargs.get('type')
        self.limit = config_args['limit']
        self.outputDir = makeOutputFolder('stdDev')
        self.aDir = os.path.join(self.outputDir,'above')
        self.bDir = os.path.join(self.outputDir,'below')
        os.mkdir(self.aDir)
        os.mkdir(self.bDir)

    def stdDev(self, data):
        stdD = np.std(data)
        return self.bDir if stdD < self.limit else self.aDir
        
    def process(self, image: MyImage):
        resultOutDir = self.stdDev(image.data)
        fn = newFilename(image.image_filename, suffix=".png", outdir=resultOutDir)
        image.image.save(fn)


class Tile(Processor):
    def __init__(self, config_args):
        self.tileSize = config_args['size']
        self.outputDir = makeOutputFolder('tile')


    def tile(self, size):
        if size[1] % self.tileSize[1]  != 0 or size[0] % self.tileSize[0]  != 0:
            raise Exception("Tile size must be devisable by the size of the image.")
        cropList = []
        for i, j in product(range(0, size[1], self.tileSize[1]), range(0, size[0], self.tileSize[0])):
            box = (j, i, j+self.tileSize[0], i+self.tileSize[1])
            cropList.append(box)
        return cropList
        
    def process(self, image: MyImage):
        # try:
        #     cropList = self.tile(image.size)
        # except Exception as e:
        #     print(e)
        #     return 
        cropList = self.tile(image.size)
        for crop in cropList:
            fn = newFilename(image.image_filename, suffix=f"_{crop[1]}_{crop[0]}.png", outdir=self.outputDir)
            image.image.crop(crop).save(fn) 


class BlackLineRemoval(Processor):
    def __init__(self, config_args):
        self.expectedSize = config_args['size']
        #After turning from 16bit to 8bit black lines have the value of 6
        #0 normally, both in 8bit and 16bit, if clamp is at -10,6500; 6 if clamp is at -160,6500;  #39
        self.blackLineValue = config_args['value']
        self.outputDir = makeOutputFolder('removeLines')
        

    def checkBorderForBlackLines(self, imgData, axis, startIdx, endIdx, value = 0):  
        blackLineCount = 0  
        for i in range(startIdx, endIdx):
            if(axis == 0):
                if (imgData[:,i] == value).all(): 
                    blackLineCount += 1
            if(axis == 1):
                if (imgData[i,:] == value).all():
                    blackLineCount += 1
        return blackLineCount

    def getCropForBlackLines(self, data, size):
        if(size == self.expectedSize):
            print("Image is already the expected shape.")
            return (0, 0, size[1], size[0])

        
        rowExtra = size[0] - self.expectedSize[0]
        colExtra = size[1] - self.expectedSize[1]        

        row0Clip = self.checkBorderForBlackLines(data, 0, 0, rowExtra, self.blackLineValue)
        col0Clip = self.checkBorderForBlackLines(data, 1, 0, colExtra, self.blackLineValue)
        row1Clip = self.checkBorderForBlackLines(data, 0, size[0] - rowExtra, size[0], self.blackLineValue)
        col1Clip = self.checkBorderForBlackLines(data, 1, size[1] - colExtra, size[1], self.blackLineValue)



        #TODO: Check surrounding pixels to see if they are also black in case of too many black lines found
        #If too many lines are to be removed then iterate over both sides crop size and remove 1 until the size is good
        removalCount = 0
        while row0Clip + row1Clip > rowExtra:
            row0Clip -= 1 if removalCount%2 else row1Clip 
            removalCount += 1

        removalCount = 0
        while col0Clip + col1Clip > colExtra:
            col0Clip -= 1 if removalCount%2 else col1Clip 
            removalCount += 1

            
        # global count
        # a = col0Clip + row1Clip +row0Clip +col1Clip
        # pixelOverlap = (col0Clip + col1Clip) * (row0Clip +row1Clip)
        # count += (a*size[0] - pixelOverlap)

        print("Line Crop Measures:", (col0Clip, row0Clip, size[1] - col1Clip, size[0] - row1Clip))
        return (col0Clip, row0Clip, size[1] - col1Clip, size[0] - row1Clip)

        
    def process(self, image: MyImage):
        resultCropSize = self.getCropForBlackLines(image.data, image.size)
        croppedImage = image.image.crop(resultCropSize)
        fn = newFilename(image.image_filename, suffix=".png", outdir=self.outputDir)
        croppedImage.save(fn)

class Rescale(Processor):
    def __init__(self, config_args) -> None:
        self.typeT = config_args['type']
        self.min = config_args['min']
        self.max = config_args['max']
        self.outputDir = makeOutputFolder(self.typeT+'_rescale')

    
    def myReverseInverseExponential(arr, mean=0.11549015741807088):
        return -mean * np.log(-arr + 1)
    
    

    def rescale(self, data):
        normalizedData = (data - self.min)/(self.max-self.min)          # Normalize [0,1]
        scalingFunction = None
        if (self.typeT == 'cbrt'):
            scalingFunction = np
        elif (self.typeT == 'sqrt'):
            scalingFunction = np.square
        elif (self.typeT == 'iExp'):
            scalingFunction = np.vectorize(self.myReverseInverseExponential)
        
        rescaledData = scalingFunction(normalizedData) #* (maxOriginal - minOriginal) + minOriginal

        #rescaledDatanormalized = (data - min)/(max-min)
        bit8ScaledData = rescaledData * (255 - 0)
        bit8ScaledData = bit8ScaledData.astype('uint8')
        
        return bit8ScaledData


    def process(self, image: MyImage):
        result = self.rescale(image.data)
        updatedImage = image.post_process_step(processedData=result)
        fn = newFilename(image.image_filename, suffix=".png", outdir=self.outputDir)
        updatedImage.save(fn)



class DoRandomRotate90(RandomRotate90):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
    
    def get_params(self):
        # Random int in the range [1, 3]
        return {"factor": random.randint(1, 3)}

class Augment(Processor):
    def __init__(self, config_args) -> None:
        self.typeT = config_args['type']
        self.min = config_args['min']
        self.max = config_args['max']
        self.outputDir = makeOutputFolder('augment')



    def getAugmentTransform():
        transform = albumentations.Compose([
            albumentations.OneOf([
                albumentations.HorizontalFlip(p=1),
                albumentations.VerticalFlip(p=1),
            ], p=1),
            DoRandomRotate90(p=0.5),
            ]
        )
        return transform

    def process(self, image: MyImage):
        transform = self.getAugmentTransform()
        fn = newFilename(image.image_filename, suffix=".png", outdir=self.outputDir)
        img = Image.open(image)
        img = transform(image=img)["image"]
        img.save(fn)



def changeMode(self, **kwargs):
    outDir = kwargs.get('outDir','.')
    newFileName = newFilename(self.image_filename, suffix=".png",outdir=outDir)
    channels = self.image.getbands()
    if(len(channels) == 1):
        self.image = self.image.convert(mode='L')
    elif(len(channels) == 3):
        self.image = self.image.convert(mode='RGB')
    elif(len(channels) == 4):
        self.image = self.image.convert(mode='RGBA')
    else: 
        print("Image format not recognized!")
        return 
    self.image.save(newFileName)


def to16bit(self, **kwargs):    
    outDir = kwargs.get('outDir','.')
    self.image = self.image.convert(mode='I;16')
    fn = newFilename(self.image_filename, suffix=".png", outdir=outDir)
    self.image.save(fn)

#if 16bit (unsigned int), [0, 65455]
#if 16bit (signed int), [-32768, 32767]
def to8bit(self,  **kwargs):
    outDir = kwargs.get('outDir','.')
    min = kwargs.get('min')
    max = kwargs.get('max')
    self.data = self.data - min                               # make positive
    max = max - min
    min = 0
    # img = rasterio.open(image)
    # data = img.read()
    #print(img.bounds)
    normalizedData = (self.data- min)/(max-min)
    rescaledData = normalizedData * (255 - 0)  # 8bit -> [0,255]
    rescaledData = rescaledData.astype('uint8')
    newImg = Image.fromarray(rescaledData) # makes the mode L
    # fn = newFilename(image, suffix="_8bit.png", outdir=outDir)
    

# given that the SRTM image has a range estimate (according to earth engine) of:
# [-10, 6500]
# TODO: Use ee.reducer.max() and .min() to find the actual value limits of the SRTM image.
# SOLUTIONS: because it cant calculate the max over the whole image, do max/min convolution over it until getting true values




    # def process(self, args, outDirectory=None):
    #     outDir = outDirectory if outDirectory != None else setOutputDirectoryName(args)
        
    #     if args.command == 'changeMode':
    #         self.changeMode(outDir=outDir)
    #     elif args.command == '8bit':
    #         self.to8bit(min=args.min, max=args.max, outDir=outDir)
    #     elif args.command == 'scale':
    #         self.scale(type=args.type, min=args.min, max=args.max, outDir=outDir)
    #     elif args.command == 'rescale':
    #         self.rescale(type=args.type, min=args.min, max=args.max, outDir=outDir)
    #     elif args.command == 'crop':
    #         self.crop(size=args.size,outDir=outDir)
    #     elif args.command == 'removeLines':
    #         self.removeBlackLines(size=args.size, outDir=outDir)
    #         # processDirOrFile(countBlackLines, target=args.image_filename, size=args.size, outDir=outDir)
    #         # print(count)
    #     elif args.command == 'tile':
    #         self.tile(size=args.size, outDir=outDir)
    #     elif args.command == 'augment':
    #         self.augment(outDir=outDir)
    #     elif args.command == 'stdDev':
    #         self.stdDev(limit=args.limit, outDir=outDir)

    


# if __name__ == '__main__':
#     parser = getParser()
#     args = parser.parse_args()
#     print(args)

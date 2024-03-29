#import rasterio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from Image import *
import os, json
from src.processors.process import Processor
# import albumentations
# from albumentations import RandomRotate90

from src.utils import newFilename, makeOutputFolder




class StdDevSort(Processor):
    def __init__(self, config_args):
        super().__init__()
        # self.aDir = config_args['limit']
        # self.bDir = kwargs.get('type')
        self.limit = config_args['limit']
        self.channel = config_args['channel']
        self.outputDir = makeOutputFolder('stdDev')
        self.aDir = os.path.join(self.outputDir,'above')
        self.bDir = os.path.join(self.outputDir,'below')
        os.mkdir(self.aDir)
        os.mkdir(self.bDir)

    def stdDev(self, data):
        stdD = np.std(data)
        print(stdD)
        return self.bDir if stdD < self.limit else self.aDir
        
    def process(self, image: MyImage):
        data = image.data[self.channel]
        resultOutDir = self.stdDev(data)
        fn = newFilename(image.image_filename, suffix=".png", outdir=resultOutDir)
        image.image.save(fn)
    
    def getOutputFolder(self) -> str:
        return os.path.join(self.outputDir,'above') 




class BlackLineRemoval(Processor):
    def __init__(self, config_args):
        super().__init__()
        self.expectedSize = config_args['size']
        #After turning from 16bit to 8bit black lines have the value of 6
        #0 normally, both in 8bit and 16bit, if clamp is at -10,6500; 6 if clamp is at -160,6500;  #39
        self.blackLineValue = config_args['value']
        self.countPixels = config_args['countPixels']

        if 'dictionary' in config_args:
            with open(config_args['dictionary'], 'r') as f:
                self.dictionary = json.load(f)
            with open(config_args['dictionaryRGBtoAMatch'], 'r') as f:
                self.dictionaryMatch = json.load(f)
        else:
           self.dictionary = None

        if self.countPixels:
            self.count = 0
        self.logRowCols = config_args['logRowCols']
        if self.logRowCols:
            self.log = {}
            
        self.channel = config_args['channel']
        self.outputDir = makeOutputFolder('removeLines')
        

    def checkBorderForBlackLines(self, imgData, axis, startIdx, endIdx, value = 0):  
        blackLineCount = 0  
        for i in range(startIdx, endIdx):
            if(axis == 0):
                if (imgData[i,:] == value).all(): 
                    blackLineCount += 1
            if(axis == 1):
                if (imgData[:,i] == value).all():
                    blackLineCount += 1
        return blackLineCount

    def getCropForBlackLines(self, data, size, filename):
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
        #Does: If too many lines are to be removed then iterate over both sides crop size and remove 1 until the size is good
        removalCount = 0
        while row0Clip + row1Clip > rowExtra:
            row0Clip -= 1 if removalCount%2 else row1Clip 
            removalCount += 1

        removalCount = 0
        while col0Clip + col1Clip > colExtra:
            col0Clip -= 1 if removalCount%2 else col1Clip 
            removalCount += 1

            
        #TODO: check if the size[x] is right
        if self.countPixels:
            totalRow = (row1Clip + row0Clip) * size[0]
            totalCol = (col0Clip + col1Clip) * size[1]
            extraFromPixelOverlap = (col0Clip + col1Clip) * (row0Clip + row1Clip)
            self.count += (totalRow + totalCol - extraFromPixelOverlap)

        print("Line Crop Measures:", (col0Clip, row0Clip, size[1] - col1Clip, size[0] - row1Clip))
        if self.logRowCols:
            self.log[filename.split(".")[0]] = [col0Clip, row0Clip, size[1] - col1Clip, size[0] - row1Clip]

        return (col0Clip, row0Clip, size[1] - col1Clip, size[0] - row1Clip)

        
    def process(self, image: MyImage):
        if (len(image.shape) == 2):
            img = image.data
        else:
            assert(image.shape[0] > self.channel)
            img = image.data[self.channel]


        if self.dictionary != None:
            if os.path.basename(image.image_filename).split(".")[0] in self.dictionaryMatch: 
                if  self.dictionaryMatch[os.path.basename(image.image_filename).split(".")[0]] in self.dictionary:
                    resultCropSize = self.dictionary[ self.dictionaryMatch[os.path.basename(image.image_filename).split(".")[0]] ]
                else:
                    return
            else:
                return
        else:
            resultCropSize = self.getCropForBlackLines(img, image.shape,os.path.basename(image.image_filename))

        croppedImage = image.image.crop(resultCropSize)
        fn = newFilename(image.image_filename, suffix=".png", outdir=self.outputDir)
        croppedImage.save(fn)


    def onBatchFinish(self):
        if self.logRowCols:
            outDir = makeOutputFolder('blackLineCropDictionary')
            with open(os.path.join(outDir,"blackLineLog.json"), 'w') as f:
                f.write(json.dumps(self.log, indent=4))
        if self.countPixels:
            print("The Number of total Black Pixels was ", self.count)


class Rescale(Processor):
    def __init__(self, config_args) -> None:
        super().__init__()
        self.channel = config_args['channel']
        self.typeT = config_args['type']
        if(self.typeT == 'iExp'):
            self.mean = config_args['mean']
        self.typeT = config_args['type']
        self.min = config_args['min']
        self.max = config_args['max']
        self.outputDir = makeOutputFolder(self.typeT+'_rescale')

    
    def myReverseInverseExponential(self, arr):
        return -self.mean * np.log(-arr + 1)
    
    def ExponentialCDF(self, arr):
        return 1 - np.exp(-(1/self.mean) * arr)

    def rescale(self, data):
        normalizedData = (data - self.min)/(self.max-self.min)          # Normalize [0,1]
        scalingFunction = None
        if (self.typeT == 'cbrt'):
            scalingFunction = np
        elif (self.typeT == 'sqrt'):
            scalingFunction = np.square
        elif (self.typeT == 'iExp'):
            scalingFunction = np.vectorize(self.ExponentialCDF)
        
        rescaledData = scalingFunction(normalizedData) #* (maxOriginal - minOriginal) + minOriginal

        #rescaledDatanormalized = (data - min)/(max-min)
        bit8ScaledData = rescaledData * (255 - 0)
        bit8ScaledData = bit8ScaledData.astype('uint8')
        
        return bit8ScaledData


    def process(self, image: MyImage):
        assert(image.shape[0]>self.channel)
        result = self.rescale(image.data[self.channel])
        updatedImage = image.post_process_step(processedData=result)
        fn = newFilename(image.image_filename, suffix=".png", outdir=self.outputDir)
        updatedImage.save(fn)





class Scale(Processor):
    def __init__(self, config_args):
        super().__init__()
        self.channel = config_args['channel']
        self.typeT = config_args['type']
        if(self.typeT == 'iExp'):
            self.mean = config_args['mean']
        self.min = config_args['min']
        self.max = config_args['max']
        self.outputDir = makeOutputFolder(self.typeT+'_scale')

    def ExponentialCDF(self, arr):
        return 1 - np.exp(-(1/self.mean) * arr)

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
            scalingFunction = np.vectorize(self.ExponentialCDF)
        
        scaledNormData = scalingFunction(normalizedData)

        bit8ScaledData = scaledNormData * (255 - 0)
        bit8ScaledData = bit8ScaledData.astype('uint8')
        return bit8ScaledData

    def process(self, image: MyImage):
        if (len(image.shape) == 2):
            img = image.data
        else:
            assert(image.shape[0] > self.channel)
            img = image.data[self.channel]
        result = self.scale(img)
        imggg = Image.fromarray(result)

        # updatedImage = image.post_process_step(processedData=result)
        fn = newFilename(image.image_filename, suffix=".png", outdir=self.outputDir)
        imggg.save(fn)
        


class EdgeDetection(Processor):
    def __init__(self, config_args):
        super().__init__()
        self.channel = config_args['channel']
        self.outputDir = makeOutputFolder('edge_detection')


    def process(self, image: MyImage):
        newImg = image.data[self.channel]
        pilImg = Image.fromarray(newImg)
        img = pilImg.convert('L')
                
        # Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
        final = img.filter(ImageFilter.FIND_EDGES)
        # final = img.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8,
        #                                   -1, -1, -1, -1), 1, 0))
        
        # Saving the Image Under the name Edge_Sample.png
        fn = newFilename(image.image_filename, suffix=".png", outdir=self.outputDir)
        
        final.save(fn)
        

import cv2
class KMeansSegmentation(Processor):
    def __init__(self, config_args):
        super().__init__()
        self.channel = config_args['channel']
        self.k = config_args['k']
        self.outputDir = makeOutputFolder('k-means')


    def process(self, image: MyImage):
        newImg = image.data[self.channel]
        print(newImg.shape)
        vectorized_img = newImg.reshape((-1,1))
        print(vectorized_img.shape)
        vectorized = np.float32(vectorized_img)

        # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
        _, label, center = cv2.kmeans(vectorized, self.k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((newImg.shape))

        out = Image.fromarray(res2)
        fn = newFilename(image.image_filename, suffix=".png", outdir=self.outputDir)
        out.save(fn)
        

class MedianDenoise(Processor):
    def __init__(self, config_args):
        super().__init__()
        self.channel = config_args['channel']
        self.kernelSize = config_args['kernelSize']
        self.outputDir = makeOutputFolder('denoise_median')


    def denoise(data, kernelSize) -> np.array:
        return cv2.medianBlur(data, kernelSize)
        # median2 = cv2.medianBlur(data, 5)
        # blur2 = cv2.bilateralFilter(data, 5, 3,3 )
        # compare = np.concatenate((data, median, median2, blur2), axis=1) #side by side comparison
        # plt.imshow(compare)
        # plt.show()



    def process(self, image: MyImage):
        newImg = image.data[self.channel]
        out = MedianDenoise.denoise(newImg,self.kernelSize)
        out = Image.fromarray(out)
        fn = newFilename(image.image_filename, suffix=".png", outdir=self.outputDir)
        out.save(fn)
        

        
class GaussianDenoise(Processor):
    def __init__(self, config_args):
        super().__init__()
        self.channel = config_args['channel']
        self.kernelSize = config_args['kernelSize']
        self.outputDir = makeOutputFolder('denoise_median')


    def denoise(data, kernelSize) -> np.array:
        raise Exception("Unimplemented")


    def process(self, image: MyImage):
        newImg = image.data[self.channel]
        out = GaussianDenoise.denoise(newImg,self.kernelSize)
        out = Image.fromarray(out)
        fn = newFilename(image.image_filename, suffix=".png", outdir=self.outputDir)
        out.save(fn)
        
        
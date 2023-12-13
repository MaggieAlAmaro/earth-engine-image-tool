import os
import random
import math
import time
import pickle,json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from Image import MyImage
from process import Processor
from utils import makeOutputFolder, newFilename

# def getParser(**parser_kwargs):
#     parser = argparse.ArgumentParser(**parser_kwargs, formatter_class=argparse.RawTextHelpFormatter)
#     parser.add_argument('dir', type=str, help='Name of data directory') 
#     parser.add_argument('-r','--ratio', type=float, help='Ratio of training to validation images', default=1/6) 
#     return parser

class DatasetSplit(Processor):
    def __init__(self, config_args):
        self.ratio = config_args['valPercent']
        self.shuffle = config_args['shuffle']
        self.outputDir = makeOutputFolder('datasetSplit')


    def checkIfDirValid(self, dataDir):
        for f in os.listdir(dataDir):
            if not f.endswith('.png'):
                print('Found file with incorrect format (not PNG):' + f)
                return False
        return True
    
    def filenameTxt(self, dataDir):
        # fn = os.path.basename(dataDir) + '.txt'
        fn = dataDir + '.txt'
        f = open(fn, "a")
        for filename in os.listdir(dataDir):
            f.write(filename+"\n")
        f.close()
        print("Created new file: " + fn)
        return fn
    
    def shuffleFile(self, filename):
        with open(filename,'r') as f:
            lines = f.readlines()
            random.shuffle(lines)
        with open(filename,'w') as f:
            f.writelines(lines)





    def createFilenameTxt(self, dataDir):
        # dataSize = len([name for name in os.listdir(h) if os.path.isfile(h+os.sep+name)])
        if not self.checkIfDirValid(dataDir):
            return
        fn = self.filenameTxt(dataDir)
        if self.shuffle:
            self.shuffleFile(fn)
        return fn



    def splitIntoTrainValidationTxt(self, filenameTxt, ratio):
        f = open(filenameTxt, "r")
        lines = f.readlines()
        f.close()
        print("Total files:" + str(len(lines)))

        validationSize = int(math.floor(len(lines) * ratio))
        trainSize = len(lines) - validationSize
        print("Train Size: " +str(trainSize))
        print("Validation Size: " + str(validationSize))
        train = lines[:trainSize]
        val = lines[trainSize:]
                
        name = os.path.splitext(filenameTxt)[0]
        ftrain = name + '_train.txt'
        fval = name + '_validation.txt'

        #train
        ft = open(ftrain, "w")
        ft.writelines(train)
        ft.close()

        #validation
        fv = open(fval, "w")
        fv.writelines(val)
        fv.close()


    def process(self, image: MyImage or str):
        fn = self.createFilenameTxt(image)
        if fn != None:
            self.splitIntoTrainValidationTxt(fn, self.ratio)


class HistogramAnalysis(Processor):
    def __init__(self, config_args):
        self.image_type = config_args['image_type']
        self.size = config_args['size']

        self.bit_depth = config_args['bit_depth']
        if self.bit_depth == 's16bit':
            self.pixel_range = 65536
            self.min = -32768
        elif self.bit_depth == 'u16bit':
            self.pixel_range = 65536
            self.min = 0
        elif self.bit_depth == 'u8bit':
            self.pixel_range = 256
            self.min = 0
        elif self.bit_depth == 'custom':
            self.pixel_range = config_args['pixel_range']   #6500
            self.min = config_args['min']                   #-10

        self.dist = np.zeros(shape=self.pixel_range) 
        self.outputDir = makeOutputFolder('histogram')


    def plotAndSaveImagePixelDist(self):
        firstNonZeroIndex = np.nonzero(self.dist)[0][0] #(np.where(x > 1)[0][0]) 
        LastNonZeroIndex = np.nonzero(self.dist)[0][-1]
        #get smaller distribution (The more indexes on the x axis the longer it takes for it to load)
        arr = self.dist[firstNonZeroIndex:LastNonZeroIndex]

        #set indexes to be from firstNonZeroIndex to LastNonZeroIndex.
        index = np.arange((firstNonZeroIndex + self.min), (LastNonZeroIndex + self.min), 1) #np.argmax(x),1)#-32768,32767,1)#np.argmin(x),np.argmax(x),1)
        
        #set indexes to be positive 
        index = index+abs( (firstNonZeroIndex + self.min))
        #set indexes to [0,1]
        index = index/len(index) 

        n = np.sum(arr)
        mean = np.dot(arr,index)/n
        print("Mean:",mean)
        std = np.sqrt(np.sum(arr*np.square(index-mean))/n)
        print("Std:",std)

        # normalizedData = (x - firstNonZeroIndex)/(LastNonZeroIndex-firstNonZeroIndex)  
        print("Max Value:", np.max(arr))


        #Confirmation
        # new = []
        # for i in range(0,len(x)):
        #     gugu = np.zeros(int(x[i]))
        #     gugu.fill(index[i])
        #     new.extend(gugu)
        # std = np.std(new)
        # mean = np.mean(new)
        # print("Std2:",std)
        # print("Mean2:",mean)      



        # scaled = (np.sqrt(2*np.pi)*std)/(np.exp(-np.square(arr-mean)/(2*(std**2))) * interval)
        # plt.plot(index, scaled,'--')
        

        plt.plot(index, arr,'--', label='distribution')
        plt.legend(loc="upper right")
        file = newFilename('distribution', suffix='.png', outdir=self.outputDir)
        plt.savefig(file)
        plt.show()
        return mean,std
        # plt.bar(index,x,color="blue", width=1)
        # plt.show()

    def plotFitDistributions(self, mean, std):
        firstNonZeroIndex = np.nonzero(self.dist)[0][0] #(np.where(x > 1)[0][0]) 
        LastNonZeroIndex = np.nonzero(self.dist)[0][-1]
        #get smaller distribution (The more indexes on the x axis the longer it takes for it to load)
        arr = self.dist[firstNonZeroIndex:LastNonZeroIndex]

        #set indexes to be from firstNonZeroIndex to LastNonZeroIndex.
        index = np.arange((firstNonZeroIndex + self.min), (LastNonZeroIndex + self.min), 1) #np.argmax(x),1)#-32768,32767,1)#np.argmin(x),np.argmax(x),1)
        
        #set indexes to be positive 
        index = index+abs( (firstNonZeroIndex + self.min))
        #set indexes to [0,1]
        index = index/len(index) 
        plt.plot(index, arr,'-', label='distribution')
        #arr = arr/np.max(arr)     #normalize
        
        x = np.linspace(0, 1, 100)
        x99 = np.linspace(mean - 3*std, mean + 3*std, 100)

        #for normal fit
        normalDistribution = np.exp(-np.square(x-mean)/(2*(std**2))) / (np.sqrt(2*np.pi)*std)
        plt.plot(x, normalDistribution,'-',label='normal')
        cdf = np.cumsum(normalDistribution)/len(x)
        plt.plot(x, cdf,'-',label='normalCdf')


        #for exponential fit
        lmbda = 1/mean
        exp = lmbda* np.exp(-lmbda * x)
        expNorm = np.exp(-lmbda * x)
        plt.plot(x, exp,'--', label='exp')
        plt.plot(x, expNorm,'-', label='expNorm')
        file = newFilename('fit', suffix='.png', outdir=self.outputDir)
        plt.savefig(file)
        plt.show()
  
    def flippedExponential(x, mean, std):
        return (np.sqrt(2*np.pi)*std)/(np.exp(-np.square(x-mean)/(2*(std**2))))


    def pixelDistInfo(self):
        info = {
            "Max Value At index" : str(np.argmax(self.dist)),
            "Max Value" : str(np.max(self.dist)),
            "Min Value At index" : str(np.argmin(self.dist)),
            "Min Value" : str(np.min(self.dist)),
            "First Non-zero Value at index" : str(np.where(self.dist > 0)[0][0]),
            "-in int16 index": str((np.where(self.dist > 0)[0][0]) + self.min),
            "Last Non-zero Value at index " : str(np.nonzero(self.dist)[0][-1]),
            "-in int16 index" : str(np.nonzero(self.dist)[0][-1] + self.min)
              }
        return info

    def save_info(self, mean, std):
        info = self.pixelDistInfo()
        
        info['mean'] = mean
        info['std'] = std
        print(json.dumps(info, indent=4))
        file = newFilename('info', suffix='.json', outdir=self.outputDir) 
        with open(file, 'w') as f: 
            f.write(json.dumps(info))

    def save_pickle(self, pickleName):
        file = newFilename(pickleName, suffix='.pickle', outdir=self.outputDir) 
        #mode = 'ab' if os.path.isfile(file) else 'wb'
        with open(file, 'wb') as f:
            print("Saving pickled error data...")
            pickle.dump(self.dist, f)

  
    def addPixelValues(self, image, totalPixels):
        img = Image.open(image)
        data = np.array(img)

        print("On image", image)
        for row in data:
            for pixelValue in row: 
                #if self.x[pixelValue - self.min] < 999999999999 :
                self.dist[pixelValue - self.min] += 1/totalPixels
        

    def process(self, image: MyImage or str):
        totalPixels = len(os.listdir(image)) * self.size[0] * self.size[1]
        for i in os.listdir(image):
            self.addPixelValues(os.path.join(image,i), totalPixels)

        mean,std = self.plotAndSaveImagePixelDist()
        self.plotFitDistributions(mean,std)
        self.save_pickle('time')
        self.save_info(mean,std)





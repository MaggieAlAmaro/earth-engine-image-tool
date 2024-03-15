import os
import random
import math
import time
import pickle,json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# from Image import str
from src.utils import makeOutputFolder, newFilename
from process import Processor


class DatasetSplit(Processor):
    def __init__(self, config_args):
        super().__init__()
        self.valPercent = config_args['valPercent']
        self.testPercent = config_args['testPercent']
        self.trainPercent = config_args['trainPercent']
        assert (self.valPercent + self.trainPercent + self.testPercent) == 1
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
        fn = os.path.join(self.outputDir,dataDir + '.txt')
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


    def splitIntoTrainTestValidationTxt(self, filenameTxt, validationPercent,testPercent):
        f = open(filenameTxt, "r")
        lines = f.readlines()
        f.close()
        print("Total files:" + str(len(lines)))

        validationSize = int(math.floor(len(lines) * validationPercent))
        testSize = int(math.floor(len(lines) * testPercent))
        trainSize = len(lines) - (validationSize + testSize)
        print("Train Size: " +str(trainSize))
        print("Test Size: " + str(testSize))
        print("Validation Size: " + str(validationSize))
        train = lines[:trainSize]
        val = lines[trainSize:(trainSize+validationSize)]
        test = lines[(trainSize+validationSize):]
                

        name = os.path.splitext(filenameTxt)[0]
        ftrain = name + '_train.txt'
        fval = name + '_validation.txt'
        ftest = name + '_test.txt'

        #train
        ft = open(ftrain, "w")
        ft.writelines(train)
        ft.close()

        #validation
        fv = open(fval, "w")
        fv.writelines(val)
        fv.close()

        
        #test
        fv = open(ftest, "w")
        fv.writelines(test)
        fv.close()


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


    def process(self, batch_path: str):
        fn = self.createFilenameTxt(batch_path)
        if fn != None:
            # self.splitIntoTrainValidationTxt(fn, self.ratio)
            self.splitIntoTrainTestValidationTxt(fn, self.valPercent,self.testPercent)


class Histogram():
    def __init__(self, size, pixel_range, min, distribution, bit_depth=None):
        self.size = size

        self.bit_depth = bit_depth
        self.pixel_range = pixel_range
        self.min = min

        self.dist = distribution
        # self.outputDir = makeOutputFolder('histogram')

    def calculateSuplementaryInfo(self):
        firstNonZeroIndex = np.nonzero(self.dist)[0][0] #(np.where(x > 1)[0][0]) 
        LastNonZeroIndex = np.nonzero(self.dist)[0][-1]

        #get smaller distribution (The more indexes on the x axis the longer it takes for it to load)
        self.compressedDist = self.dist[firstNonZeroIndex:LastNonZeroIndex]

        
        #set indexes, i.e the x axis values to be from firstNonZeroIndex to LastNonZeroIndex.
        index = np.arange((firstNonZeroIndex + self.min), (LastNonZeroIndex + self.min), 1) #np.argmax(x),1)#-32768,32767,1)#np.argmin(x),np.argmax(x),1)
        # set indexes to be positive 
        index = index + abs( (firstNonZeroIndex + self.min))
        # set indexes to [0,1]
        self.index = index/len(index) 

        self.n = np.sum(self.compressedDist )
        self.mean = np.dot(self.compressedDist ,self.index)/self.n
        print("Mean:", self.mean)
        self.std = np.sqrt(np.sum(self.compressedDist * np.square(self.index-self.mean))/self.n)
        print("Std:", self.std)
        print("Max Value:", np.max(self.compressedDist))

        print("Total Probability", np.sum(self.compressedDist))
        missing = 1 - np.sum(self.compressedDist)
        self.compressedDist += missing/len(self.compressedDist)
        print("Total Probability", np.sum(self.compressedDist))

    def plotDistribution(self, outputDir=None):
        # normalizedData = (x - firstNonZeroIndex)/(LastNonZeroIndex-firstNonZeroIndex)  

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
        

        plt.plot(self.index, self.compressedDist,'--', label='distribution')
        # plt.bar(index,x,color="blue", width=1)
        plt.legend(loc="upper right")
        if(outputDir != None):
            file = newFilename('distribution', suffix='.png', outdir=outputDir)
            plt.savefig(file)
        plt.show()
        

    def fitDistributions(self, outputDir=None):
        n_points = 2000

        plt.plot(self.index, self.compressedDist,'-', label='distribution')
        #arr = arr/np.max(arr)     #normalize
        
        x = np.linspace(0,1, n_points)
        x99 = np.linspace(self.mean - 5*self.std, self.mean + 5*self.std, n_points)  # > 100 of dist 

        #for normal fit
        normalDistribution = np.exp(-np.square(x99-self.mean)/(2*(self.std**2))) / (np.sqrt(2*np.pi)*self.std)
        plt.plot(x99, normalDistribution,'-',label='normal')
        dx = ((self.mean + 5*self.std) -(self.mean - 5*self.std))/n_points
        cdf = np.cumsum(normalDistribution) * dx
        plt.plot(x99, cdf,'-',label='normal Cdf')


        #for exponential fit
        lmbda = 1/self.mean
        exp = lmbda * np.exp(-lmbda * x)
        expCdf = 1 - np.exp(-lmbda * x)
        expNorm = np.exp(-lmbda * x)
        plt.plot(x, exp,'--', label='exp')
        plt.plot(x, expCdf,'--', label='exp Cdf')
        plt.plot(x, expNorm,'-', label='exp Normalized')
        plt.legend(loc="upper right")
        if(outputDir != None):
            file = newFilename('fit', suffix='.png', outdir=outputDir)
            plt.savefig(file)
        plt.show()

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
            self.pixel_range = config_args['pixel_range']   ##6500
            self.min = config_args['min']                   ##-10
        else:
            raise Exception("Bit Depth not recognized.")

        self.dist = np.zeros(shape=self.pixel_range) 
        self.outputDir = makeOutputFolder('histogram')



  
    def flippedExponential(x, mean, std):
        return (np.sqrt(2*np.pi)*std)/(np.exp(-np.square(x-mean)/(2*(std**2))))


    def pixelDistInfo(self):
        info = {
            "Max Value At index" : str(np.argmax(self.dist)),
            "Max Value" : str(np.max(self.dist)),
            "Min Value At index" : str(np.argmin(self.dist)),
            "Min Value" : str(np.min(self.dist)),
            "First Non-zero Value at index" : str(np.where(self.dist > 0)[0][0]),
            "   In int16 index": str((np.where(self.dist > 0)[0][0]) + self.min),
            " Last int16 index": str(np.nonzero(self.dist)[0][-1] + self.min),
            "Last Non-zero Value at index " : str(np.nonzero(self.dist)[0][-1]),
            "Mean" : self.histogram.mean,
            "Standard Deviation": self.histogram.std,
            }
        return info

    def save_info(self):
        info = self.pixelDistInfo()
        print(info)
        # print(json.dumps(info, indent=4))
        file = newFilename('info', suffix='.json', outdir=self.outputDir) 
        with open(file, 'w') as f: 
            f.write(json.dumps(info, indent=4))

    def save_pickle(self, pickleName):
        fileName = newFilename(pickleName, suffix='.pickle', outdir=self.outputDir) 
        #mode = 'ab' if os.path.isfile(file) else 'wb'
        with open(fileName, 'wb') as f:
            print("Saving pickled Histogram object...")
            pickle.dump(self.histogram, f)

  
    def addPixelValues(self, image, totalPixels):
        img = Image.open(image)
        data = np.array(img)

        print("On image", image)
        for row in data:
            for pixelValue in row: 
                #if self.x[pixelValue - self.min] < 999999999999 :
                # self.dist[pixelValue - self.min] += 1/totalPixels
                self.dist[pixelValue - self.min] += 1
        

    def process(self, image: str):
        totalPixels = len(os.listdir(image)) * self.size[0] * self.size[1]
        for i in os.listdir(image):
            self.addPixelValues(os.path.join(image,i), totalPixels)

        self.histogram = Histogram(self.size, self.pixel_range, self.min, self.dist ,self.bit_depth)
        self.histogram.calculateSuplementaryInfo()

        self.histogram.plotDistribution(self.outputDir)
        self.histogram.fitDistributions(self.outputDir)
        self.save_pickle('histogram')
        self.save_info()



def openHistogramPickle(filename) -> Histogram:
    with open(filename, 'rb') as f:
        hist: Histogram = pickle.load(f)
        return hist




if __name__ == '__main__':
    hist = openHistogramPickle("output\\histogram-2024-01-24-21-01-32\\histogram.pickle")
    hist.plotDistribution(save=False)
    hist.fitDistributions(save=False)
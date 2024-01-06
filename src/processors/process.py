import numpy as np
from PIL import Image
import rasterio
from itertools import product
import os, sys, math
from abc import ABC, abstractmethod
# import albumentations
# from albumentations import RandomRotate90


from Image import *
from utils import newFilename, makeOutputFolder




class Processor(ABC):
    def __init__(self) -> None:
        self.outputDir = None

    @abstractmethod
    def process(self, image: MyImage or str):
        pass

    def getOutputFolder(self) -> str:
        return self.outputDir
    
    def onBatchFinish(self):
        pass


class Crop(Processor):
    def __init__(self, config_args):
        super().__init__()
        self.outSize = config_args['size']
        self.outputDir  = makeOutputFolder('crop')

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

class Tile(Processor):
    def __init__(self, config_args):
        super().__init__()
        self.outputDir = makeOutputFolder('tile')
        self.tileSize = config_args['size']
        #self.outputDir = makeOutputFolder('tile')


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



# class DoRandomRotate90(RandomRotate90):
#     def __init__(self, always_apply: bool = False, p: float = 0.5):
#         super().__init__(always_apply, p)
    
#     def get_params(self):
#         # Random int in the range [1, 3]
#         return {"factor": random.randint(1, 3)}

# class Augment(Processor):
#     def __init__(self, config_args) -> None:
        # super().__init__()
#         self.typeT = config_args['type']
#         self.min = config_args['min']
#         self.max = config_args['max']
#         self.outputDir = makeOutputFolder('augment')



#     def getAugmentTransform():
#         transform = albumentations.Compose([
#             albumentations.OneOf([
#                 albumentations.HorizontalFlip(p=1),
#                 albumentations.VerticalFlip(p=1),
#             ], p=1),
#             DoRandomRotate90(p=0.5),
#             ]
#         )
#         return transform

#     def process(self, image: MyImage):
#         transform = self.getAugmentTransform()
#         fn = newFilename(image.image_filename, suffix=".png", outdir=self.outputDir)
#         img = Image.open(image)
#         img = transform(image=img)["image"]
#         img.save(fn)


#TODO make assert to check if the original image is 8bit
class To8bit(Processor):
    def __init__(self, config_args) -> None:
        super().__init__()
        self.min = config_args['min']
        self.max = config_args['max']
        self.outputDir  = makeOutputFolder('8bit')

        
    def to8bit(self, data):
        # make unsigned
        if self.min != 0:
            data = data - self.min                             
            max = self.max - self.min
            min = 0
        else:
            min = self.min
            max = self.max

        normalizedData = (data - min)/(max - min)
        rescaledData = normalizedData * (255 - 0)  # 8bit -> [0,255]
        return rescaledData.astype('uint8')

    def process(self, image: MyImage):
        resultData = self.to8bit(image.data)
        img = image.post_process_step(resultData)
        fn = newFilename(image.image_filename, suffix=".png", outdir=self.outputDir)
        img.save(fn)


#if 16bit (unsigned int), [0, 65455]
#if 16bit (signed int), [-32768, 32767]
#TODO make this work lol
class To16bit(Processor):
    def __init__(self, config_args) -> None:
        super().__init__()
        self.min = config_args['min']
        self.max = config_args['max']
        self.outputDir = makeOutputFolder('16bit')

        
    def to16bit(self, data):
        # TODO
        return
    
    def process(self, image: MyImage):
        img = self.image.convert(mode='I;16')

        #resultData = self.to16bit(image.data)
        #img = image.post_process_step(resultData)
        fn = newFilename(image.image_filename, suffix=".png", outdir=self.outputDir)
        img.save(fn)




class SizeCheckSort(Processor):
    def __init__(self, config_args) -> None:
        super().__init__()
        self.type = config_args['type']
        self.size = config_args['size']
        self.outputDir = makeOutputFolder('sizeCheck')
        self.notOk = os.path.join(self.outputDir,'notOk')
        self.ok = os.path.join(self.outputDir,'ok')
        os.mkdir(self.notOk)
        os.mkdir(self.ok)


        
    def sizeCheck(self, size):
        if self.type == 'less':
            return self.notOk if (np.array(self.size) < np.array(size)).any() else self.ok
        if self.type == 'exact':
            return self.notOk if (np.array(self.size) != np.array(size)).any() else self.ok
        
    
    def process(self, image: MyImage):
        resultOutDir = self.sizeCheck(image.size)
        fn = newFilename(image.image_filename, suffix=".png", outdir=resultOutDir)
        image.image.save(fn)


    def getOutputFolder(self) -> str:
        return os.path.join(self.outputDir,'ok') 



class ChangeMode(Processor):
    def __init__(self, config_args) -> None:
        super().__init__()
        self.mode = config_args['mode']
        self.format = config_args['format']
        self.outputDir = makeOutputFolder('mode')
    
    def process(self, image: MyImage):
        imageConverted = image.image.convert(mode=self.mode)
        fn = newFilename(image.image_filename, suffix="."+self.format, outdir=self.outputDir)
        imageConverted.save(fn)



class Info(Processor):
    def __init__(self, config_args) -> None:
        super().__init__()
        # self.outputDir = makeOutputFolder('info')

        
    def info(self, image: MyImage):
        img = rasterio.open(image.image_filename)
        print("Coordinate bounds:", img.bounds)
        print("Crs:", img.crs)
        print()
        print("Shape:", image.data.shape)
        print("PIL Info:", image.image.info)
        print("PIL mode:", image.image.mode)
        if len(image.data.shape) == 2:
            print("For alpha channel:")
            print(" Mean", np.mean(image.data), "\n Std ", np.std(image.data))
        else:
            for channel in range(len(image.image.getbands())):
                print("For channel " + str(channel) + ":")
                print(" Mean", np.mean(image.data[channel]), "\n Std ", np.std(image.data[channel]))
        print("==============================================")
        print()


    def process(self, image: MyImage):
        self.outputDir = os.path.dirname(image.image_filename)
        self.info(image)

    
    



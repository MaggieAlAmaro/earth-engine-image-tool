import numpy as np
import argparse
from PIL import Image
import os
#import rasterio
from Image import MyImage
from utils import newFilename, makeOutputFolder
from src.processors.process import Processor
import rasterio

def getParser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs, formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")
    mergeParser = subparsers.add_parser("merge", help="merges an RGB and a 1 channel (A) images")
    mergeParser.add_argument('rgb_name', type=str, help='Name of image or directory to merge with.')    
    mergeParser.add_argument('a_name', type=str, help='Name of image or directory to merge with.')    
    mergeParser.add_argument('-o','--out_dir', type=str, help='Output Directory')

    seperateParser = subparsers.add_parser("seperate", help="separate the RGB and A channels of an RGBA image")
    seperateParser.add_argument('rgba_name', type=str, help='Name of image or directory to merge with.')
    seperateParser.add_argument('-o','--out_dir', type=str, help='Output Directory')

    stdParser = subparsers.add_parser("stdDev", help="Convert image to 8bit")
    stdParser.add_argument('image_filename', type=str, help='Name of image')   
    stdParser.add_argument('-limit', type=float, help='Type of scaling', default=4.5)

    #return mergeParser, seperateParser
    return parser

def checkBounds(fileRGB, fileA):
    img1 = rasterio.open(fileRGB)
    img2 = rasterio.open(fileA)
    return img1.bounds == img2.bounds
    
def myLogToJsonDict(logFile, jsonFilename):
    with open(logFile, 'r') as f:
        gsData = f.read()
        gsData = gsData[:-2]    #remove newLine and last comma
    with open(jsonFilename, 'w') as f:
        f.write("{ \n" + gsData + "\n }")



class SeparateRGBFromA(Processor):
    def __init__(self, config_args) -> None:
        super().__init__()
        self.outputDir = makeOutputFolder('separateRGBA')
        self.aDir = os.path.join(self.outputDir,'a')
        self.rgbDir = os.path.join(self.outputDir,'rgb')
        os.mkdir(self.aDir)
        os.mkdir(self.rgbDir)

    def separateAandRGB(self, image : Image, image_filename):
        fna = newFilename(image_filename, suffix=".png", outdir=self.aDir)
        fnrgb = newFilename(image_filename, suffix=".png", outdir=self.rgbDir)

        r, g, b, a = image.split()
        a = a.convert('L')
        rgb = Image.merge('RGB', (r, g, b))

        rgb.save(fnrgb)
        a.save(fna)

    def process(self, image: MyImage):
        assert(image.size[2] == 4)
        self.separateAandRGB(image.image, image.image_filename)
        




from src.processors.two_data_processor import zipData

class MergeRGBandA(Processor):
    def __init__(self, config_args) -> None:
        super().__init__()
        self.outputDir = makeOutputFolder('mergeRGBA')
        self.data_target = config_args['alpha_target']

    def mergeRGBandAImages(self, rgb_name, a_name) -> Image.Image:
        assert(rgb_name.size[0] == 3 and len(a_name.size) == 2)
        rgb = Image.open(rgb_name)
        a = Image.open(a_name)
        a = a.convert('L')
        r, g, b = rgb.split()
        return Image.merge('RGBA', (r, g, b, a))

    def mergeRGBAImagesBatch(self, zippedData):
        for image1_name, image2_name in zippedData:
            resultImage = self.mergeRGBandAImages(image1_name,image2_name)
            fnrgba = newFilename(image1_name + image2_name, suffix=".png", outdir=self.outputDir)
            resultImage.save(fnrgba)


    def process(self, data_path: str):
        if(os.path.isdir(data_path) and os.path.isdir(self.data_target)):
            self.mergeRGBAImagesBatch(zipData(data_path,self.data_target))
        


import json
class MatchRGBA(Processor):
    def __init__(self, config_args) -> None:
        super().__init__()
        self.outputDir = makeOutputFolder('matchRGBA')
        self.alpha_format = config_args['alpha_format']
        self.rgb_target = config_args['rgb_target']
        self.rgb_format = config_args['rgb_format']

        if 'rgb_log' in config_args:
            self.jsonFolder = makeOutputFolder('logToJson')
            self.rgb_json = os.path.join(self.jsonFolder, 'rgb')
            myLogToJsonDict(config_args['rgb_log'],self.rgb_json)
        else:
            self.rgb_json = config_args['rgb_json']

        if 'a_log' in config_args:
            if not hasattr(self, 'jsonFolder'):
                self.jsonFolder = makeOutputFolder('logToJson')
            self.a_json = os.path.join(self.jsonFolder, 'a')
            myLogToJsonDict(config_args['a_log'],self.a_json)
        else:
            self.a_json = config_args['a_json']
            


    def matchWithJson(self, data_path):
        with open(self.a_json, 'r') as f:
            aDict = json.load(f)
        with open(self.rgb_json, 'r') as f:
            rgbDict = json.load(f)
            # unique_values = [k for k,v in aDict.items() if list(aDict.values()).count(v)>1]
            # print(unique_values)
            # print([v for k, v in aDict.items()])

        
        skipNext = False
        alphaIter = iter(aDict.items())
        # gsIter =iter(sorted(aDict.items()))
        for rgbK, v in rgbDict.items():
            if not skipNext:
                currentAlphaK, currentAlphaV  = next(alphaIter)
            else:
                skipNext = False
            while v != currentAlphaV: #TODO set up a tolerance
                if ((currentAlphaV[0] > v[0] and currentAlphaV[1] >= v[1]) or (currentAlphaV[1] > v[1])):
                    print("Didn't find pair for image", rgbK)
                    skipNext = True
                    break
                currentAlphaK, currentAlphaV  = next(alphaIter)
            
            if v == currentAlphaV:
                rgbFile = os.path.join(self.rgb_target, rgbK + self.rgb_format)
                aFile = os.path.join(data_path, currentAlphaK + self.alpha_format)
                if self.rgb_format == ".tif" and  self.alpha_format == ".tif" :
                    if not checkBounds(rgbFile,aFile):
                        print("Bounds DIFFERENT")
                        continue
                resultImage = MergeRGBandA.mergeRGBandAImages(rgbFile, aFile)
                fn = newFilename(rgbK+"_"+currentAlphaK, suffix=".png",outdir=self.outputDir)
                resultImage.save(fn)
                # print(os.path.join(rgbDir,currentGsK+".tif"))

        

    def process(self, data_path: str):
        self.matchWithJson(data_path)

        

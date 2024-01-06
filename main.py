import matplotlib.pyplot as plt
from PIL import Image
import os, sys
import yaml
from typing import List, Dict


#import analytics, tests, visualize, rgba
import importlib
from src.processors.process import Processor
from src.processors.batch_processor import DatasetSplit, HistogramAnalysis

from Image import MyImage
# from abc import ABC, abstractmethod
# import numpy as np
 

'''
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


    # parser = getParser()
    # args = parser.parse_args()
    # print(args)
    # if args.command == 'blah blah':
    #     processDirOrFile(blah)
''' 

class ImageProcessingFacade():
    def __init__(self, image_directory, processes: Dict[str,Processor], imageType: MyImage) -> None:
        self.image_directory = image_directory
        self.processes = processes
        self.imageType = imageType

    def getFiles(self):
        return os.listdir(self.image_directory)


    def processImage(self, processor: Processor):
        for filename in os.listdir(self.image_directory):
            f = os.path.join(self.image_directory, filename)
            imgObj = self.imageType(f)
            processor.process(imgObj)
        
    def processBatch(self, processor: Processor):
        processor.process(self.image_directory)

    def conductAllImageOperations(self, operations: List[str]): 
        for op in operations:
            self.processImage(self.processes[op])
            self.processes[op].onBatchFinish()
            self.image_directory = self.processes[op].getOutputFolder()

        
    def conductAllBatchOperations(self, operations: List[str]): 
        for op in operations:
            self.processBatch(self.processes[op])
            



def getLineValue(img, axis = 0, index = 1025):
    data = img[index,:] if(axis == 0) else img[:,index]
    print(data)
    return (data == data[0]).all()





#TODO: use omegaconf to merge config yaml files and cli commands into 1 thing
if __name__ == '__main__':
    with open("configs\default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open("configs\operations.yaml", "r") as f:
        ops = yaml.safe_load(f)
    
    
    # init image class
    imgType = MyImage


    # init needed processors
    factoriesDict = {}
    if 'image' in ops['operations']:
        for op in ops['operations']['image']:
            module, cls = config[op]['class'].rsplit(".", 1)
            _class = getattr(importlib.import_module(module, package=None), cls)
            factoriesDict[op] = _class(config[op]['params'])#**config[op].get('params',dict())) 
            
    if 'batch' in ops['operations']:
        for op in ops['operations']['batch']:
            module, cls = config[op]['class'].rsplit(".", 1)
            print(config[op])
            _class = getattr(importlib.import_module(module, package=None), cls)
            factoriesDict[op] = _class(config[op]['params'])#**config[op].get('params',dict())) 
    
    if os.path.isdir(ops['data_target']):
        ipf = ImageProcessingFacade(ops['data_target'], factoriesDict, imgType)
        if 'image' in ops['operations']:
            ipf.conductAllImageOperations(ops['operations']['image'])
        if 'batch' in ops['operations']:
            ipf.conductAllBatchOperations(ops['operations']['batch'])
    else:
        print("Error. Check operation file.")

    # print()
    # print("FINISHED")



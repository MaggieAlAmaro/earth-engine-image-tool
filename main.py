import matplotlib.pyplot as plt
from PIL import Image
import os, sys
import yaml
from typing import List, Dict


#import analytics, tests, visualize, rgba
from process import Processor
import importlib
from Image import MyImage, RGBAImage, GrayscaleImage
from datasetSplit import DatasetSplit, HistogramAnalysis

 



class ImageProcessingFacade():
    def __init__(self, image_directory, processes: Dict[Processor,str], imageType: MyImage) -> None:
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

    #TODO split this up into one off funtions or else it doesnt make sense
    def conductAllImageOperations(self, operations: List[str]): 
        for op in operations:
            self.processImage(self.processes[op])
            self.image_directory = self.processes[op].outputDir
        
    def conductAllBatchOperations(self, operations: List[str]): 
        for op in operations:
            self.processBatch(self.processes[op])
            





#TODO: use omegaconf to merge config yaml files and cli commands into 1 thing
if __name__ == '__main__':
    with open("process_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open("operations.yaml", "r") as f:
        ops = yaml.safe_load(f)
    
    
    # init image class
    imgType = MyImage
    if ops['data_type'] == "RGBA":
        imgType = RGBAImage
    elif ops['data_type'] == "grayscale":
        imgType = GrayscaleImage


    # init needed processors
    factoriesDict = {}
    for op in ops['operations']:
        module, cls = config[op]['class'].rsplit(".", 1)
        _class = getattr(importlib.import_module(module, package=None), cls)
        factoriesDict[op] = _class(config[op]['params'])#**config[op].get('params',dict())) 
    
    # h = HistogramAnalysis(config['histogram'])
    # h.process(ops['data_target'])
    if os.path.isdir(ops['data_target']):
        ipf = ImageProcessingFacade(ops['data_target'], factoriesDict, imgType)
        # ipf.conductAllImageOperations(ops['operations'])
        ipf.conductAllBatchOperations(ops['operations'])
    else:
        print("Error. Check operation file.")

    # print()
    # print("FINISHED")



    # parser = getParser()
    # args = parser.parse_args()
    # print(args)
    # if args.command == 'blah blah':
    #     processDirOrFile(blah)
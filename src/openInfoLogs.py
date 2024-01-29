import json, pickle
import numpy as np
import argparse
from processors.batch_processor import Histogram

def getParser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs, formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")
    openParser = subparsers.add_parser("open", help="Convert to png")   
    openParser.add_argument('pickle_filename', type=str, help='Name of image')  
    openParser.add_argument('-min', type=int, help='Name of image',default= -32768)#-10)


    
def openImage(filename):
    with open(filename, 'rb') as f:
        pixels: np.array = pickle.load(f)
        return pixels
    
def openHistogramPickle(filename) -> Histogram:
    with open(filename, 'rb') as f:
        hist: Histogram = pickle.load(f)
        return hist




if __name__ == '__main__':
    hist = openHistogramPickle("output\\GRAYSCALE_TREATED\histogram-2024-01-24-23-39-57\\histogram.pickle")
    hist.plotDistribution()
    hist.fitDistributions()
    # parser = getParser()
    # args = parser.parse_args()
    # print(args)
    # if args.command == 'open':
    #     processDirOrFile(blah)
    # with open("info.json", "r") as f:
    #     data = json.loads(f)
    #     print(data)

import time
import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image
import os

from utils import processDirOrFile, newFilename, makeOutputFolder


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
    #return mergeParser, seperateParser
    return parser


def separateAandRGB(image, **kwargs):
    outFn = kwargs.get('out', None)
    aDir = kwargs.get('a', a)
    rgbDir = kwargs.get('rgb', rgb)
    if(outFn == None):
        newRGBFileName = newFilename(image, suffix="RGB.png")
        newRGBFileName = newFilename(image, suffix="A.png")
    else:
        newRGBFileName = outFn + "RGB.png"
        newAFileName = outFn + "A.png"

    img = Image.open(image)
    r, g, b, a = img.split()
    a = a.convert('L')
    rgb = Image.merge('RGB', (r, g, b))
    
    

    rgb.save(os.path.join(rgbDir,newRGBFileName))
    a.save(os.path.join(aDir,newAFileName))


def mergeRGBA(fileRGB, fileA, **kwargs):
    outFn = kwargs.get('out', None)
    if (outFn == None):
        fn = newFilename(fileRGB, suffix="RGBA.png")
    else:
        fn = outFn

    #TODO: mcheck if filename part is same, if so get new filenameFrom file

    rgb = Image.open(fileRGB)
    a = Image.open(fileA)
    a = a.convert('L')
    r, g, b = rgb.split()
    rgba = Image.merge('RGBA', (r, g, b, a))
    rgba.save(fn)

    
if __name__ == '__main__':
    parser = getParser()
    args = parser.parse_args()
    print(args)
    if args.command == 'merge':
        processDirOrFile(mergeRGBA, target=args.rgb_name, destination=args.a_name)
    if args.command == 'seperate':
        outputDir = makeOutputFolder('separate')
        rgbDir = os.path.join(outputDir,'rgb')
        aDir = os.path.join(outputDir,'a')
        os.mkdir(rgbDir)
        os.mkdir(aDir)
        processDirOrFile(separateAandRGB, target=args.rgba_name)
#import rasterio
import matplotlib.pyplot as plt
#from rasterio.plot import show
#from rasterio.merge import merge
import numpy as np
import argparse
from PIL import Image
from itertools import product
import os, sys, math, typing


from utils import processDirOrFile, newFilename, makeOutputFolder

def getParser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs, formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")
    mergeParser = subparsers.add_parser("compare", help="check if 2 images are the same")
    mergeParser.add_argument('image1', type=str, help='Name of image or directory to merge with.')    
    mergeParser.add_argument('image2', type=str, help='Name of image or directory to merge with.')  

    mergeParser = subparsers.add_parser("subtract", help="check if 2 images are the same")
    mergeParser.add_argument('image1', type=str, help='Name of image or directory to merge with.')    
    mergeParser.add_argument('image2', type=str, help='Name of image or directory to merge with.')  

    mergeParser = subparsers.add_parser("sizeCheck", help="check if 2 images are the same")
    mergeParser.add_argument('image', type=str, help='Name of image or directory to merge with.')    
    mergeParser.add_argument('size', type=int, help='Name of image or directory to merge with.')    
    mergeParser.add_argument('type', type=str, help='Name of image or directory to merge with.', choices={'exact','less'})    
    #return mergeParser, seperateParser
    return parser


# ONLY CHECKS IF VALUES ARE LESS THAN EXPECTED VALUE
def sizeCheck(image,  **kwargs):
    size = kwargs.get('size')
    t = kwargs.get('type', 'exact')
    remove = False
    with Image.open(image) as img:
        if t == 'less':
            if type(size) is tuple:
                expr = img.size[0] < size[0] or img.size[1] < size[1]
            elif type(size) is int:
                expr = img.size[0] < size or img.size[1] < size
        elif t == 'exact':
            if type(size) is tuple:
                expr = img.size[0] != size[0] or img.size[1] != size[1]
            elif type(size) is int:
                expr = img.size[0] != size or img.size[1] != size
        if expr:
            print("Image " + image + " is NOT of expected size")
            print(img.size)
            remove = True
    #if remove:
    #    os.remove(image)

    # with rasterio.open(f) as img:
    #     bandNbr = [band for band in range(1,img.count + 1)]
    #     data = img.read(bandNbr)
    #     print(data.shape == expectedSize)
             

class Sub():
    def __init__(self) -> None:
        self.mean = []

    def subtractImage(self, images,  **kwargs):
        outdir = kwargs.get('outdir')
        path1 = kwargs.get('path1')
        path2 = kwargs.get('path2')
        for image1, image2 in images:
            img1 = Image.open(os.path.join(path1,image1))
            img2 = Image.open(os.path.join(path2,image2))
            data1 = np.array(img1)
            data2 = np.array(img2)
            #result = abs(data1 - data2)
            #img = Image.fromarray(result)
            mse = (np.square(data1 - data2)).mean()
            #print("MSE:", mse)
            self.mean.append(mse)
            #print(np.std(result,axis=2) )
            #print([for i in range(0,len(result.shape))])
            #newFileName = newFilename(image1, suffix=".png",outdir=outdir)
            #img.save(newFileName)


def compare(image1, image2, **kwargs):
    img1 = Image.open(image1)
    img2 = Image.open(image2)
    img1 = np.array(img1)
    img2 = np.array(img2)
    if(img1.shape != img2.shape):
        print("The image shapes don't match:" + str(img1.shape) + "and" + str(img2.shape))
        print(image1)
        print(image2)
        return False      
    elif((img1 == img2).all()):
        #print("The images are EQUAL.")
        return True
    else:
        print("The images are NOT equal.")
        return False

def compareDir(target, dest, **kwargs):
    if(os.path.isdir(target) and os.path.isdir(dest)):
        for target_file, destination_file in zip(os.listdir(target), os.listdir(dest)):
            target_f = os.path.join(target, target_file)
            destination_f = os.path.join(dest, destination_file)
            if os.path.isfile(target_f) and os.path.isfile(destination_f):
                if not compare(target_f,destination_f):
                    return False
    elif(os.path.isfile(target) and os.path.isfile(dest)):
        if not compare(target,dest):
            return False
    else:
        return False
    print("All images are EQUAL.")
    return True

    
if __name__ == '__main__':
    parser = getParser()
    args = parser.parse_args()
    print(args)
    if args.command == 'compare':
        compareDir(target=args.image1, dest=args.image2)
        #processDirOrFile(compare, target=args.image1, destination=args.image2)
    if args.command == 'sizeCheck':
        processDirOrFile(sizeCheck, target=args.image, size=args.size, type=args.type)
    if args.command == 'subtract':
        if(os.path.isdir(args.image1) and os.path.isdir(args.image2)):
            #outputDir = makeOutputFolder('subtract')
            sub = Sub()
            sub.subtractImage(zip(os.listdir(args.image1), os.listdir(args.image2)), path1=args.image1, path2=args.image2)
            print("Mean of MSE", np.array(sub.mean).mean())
    else:
        img = 'output\\GRAYSCALE_TREATED\\stdDev-2023-12-15-10-02-23\\below\\6_0_512.png'
        image = Image.open(img)
        data = np.array(image)
        print(np.std(data))
        
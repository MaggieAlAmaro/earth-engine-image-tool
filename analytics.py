import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image
import os, sys, math
import pickle, typing, time

from utils import processDirOrFile, newFilename
from visualize import grid


plt.style.use('default')





def getParser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs, formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")
    openParser = subparsers.add_parser("open", help="Convert to png")   
    openParser.add_argument('pickle_filename', type=str, help='Name of image')  
    openParser.add_argument('-min', type=int, help='Name of image',default= -32768)#-10)  
    
    infoParser = subparsers.add_parser("info", help="Convert to png")   
    infoParser.add_argument('image_filename', type=str, help='Name of image')  

    distParser = subparsers.add_parser("hist", help="Convert to png")   
    distParser.add_argument('image_filename', type=str, help='Name of image')  
    distParser.add_argument('-min', type=int, help='Name of image',default= -32768)#-10)  
    distParser.add_argument('-max', type=int, help='Name of image',default=6500)  
    return parser
    

x = np.zeros(shape=65536) #6511

def add(image, **kwargs):
    min = kwargs.get('min', -32768) 
    img = Image.open(image)
    data = np.array(img)

    print("on image", image)
    global x
    for row in data:
        for pixelValue in row: 
            if x[pixelValue - min] < 999999999999 :
                x[pixelValue - min] += 1 
    
    # x = x / imgSize (noramlize)



# def get_error_logs_from_pickle(filename) -> typing.Tuple[ErrorLogger]:
#     with open(filename, 'rb') as f:
#         a = 0
#         objs = []
#         while 1:
#             try:
#                 a += 1
#                 tError: ErrorLogger = pickle.load(f)
#                 print("Loading error log "+str(a))
#                 objs.append(tError)
#             except EOFError:
#                 break

#     return objs

def plotError(x,minL):
    firstNonZeroIndex = np.nonzero(x)[0][0] #(np.where(x > 1)[0][0]) 
    LastNonZeroIndex = np.nonzero(x)[0][-1]
    index = np.arange((firstNonZeroIndex + minL), (LastNonZeroIndex + minL), 1) #np.argmax(x),1)#-32768,32767,1)#np.argmin(x),np.argmax(x),1)
    x = x[firstNonZeroIndex:LastNonZeroIndex]

    x = x/np.max(x)     #normalize
    
    plt.plot(index, x)
    plt.show()

    # plt.bar(index,x,color="blue", width=1)
    # plt.show()



def open_pickle(filename):
    with open(filename, 'rb') as f:
        pixels: np.array = pickle.load(f)
        return pixels



def save_pickle(data):
    file = newFilename(os.path.join('output', str(time.strftime("%Y-%m-%d-%H-%M-%S"))), suffix='.pickle') #os.path.join(logdir,filename)
    #mode = 'ab' if os.path.isfile(file) else 'wb'
    with open(file, 'wb') as f:
        print("Saving pickled error data...")
        pickle.dump(data,f)


def pixelDistInfo(x,min):
    print("Max Value At index: " + str(np.argmax(x)))
    print("Max Value: " + str(np.max(x)))
    print("Min Value At index: " + str(np.argmin(x)))
    print("Min Value: " + str(np.min(x)))
    print("First Non-zero Value at index: " + str((np.where(x > 1)[0][0])) + " in int16 index: " + str((np.where(x > 1)[0][0]) +min))
    print("Last Non-zero Value at index: " + str(np.nonzero(x)[0][-1]) + " in int16 index: " + str(np.nonzero(x)[0][-1] +min))

isStd = []

def stdDev(image,**kwargs):
    img = Image.open(image)
    print("on image", image)
    data = np.array(img)
    stdD = np.std(data)
    print("Standard Deviation is: " + str(stdD))
    print()
    global isStd
    if(stdD > 1.6):
        isStd.append(image)



def imageInfo(image):
    img = Image.open(image)
    print("Image Size: " +str( img.size))
    print("Image Mode: " +str( img.mode))
    #data = np.array(img)



if __name__ == '__main__':
    parser2 = getParser()
    args = parser2.parse_args()
    print(args)
    if args.command == 'open':
        distribution = open_pickle(args.pickle_filename)
        min = args.min
        x = distribution
    # elif args.command == 'info':
    #     processDirOrFile(imageInfo, target=args.image_filename)
    elif args.command == 'hist':
        processDirOrFile(add, target=args.image_filename, min=args.min, max=args.max)
        min = args.min
        save_pickle(x)
    pixelDistInfo(x,min)
    plotError(x,min)

    
    #grid("data\\stdv",5)
    #processDirOrFile(add, target="..\\Elevation")
    #grid(isStd,5)

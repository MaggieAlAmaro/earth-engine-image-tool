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

    
    tParser = subparsers.add_parser("transform", help="Convert to png")   
    tParser.add_argument('pickle_filename', type=str, help='Name of image')  
    tParser.add_argument('-min', type=int, help='Name of image',default= -32768)#-10)  
    tParser.add_argument('-max', type=int, help='Name of image',default=255)  
    
    infoParser = subparsers.add_parser("info", help="Convert to png")   
    infoParser.add_argument('image_filename', type=str, help='Name of image')  

    distParser = subparsers.add_parser("hist", help="Convert to png")   
    distParser.add_argument('image_filename', type=str, help='Name of image')  
    distParser.add_argument('-min', type=int, help='Name of image',default= -32768)#-10)  
    distParser.add_argument('-max', type=int, help='Name of image',default=6500)  
    return parser
    

global x
x = np.zeros(shape=256) #6511 #65536

def add(image, **kwargs):
    min = kwargs.get('min', -32768) 
    img = Image.open(image)
    data = np.array(img)

    print("on image", image)
    global x
    for row in data:
        for pixelValue in row: 
            if x[pixelValue + abs(min)] < 999999999999 :
                x[pixelValue +  abs(min)] += 1 
    
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

def plotError(arr,minL):
    firstNonZeroIndex = np.nonzero(arr)[0][0] #(np.where(x > 1)[0][0]) 
    LastNonZeroIndex = np.nonzero(arr)[0][-1]
    print((firstNonZeroIndex + minL))
    index = np.arange((firstNonZeroIndex + minL), (LastNonZeroIndex + minL), 1) #np.argmax(x),1)#-32768,32767,1)#np.argmin(x),np.argmax(x),1)
    index = index+abs( (firstNonZeroIndex + minL))
    index = index/len(index)
    arr = arr[firstNonZeroIndex:LastNonZeroIndex]

    n = np.sum(arr)


    mean = np.dot(arr,index)/n
   

    print("Mean:",mean)
    lmbda = 1/mean
    std = np.sqrt(np.sum(arr*np.square(index-mean))/n)
    print("Std:",std)

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

    

    # normalizedData = (x - firstNonZeroIndex)/(LastNonZeroIndex-firstNonZeroIndex)  
    print(np.max(arr))
    arr = arr/np.max(arr)     #normalize
    
    #prob dist
    #x = x/n

    # thing = np.linspace(mean - 3*std, mean + 3*std, 1000)
    
    thing = np.linspace(0, 1, 100)
    f = np.exp(-np.square(thing-mean)/(2*(std**2)))/(np.sqrt(2*np.pi)*std)
    plt.plot(thing, f,'-')



    thing2 = np.linspace(0, mean + 3*std, 100)

    # uniform = 1/ (b - a)
    interval = 1 - 0
    # #scaler = (np.sqrt(2*np.pi)*std)/(np.exp(-np.square(thing-mean)/(2*(std**2))) * interval)
    # print("uhuhu",(np.sqrt(2*np.pi)*std)* 0.9/(np.exp(-np.square(0.9-mean)/(2*(std**2))) * interval))
    # plt.plot(thing, scaler,'-')


    # plt.show()



    plt.plot(index, arr,'--')

    # scaled = (np.sqrt(2*np.pi)*std)/(np.exp(-np.square(arr-mean)/(2*(std**2))) * interval)
    # plt.plot(index, scaled,'--')
    exp = lmbda* np.exp(-lmbda * thing)
    expNorm = np.exp(-lmbda * thing)
    plt.plot(thing, exp,'-')
    plt.plot(thing, expNorm,'-')
    plt.show()
    return mean,std
    # plt.bar(index,x,color="blue", width=1)
    # plt.show()

def function(x, mean, std):
    return (np.sqrt(2*np.pi)*std)/(np.exp(-np.square(x-mean)/(2*(std**2))))


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





def imageInfo(image):
    img = Image.open(image)
    print("Image Size: " +str( img.size))
    print("Image Mode: " +str( img.mode))
    #data = np.array(img)

def myfunc(x):
    return x**(1/300)


if __name__ == '__main__':
    parser2 = getParser()
    args = parser2.parse_args()
    print(args)
    if args.command == 'open':
        distribution = open_pickle(args.pickle_filename)
        min = args.min
        x = distribution


    
    elif args.command == 'transform':
        distribution = open_pickle(args.pickle_filename)
        #normalize
        min = 0 #args.min

        myfunc_vec = np.vectorize(myfunc)
        normalizedData = myfunc_vec(distribution)
        
        #normalizedData = np.cbrt(distribution)
        print(np.max(normalizedData))
        x = normalizedData

    # elif args.command == 'info':
    #     processDirOrFile(imageInfo, target=args.image_filename)
    elif args.command == 'hist':
        processDirOrFile(add, target=args.image_filename, min=args.min, max=args.max)
        min = args.min
        # x[32768] -= 3633938
        save_pickle(x)
    pixelDistInfo(x,min)
    plotError(x,min)
    
    #grid("data\\stdv",5)
    #processDirOrFile(add, target="..\\Elevation")
    #grid(isStd,5)

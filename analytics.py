import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image
import os, sys, math
import pickle, typing, time

from utils import processDirOrFile, newFilename


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "fileOrDir",
        type=str,
        nargs="?",
        action='store_const',
        help="Name of image or directory to work with.",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        nargs="?",
        help="name of raster image",
    )
    parser.add_argument(
        "-m",
        "--merge",
        nargs='+',
        type=str,
        help="merge 2 tif images",
    )
    parser.add_argument(
        "-cm",
        "--channelMerge",
        nargs='+',
        type=str,
        help="merge an rgb and a 1 channel image",
    )
    parser.add_argument(
        "-c",
        "--compare",
        nargs='+',
        type=str,
        help="compares 2 images and evaluates if they are the same",
    )
    parser.add_argument(
        "-t",
        "--toType",
        nargs='+',
        type=str,
        help="convert img To Png",
    )
    parser.add_argument(
        "-s",
        "--separate",
        const=True,
        nargs="?",
        type=str,
        help="Separate RGB and A from RGBA image into 2 seperate images",
    )
    return parser


    

x = np.ones(shape=65535)

def add(image, **kwargs):
    img = Image.open(image)
    print("on image", image)
    data = np.array(img)
    #TODO get min value processed to cap graph x axis and make graph smaller
    #print(data)
    global x
    for i in data:
        for e in i:
            if x[e + 32768] < 999999999999 :
                x[e + 32768] += 1 # e + 10
    
    # x = x / 1024


    


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

def plotError(x):
    x = x[np.argmin(x)+32768:np.argmax(x)+32768]
    index = np.arange(np.argmin(x),np.argmax(x),1)#-32768,32767,1)
    print(x.shape, index.shape)
    # x = x/(1024*11)
    # print(x[12768])
    plt.bar(index,x,color="blue", width=1)
    plt.show()



def open_pickle(filename):
    with open(filename, 'rb') as f:
        pixels: np.array = pickle.load(f)
        print(pixels)

def save_pickle():
    file = newFilename(os.path.join('data',str(time.strftime("%Y-%m-%d-%H-%M"))),suffix='.pickle') #os.path.join(logdir,filename)
    mode = 'ab' if os.path.isfile(file) else 'wb'
    with open(file, mode) as f:
        print("Saving pickled error data...")
        pickle.dump(x,f)
        print("Data saved in " + file)

# def valueDistribution():

def datasetInfo(x):
    print("Max Value At index:" + str(np.argmax(x)))
    print("Max Value:" + str(np.max(x)))
    print("Min Value At index:" + str(np.argmin(x)))
    print("Min Value:" + str(np.min(x)))
    print("Min Value At index:" + str((np.where(x > 1)[0][0])))

if __name__ == '__main__':
    #open_error_pickle("wubelev.pickle")
    
    #x = np.ones(shape=65535)
    processDirOrFile(add, target="data\\vv")
    datasetInfo(x)
    # parser = get_parser()
    # args = parser.parse_args()
    
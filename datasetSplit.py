import os
# import shutil
import random
import argparse
import math

def getParser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('dir', type=str, help='Name of data directory') 
    parser.add_argument('-r','--ratio', type=int, help='Ratio of training to validation images', default=1/6) 
    return parser

def filenameTxt(dataDir):
    # fn = os.path.basename(dataDir) + '.txt'
    fn = dataDir + '.txt'
    f = open(fn, "a")
    for filename in os.listdir(dataDir):
        f.write(filename+"\n")
    f.close()
    print("Created new file: " + fn)
    return fn


def shuffleFile(filename):
    with open(filename,'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
    with open(filename,'w') as f:
        f.writelines(lines)


def createFilenameTxt(dataDir,shuffleTxt=True):
    # dataSize = len([name for name in os.listdir(h) if os.path.isfile(h+os.sep+name)])
    if not checkIfDirValid(dataDir):
        return
    fn = filenameTxt(dataDir)
    if shuffleTxt:
        shuffleFile(fn)
    return fn

def checkIfDirValid(dataDir):
    for f in os.listdir(dataDir):
        if  not f.endswith('.png'):
            print('Found file with incorrect format (not PNG):' + f)
            return False
    return True




def splitIntoTrainValidationTxt(filenameTxt, ratio):
    f = open(filenameTxt, "r")
    lines = f.readlines()
    f.close()
    print("Total files:" + str(len(lines)))

    validationSize = int(math.floor(len(lines) * ratio))
    trainSize = len(lines) - validationSize
    print("Train Size: " +str(trainSize))
    print("Validation Size: " + str(validationSize))
    train = lines[:trainSize]
    val = lines[trainSize:]
            
    name = os.path.splitext(filenameTxt)[0]
    ftrain = name + '_train.txt'
    fval = name + '_validation.txt'

    #train
    ft = open(ftrain, "w")
    ft.writelines(train)
    ft.close()

    #validation
    fv = open(fval, "w")
    fv.writelines(val)
    fv.close()








if __name__ == '__main__':
    parser = getParser()
    args = parser.parse_args()

    fn = createFilenameTxt(args.dir)
    splitIntoTrainValidationTxt(fn, args.ratio)


import json, os
from rgba import mergeRGBA, checkBounds
from utils import newFilename, makeOutputFolder
import rasterio



def toJson(filename, jsonFilename):
    with open(filename, 'r') as f:
        gsData = f.read()
        gsData = gsData[:-2]    #remove newLine and last comma
    with open(jsonFilename, 'w') as f:
        f.write("{ \n" + gsData + "\n }")

if __name__ == '__main__':
    outputDir = makeOutputFolder('separate')
    rgbDir = 'data\\rgb'
    gsDir = 'data\\gs'
    #toJson('logs\\europeElevationLogs\\name_to_coordinate_dict.log', 'logs\\\europeElevationLogs\\gs.json')
    #toJson('logs\\rgba\\name_to_coordinate_dict.log','logs\\rgb.json')
    with open('logs\\rgb.json', 'r') as f:
        rgbDict = json.load(f)
        # print(json.dumps(gsDict, indent=4))

        # unique_values = [k for k,v in gsDict.items() if list(gsDict.values()).count(v)>1]
        # print(unique_values)
        # print([v for k, v in gsDict.items()])
    
    with open('logs\\\europeElevationLogs\\gs.json', 'r') as f:
        gsDict = json.load(f)

    skipNext = False
    gsIter = iter(gsDict.items())
    # gsIter =iter(sorted(gsDict.items()))
    for k, v in rgbDict.items():
        if not skipNext:
            currentGsK, currentGsV  = next(gsIter)
        else:
            skipNext = False
        while v != currentGsV: #TODO set up a tolerance
            if ((currentGsV[0] > v[0] and currentGsV[1] >= v[1]) or (currentGsV[1] > v[1])):
                print("didn't find pair for image", k)
                skipNext = True
                break
            currentGsK, currentGsV  = next(gsIter)
        
        if v == currentGsV:
            if not checkBounds(os.path.join(rgbDir,k+".tif"), os.path.join(gsDir, currentGsK+".tif")):
                print("Bounds DIFFERENT")
            newFileName = newFilename(k+"_"+currentGsK, suffix=".png",outdir=outputDir)
            # print(os.path.join(rgbDir,currentGsK+".tif"))
            mergeRGBA(os.path.join(rgbDir,k+".tif"), os.path.join(gsDir, currentGsK+".tif"), out=newFileName)
    # print(grayscaleCorrespont)

    # with open('logs\\name_to_coordinate_dict.log', 'r') as f:
    #     append and prepend {} respectively
    #     rgbDict = json.dumps(f)
    #     rgbDictCorrespont = dict((v, k) for k, v in rgbDict.items())

    # for k, v in rgbDictCorrespont:
    #     if grayscaleCorrespont[k] not None:

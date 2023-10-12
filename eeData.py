import numpy as np
import ee
import time

import logging, json
import os
from decimal import *
 

def setup_logger(name, log_file, level=logging.INFO, format='default'):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)   
    if format == 'dict':
        formatter = logging.Formatter('\"%(number)s\": %(message)s,')
    else:
        formatter = logging.Formatter('%(message)s')
    formatter = handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

class CoordLogger():    
    def __init__(self, filename='img_coordinates.log'): #filename='img_coordinates.json'):
        if os.path.exists(filename):
            #read last entry.
            with open(filename,"rb") as f:
                try:  # catch OSError in case of a one line file 
                    f.seek(-2, os.SEEK_END)
                    while f.read(1) != b'\n':
                        f.seek(-2, os.SEEK_CUR)
                except OSError:
                    f.seek(0)
                last_line = f.readline().decode()
                lastEntry = int(last_line.split(":")[0].replace("\"",''))
                self.entry = lastEntry + 1
        else:
            self.entry = 1
            
        self.logger = setup_logger('coordinateLogger', filename,format='dict')

        # JSON file 
        # if os.path.exists(filename):
        #     #read last entry.
        #     with open('img_coordinates.json',"r") as f:
        #         self.data = json.load(f)
            
        #     lastEntry = int(list(self.data)[-1])
        #     self.file = open('img_coordinates.json',"w")
        #     self.entry = lastEntry + 1
        # else:
        #     self.file = open('img_coordinates.json',"w")
        #     self.entry = 1
        #     self.data = {}

    def log(self, dictEntry):
        self.logger.info(dictEntry,extra={'number': self.entry})
        self.entry += 1

        # self.data[self.entry] = dictEntry
        # json.dump(self.data, self.file, indent=2)


# first file logger
logger = setup_logger('infoLogger', 'logs\\exported.log')
logger.info('New Start: ' + str(time.strftime("%Y-%m-%d-%H-%M")))

# second file logger
errorLogger = setup_logger('errorLogger', 'logs\\errors.log', logging.ERROR)
errorLogger.error('New Start: ' + str(time.strftime("%Y-%m-%d-%H-%M")))

#
processedLogger = setup_logger('processedLogger', 'logs\\processed.log')
processedLogger.info('New Start: ' + str(time.strftime("%Y-%m-%d-%H-%M")))


coordinateLogger = CoordLogger('logs\\name_to_coordinate_dict.log')





#ee.Authenticate()
ee.Initialize()


#30m nasa srtm
srtm = ee.Image('USGS/SRTMGL1_003')

#landsat 8 SR corrected
landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")


# Location is bottom-left point of image, [long, lat]
# imageSize is pixel size of result image
def getRegionCoordinates(imageSize, location):
    getcontext().prec = 6 #afraid to remove this   
    # 4503 5996 2737 0495 max fractional number in normal float percision 64bit

    # 1/3600 == 1 arc-sec == 30m (30.87)  == 1px (SRTM) 
    # appears that 0.04 = 16px
    # according to EE SRTM native scale is 30.922080775909325m
    arcsecond  = 1/3600     
    pixeloffset = arcsecond * imageSize

    coords = [location, \
                    [location[0],location[1]+ pixeloffset] , \
                    [location[0] + pixeloffset, location[1] + pixeloffset], \
                    [location[0]+ pixeloffset, location[1]]]
 
    for c in coords: 
        print(c)
    
    return coords




    
def eeExport(image,description,scale,fileName,region=None):
    if region != None:
        task = ee.batch.Export.image.toDrive(image=image, 
                                            description=description,
                                            scale = scale,
                                            folder = 'ElevationSigned',
                                            #dimensions= dimensionStr,
                                            fileNamePrefix=fileName,
                                            crs='EPSG:4326',
                                            region=region,
                                            fileFormat='GeoTIFF')
    else:
        task = ee.batch.Export.image.toDrive(image=image, 
                                    description=description,
                                    scale = scale,
                                    folder = 'ElevationSigned',
                                    #dimensions= dimensionStr,
                                    fileNamePrefix=fileName,
                                    crs='EPSG:4326',
                                    fileFormat='GeoTIFF')
    task.start()
    return task







def getImagePair(coords, outputRes):
    fileName = str(coordinateLogger.entry)
    #fileName = "rgba_"+ fileNameSuffix
    coordinateLogger.log(coords[0])

    region = ee.Geometry.Polygon(coords)
    elevation_roi = srtm.clip(region)
    elevation_16bit = elevation_roi.toInt16() # SRTM range is from -10 to 6500, signed 16bit is -32,768 to +32,767
    #elevation = elevation_16bit.clamp(-10,6500)

    
    processedLogger.info(str(coords[0]))

    if (not elevationCheck(elevation_16bit, region, outputRes)):
        return None

    sat = landsat.filter(ee.Filter.lessThan('CLOUD_COVER',5))
    sat_median = sat.median().clip(region)
    # threeBandImg = sat_median.select(['SR_B4', 'SR_B3', 'SR_B2'])
    # threeBandImg.toUint16() # LSAT range is from 1 to 65455, 16bit is 0 to 65535


    if(not satCheck(sat_median,region,outputRes)):
        return None
    
    
    logger.info(str(coords[0]))  
    
    pixelSpace_grayscale = elevation_16bit.visualize(bands=['elevation'], min=-10 , max=6500)
    pixelSpace_img = sat_median.visualize(bands=['SR_B4', 'SR_B3', 'SR_B2'], min= 7219, max= 14355)  #max= 14355, gamma=1.2
    
    rgba = pixelSpace_img.addBands(pixelSpace_grayscale)
    #return eeExport(pixelSpace_img, satFileName, 30.922080775909325, satFileName, region)
    return eeExport(elevation_16bit, fileName, 30.922080775909325, fileName, region)





def getImage(coords, datasetName, outputRes):
    fileName = str(coordinateLogger.entry)

    coordinateLogger.log(coords[0])

    region = ee.Geometry.Polygon(coords)
    if(datasetName is 'SRTM'):
        elevation_roi = srtm.clip(region)
        elevation_16bit = elevation_roi.toInt16() # SRTM range is from -10 to 6500, signed 16bit is -32,768 to +32,767
        #elevation_clampled = elevation_16bit.clamp(-10,6500)
        #pixelSpace_grayscale = elevation.visualize(bands=['elevation'], min=-10 , max=6500)
        if (elevationCheck(elevation_16bit,region,outputRes)):
            fileName = "el_"+ fileName
            processedLogger.info(str(coords[0]))
            return eeExport(elevation_16bit, fileName, 30.922080775909325, fileName)
        else:
            return None

    elif (datasetName is 'Landsat'):
        img = landsat.filter(ee.Filter.lessThan('CLOUD_COVER',5))
        img_median = img.median() #Best visual results
        
        if (satCheck(img_median,region,outputRes)):
            pixelSpace_img = img_median.visualize(bands=['SR_B4', 'SR_B3', 'SR_B2'], min= 7219, max= 14355) #Best visual results #, gamma=1.2
            print(pixelSpace_img.getInfo())
            fileName = "sat_"+ fileName
            return eeExport(pixelSpace_img, fileName, 30.922080775909325, fileName, region)
        else: 
            return None




elevationAreaCoverage = 1 # 0.8f -> at least 80% pixels not null 
minStdVar = 100 
def elevationCheck(eeImage, region, imgSize, stdCheck=False):
    # null pixel checks
    pixelCountDict = eeImage.reduceRegion(reducer=ee.Reducer.count(), geometry=region)
    pixelCount = pixelCountDict.getInfo()['elevation']      #this is how to access the result
    if (pixelCount/(imgSize*imgSize) < elevationAreaCoverage):
        print("Elevation Check (null-pixels): FAILED")
        return False
    
    # standard dev check
    if(stdCheck):
        stdDevDict = eeImage.reduceRegion(reducer=ee.Reducer.stdDev(), geometry=region)
        stdDev = stdDevDict.getInfo()['elevation']    
        #print(stdDev)  
        if (stdDev < minStdVar):
            print("Elevation Check (stdDev): FAILED")
            return False
    
    print("Elevation Check: PASSED")
    return True


satAreaCoverage = 1.0 #all pixels not null
def satCheck(eeImage, region, imgSize):
    # null pixel checks
    pixelCountDict = eeImage.reduceRegion(reducer=ee.Reducer.count(), geometry=region, scale=30.922080775909325)
    for key, value in pixelCountDict.getInfo().items():
        #pixelCount = pixelCountDict.getInfo()[key]  
        if (value/(imgSize*imgSize) < satAreaCoverage): #TODO,error epsilon
            print("Satellite Check: FAILED")
            return False

    print("Satellite Check: PASSED")
    return True








def startTask(coordinates,size, dataFunction):
    task = dataFunction(coordinates, size)
    if task is not None:
        while(str(task.status()['state']) != 'FAILED' and str(task.status()['state']) != 'COMPLETED' 
                and str(task.status()['state']) != 'CANCELLED'):
            print(task.status()['state'])
            time.sleep(5)
        print(task.status()['state'])

    return True



#going from bottom-left to top-right by row
# For whole world: The SRTM data is somewhere between 60° north and 56° south latitude so:
    # minLat = -56
    # maxLat = 60
    # minLong = -180
    # maxLong = 180
def getDataset(imgSize,dataFunction, startCoord=[-180,-56]):
    #Europe and some xtra (BBox)
    minLat = 12
    maxLat = 60
    minLong = -9
    maxLong = 60
    
    currentCoord = startCoord 
    a = 0
    while True:
        try:
            coords = getRegionCoordinates(imgSize, currentCoord)      
        except Exception as e:
            errorLogger.error("Exception Occured Serve-side (coords), with coord: " + str(currentCoord))
            print("SERVER FAIL: Logging coordenates. Continuing...")
            print(e)
            pass

        if(coords[1][1] > maxLat):
            print("REACHED OUT OF BOUNDS LATITUDE. STOPPING.")
            break
        elif(coords[2][0] > maxLong):
            currentCoord = [minLong,coords[2][1]] #reset long, next lat, i.e going to next column in same row
        else:
            currentCoord = coords[3] #next long, same lat, i.e going to next column in same row

        try:
            startTask(coords, dataFunction,imgSize)          
        except Exception as e:
            errorLogger.error("Exception Occured Serve-side (image), with coord: " + str(coords[0]))
            print("SERVER FAIL: Logging coordenates. Continuing...")
            print(e)
            pass

        print("Server OK")
        print("-----------------------------------------------")
        print()
        a += 1
        time.sleep(1)   




if __name__ == '__main__':
    imgSize = 1024
    startCoord = [-9,12]

    getDataset(imgSize,getImage,startCoord)
    #t = ee.data.getOperation(taskId)
    #print(t['metadata']['state'])
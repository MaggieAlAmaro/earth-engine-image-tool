import rasterio
from rasterio.plot import show
import numpy as np
import ee
import time

import logging

from decimal import *
 
#region Logger

'''

#logging.basicConfig(filename='logs.log', level=logging.WARNING, format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = loggind.Formatter('s%(levelname)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('export.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
'''

formatter = logging.Formatter('%(levelname)s-%(message)s')


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# first file logger
logger = setup_logger('infoLogger', 'completed.log')
logger.info('New Info Session')

# second file logger
errorLogger = setup_logger('errorLogger', 'errors.log', logging.ERROR)
errorLogger.error('New Error Session')


#endregion



#ee.Authenticate()
ee.Initialize()


#30m nasa srtm
srtm = ee.Image('USGS/SRTMGL1_003')

#landsat 8 SR corrected
landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")

xy = ee.Geometry.Point([86.9250, 27.9881])
xy2 = ee.Geometry.Point([-37.33, 28.44])


#'palette': ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']

#url = elevation.updateMask(elevation.gt(0)).getThumbURL({'min': 0, 'max': 4000, 'scale': 30,
#               'palette': ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']})


    



# Location is bottom-left point of image, [long, lat]
# imageSize is pixel size of result image
def getRegionCoordinates(imageSize, location):#, dataResolutionArcSec):
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
                                            folder = 'RGBA',
                                            #dimensions= dimensionStr,
                                            fileNamePrefix=fileName,
                                            crs='EPSG:4326',
                                            region=region,
                                            fileFormat='GeoTIFF')
    else:
        task = ee.batch.Export.image.toDrive(image=image, 
                                    description=description,
                                    scale = scale,
                                    folder = 'RGBA',
                                    #dimensions= dimensionStr,
                                    fileNamePrefix=fileName,
                                    crs='EPSG:4326',
                                    fileFormat='GeoTIFF')
    task.start()
    return task





def getImage(initialCoordenates, datasetName, outputRes):
    coords = getRegionCoordinates(outputRes, initialCoordenates)
    return getImageByCoords(coords, datasetName,outputRes)



def getImagePair(coords, outputRes):
    region = ee.Geometry.Polygon(coords)
    fileNameSuffix = str(coords[0][0])+"_"+str(coords[0][1])+"__"+str(outputRes+1) # +1 because resulting size is chunkSize + 1 for some reason
    #elFileName =  "el_"+ fileNameSuffix
    #satFileName = "sat_"+ fileNameSuffix
    fileName = "rgba_"+ fileNameSuffix
    elevation = srtm.clip(region)
    if (not elevationCheck(elevation,region,outputRes+1)):
        return None

    img = landsat.filter(ee.Filter.lessThan('CLOUD_COVER',5))
    img_median= img.median()
    threeBandImg = img_median.select(['SR_B4', 'SR_B3', 'SR_B2'])
    if(not satCheck(threeBandImg,region,outputRes+1)):
        return None
    
    
    logger.info("PASSED: " + str(coords[0]))  
    
    pixelSpace_grayscale = elevation.visualize(bands=['elevation'], min=-10 , max=6500)
    #eeExport(pixelSpace_grayscale, elFileName, 30.922080775909325, elFileName)

    pixelSpace_img = img_median.visualize(bands=['SR_B4', 'SR_B3', 'SR_B2'], min= 7219, max= 14355)  #max= 14355, gamma=1.2
    
    rgba = pixelSpace_img.addBands(pixelSpace_grayscale)
    #print(pixelSpace_img.getInfo())
    #return eeExport(pixelSpace_img, satFileName, 30.922080775909325, satFileName, region)
    return eeExport(rgba, fileName, 30.922080775909325, fileName, region)





def getImageByCoords(coords, datasetName, outputRes):
    region = ee.Geometry.Polygon(coords)
    fileNameSuffix = str(coords[0][0])+"_"+str(coords[0][1])+"__"+str(outputRes+1) # +1 because resulting size is chunkSize + 1 for some reason
   
    if(datasetName is 'SRTM'):
        elevation = srtm.clip(region)
        if (elevationCheck(elevation,region,outputRes+1)):
            pixelSpace_grayscale = elevation.visualize(bands=['elevation'], min=-10 , max=6500)
            fileName = "el_"+ fileNameSuffix
            return eeExport(pixelSpace_grayscale, fileName, 30.922080775909325, fileName)
        else:
            return None

    elif (datasetName is 'Landsat'):
        img = landsat.filter(ee.Filter.lessThan('CLOUD_COVER',5))
        img_median= img.median()
        pixelSpace_img = img_median.visualize(bands=['SR_B4', 'SR_B3', 'SR_B2'], min= 7219, max= 14355)  #, gamma=1.2
        print(pixelSpace_img.getInfo())
        fileName = "sat_"+ fileNameSuffix
        return eeExport(pixelSpace_img, fileName, 30.922080775909325, fileName, region)




elevationAreaCoverage = 0.8 #at least 80% pixels not null 
minStdVar = 100 #values deviate
def elevationCheck(eeImage, region, imgSize):
    # null pixel checks
    pixelCountDict = eeImage.reduceRegion(reducer=ee.Reducer.count(), geometry=region)
    pixelCount = pixelCountDict.getInfo()['elevation']      #this is how to access the result
    if (pixelCount/(imgSize*imgSize) < elevationAreaCoverage):
        print("Elevation Check (null-pixels): FAILED")
        return False
    
    # standard dev check
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
        #print(value)
        if (value/(imgSize*imgSize) < elevationAreaCoverage):
            print("Satellite Check: FAILED")
            return False

    print("Satellite Check: PASSED")
    return True








def getRGBAImage(coordinates,size):
    task = getImagePair(coordinates, size)
    if task is not None:
        while(str(task.status()['state']) != 'FAILED' and str(task.status()['state']) != 'COMPLETED' 
                and str(task.status()['state']) != 'CANCELLED'):
            print(task.status()['state'])
            time.sleep(5)
        print(task.status()['state'])

    return True



#going from bottom-left to top-right by row
def getRGBADataset(imgSize, startCoord=[-180,-56]):
    #SRTM between 60° north and 56° south latitude
    minLat = -56
    maxLat = 60
    minLong = -180
    maxLong = 180
    
    currentCoord = startCoord 
    a = 0
    while True:
        try:
            coords = getRegionCoordinates(imgSize, currentCoord)
        except:
            errorLogger.error("Exception Occured Serve-side (coords), with coord: " + str(currentCoord))
            print("SERVER FAIL: Logging coordenates")
            print(e)
            break

        if(coords[1][1] > maxLat):
            print("REACHED OUT OF BOUNDS LATITUDE. STOPPING.")
            break
        elif(coords[2][0] > maxLong):
            currentCoord = [minLong,coords[2][1]] #reset long, next lat, i.e going to next column in same row
        else:
            currentCoord = coords[3] #next long, same lat, i.e going to next column in same row

        try:
            getRGBAImage(coords, imgSize)          
        except Exception as e:
            errorLogger.error("Exception Occured Serve-side (image), with coord: " + str(coords[0]))
            print("SERVER FAIL: Logging coordenates")
            print(e)
            break

        print("Server OK")
        print("-----------------------------------------------")
        print()
        a += 1
        time.sleep(1)   



if __name__ == '__main__':
    imgSize = 1024
    startCoord = [37.03111111110932, 10.524444444444445]
    
    #[151.9466666666644, 0.5688888888888889]
    
    #[-59.395555555557365, 0.5688888888888889] #[34.75555555555375, 0.28444444444444444]
    getRGBADataset(imgSize,startCoord)
    #print(getRegionCoordinates(1025,[89.7338, 27.9521]))
    
    #task = getImage([89.7338, 27.9521], 'SRTM', 1024)
    #task = getImage([-6.711, 40.444], 'SRTM', 1024)
    #t = ee.data.getOperation(taskId)
    #print(t['metadata']['state'])



    
    ##START AT BOTTOM LEFT
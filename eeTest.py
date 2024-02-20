import numpy as np
import ee
from itertools import product
from decimal import *
import time
from eeData import * #getRegionCoordinates,getRegionCoordinatesWithScale

import cv2
from PIL import Image

from src.processors.two_data_processor import Compare 

from matplotlib.colors import ListedColormap, BoundaryNorm

from src.processors.one_channel_processor import MedianDenoise
from src.utils import makeOutputFolder, newFilename

def drawWithColor(imageArr):
    colorArray = np.array([
            [11, 20, 20, 20],
            [12, 56,56,56],
            [13, 128,128,128],
            [14, 235, 235, 143],
            [15, 247, 211, 17],
            [21, 170, 0, 0],
            [22, 216, 147, 130],
            [23, 221, 201, 201],
            [24,220, 205, 206],
            [31, 28, 99, 48],
            [32, 104, 170, 99],
            [33, 181, 201, 142],
            [34,225, 240, 229],
            [41,111, 25, 140],
            [42, 169, 117, 186],
            ])

    # print(np.unique(result))

    u, ind = np.unique(imageArr, return_inverse=True)
    b = ind.reshape((imageArr.shape))
    print('b',b )


    colors = colorArray[colorArray[:,0].argsort()][:,1:]/255.
    print('col', colors)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(colorArray)+1)-0.5, len(colorArray))

    # plt.imshow(b,cmap=cmap, norm=norm)
    # plt.tight_layout()
    # plt.show()
    

#ee.Authenticate()
ee.Initialize()

#30m nasa srtm
dataset = ee.Image('CSP/ERGo/1_0/Global/SRTM_landforms')
copernicus = ee.ImageCollection("COPERNICUS/DEM/GLO30")


srtm = ee.Image('USGS/SRTMGL1_003')
import matplotlib.pyplot as plt

landforms = dataset.select('constant')
water = copernicus.select('WBM')
mosaicWater = water.mosaic()
#[0.0008084837557075694, 0, -180.00001488697754, 0, -0.0008084837557075694, 70.00014053667277]
da = srtm.projection()
# dem_mean_reproj = water.reproject(da)
mosaicWater = mosaicWater.reproject(da) 
landforms = landforms.reproject(da) 
# print(landforms.getInfo())
# print(mosaicWater.getInfo())
# geo = ee.Geometry.Polygon([[-6.155555555555555, 37.031111111111045], [-6.155555555555555, 37.31555555555549], [-5.8711111111111105, 37.31555555555549], [-5.8711111111111105, 37.031111111111045]])
geo = ee.Geometry.Polygon([[-6.055555555555555, 37.031111111111045], [-6.055555555555555, 37.11555555555549], [-5.9711111111111105, 37.11555555555549], [-5.9711111111111105, 37.031111111111045]])
gedo = ee.Geometry.Point([-6.055555555555555, 37.031111111111045])


def getRectagleArray(coords, pixelCap):
    roi = ee.Geometry.Polygon(coords)
    col = landforms.sampleRectangle(region=roi)
    arr = col.get('constant')
    arr = np.array(arr.getInfo())
    print(arr.shape)
    return arr[:pixelCap,:pixelCap]

def getRectagleArrayWater(coords, pixelCap):
    roi = ee.Geometry.Polygon(coords)
    # boundedWater = water.filterBounds(roi)
    # image = boundedWater.first()
    col = mosaicWater.sampleRectangle(region=roi,properties=['WBM'])
    # print(col.getInfo())
    arr = col.get('WBM')
    arr = np.array(arr.getInfo())
    # print(arr)
    # print(arr.shape)
    return arr[:pixelCap,:pixelCap]


#!!!! pixelCap may need to be adjusted in case image is returning black Lines
def get256ImageArray(coords, bandName, datasetImage, pixelCap=256):
    roi = ee.Geometry.Polygon(coords)
    col = datasetImage.sampleRectangle(region=roi,properties=[bandName])
    arr = col.get(bandName)
    arr = np.array(arr.getInfo())
    return arr[:pixelCap,:pixelCap]



def getImageFromStartCoordenates(startCoords, bandName,datasetImage):
    regionBounds =  getRegionCoordinates(256,startCoords)
    return get256ImageArray(regionBounds,bandName,datasetImage,256)



def getRectangleImage(startCoords, datasetImage, bandName, size=[1024,1024]):
    start = time.time()
    assert size[0] % 256 == 0 and size[1] % 256 == 0, f" Size of image must be divisable by 256"
    rows = int(size[0]/256)
    cols = int(size[1]/256)
    imageCols = []
    startCoordsForRows = []
    nextCoords = startCoords
    for i in range(cols):
        # before using crs reproject I had: 
        # getRegionCoordinatesWithScale(256,startCoords,30) for dataset with 90m resolution
        regionBounds = getRegionCoordinates(256,nextCoords)
        nextCoords = regionBounds[3]
        imageCols.append(get256ImageArray(regionBounds,bandName,datasetImage,256))
        startCoordsForRows.append(regionBounds[1])

        
    for _ in range(1, rows):
        for j in range(cols):
            regionBounds = getRegionCoordinates(256, startCoordsForRows[j])
            temp = get256ImageArray(regionBounds,bandName,datasetImage,256)
            imageCols[j] = np.r_['0', temp, imageCols[j]]
            startCoordsForRows[j] = regionBounds[1]

    fullImage = np.concatenate([cols for cols in imageCols], axis=1)
    if fullImage.shape != (1024, 1024):
        print(f"Returned incorrect shape!!! Shape is {fullImage.shape}. RESIZING!!!")
        image = cv2.resize(fullImage, dsize=(1024, 1024), 
                           interpolation=cv2.INTER_NEAREST)
    int8Image = np.uint8(fullImage)

    end = time.time()
    totTime = end - start
    print(f"Total time to get {size} image was {totTime}")
    return int8Image

import math
def segmentationFullSpectrum(image):
    numberOfClasses = 6
    nbr = math.floor(255/(numberOfClasses - 1)) # -1 because classes start at 0

    #assign class by order of importance/height
    image[image == 41] = 0 #valley 
    image[image == 42] = 0 #valley narrow
    image[image == 24] = 1 #flat
    image[image == 34 ] = 1 #flat
    image[image == 31] = 2 #lower Slope
    image[image == 32] = 2 #lower Slope
    image[image == 33] = 2 #lower Slope
    image[image == 21] = 3 #upper Slope
    image[image == 22] = 3 #upper Slope
    image[image == 23] = 3 #upper Slope
    image[image == 11] = 4 #mountain/peaks
    image[image == 12] = 4 #mountain/peaks
    image[image == 13] = 4 #mountain/peaks
    image[image == 14] = 4 #mountain/peaks
    image[image == 15] = 5 #cliff

    image = image * nbr
    return image

def waterFullSpectrum(image):
    numberOfClasses = 4  
    nbr = math.floor(255/(numberOfClasses - 1)) # -1 because classes start at 0
    image = image * nbr
    return image
    
def denoiseAndSave(imageArr,kernelSize,outdir='.'):
    denoised = MedianDenoise.denoise(imageArr,kernelSize)
    denoisedImg = Image.fromarray(denoised)   
    fn = newFilename('denoised-before-9',outdir=outdir)
    denoisedImg.save(fn)
    return denoised

def GetSegmentedImage(startCoords):
    sup = 3
    
    # fullImg = Image.fromarray(image)   
    # fullImg = fullImg.convert('L')
    # output = makeOutputFolder('segmentation')
    # fn = newFilename('img-before',outdir=output)
    # # fullImg.save(fn)

    #fullspectrumRemap

    # fn = newFilename('img-after',outdir=output)
    # fullImg = Image.fromarray(image)   
    # fullImg = fullImg.convert('L')
    # fullImg.save(fn)

    #denoiseAndSave(image,9)

    # fn = newFilename('denoised-after',outdir=output)
    # denoisedImg = Image.fromarray(denoised)  
    # denoisedImg.save(fn)

    #denoiseAndSave(sceptrumRevampImage,9)
    #denoiseAndSave(sceptrumRevampImage,5)


    np.set_printoptions(threshold = np.inf)

    

# With ee things
# samples = dataset.sample(region=geo)
# print(samples.propertyNames().getInfo())
# sampleNbr = samples.size().getInfo()
# m_list = samples.toList(samples.size())
# ncols = math.ceil(math.sqrt(sampleNbr))
# nrows = math.ceil(sampleNbr/ncols)
# ee.Number(samples.get('band_order')).getInfo()
# a = np.zeros((ncols,nrows))
# # or i, j in product(range(0, 172801, width), range(0, 68683, height)):
# zippedData = product(range(ncols),range(nrows))
# for i, j in zippedData:
#     print(i)
#     print(j)
#     if i + j*ncols >= sampleNbr:
#        break
#     e = m_list.get(i + j*ncols).getInfo()['properties']['constant']
#     a[i][j] = e
# print(a)
# plt.imshow(a)
# plt.tight_layout()
# plt.show()


# colors = ListedColormap(['141414', '383838', '808080', 'ebeb8f', 'f7d311', 'aa0000', 'd89382',
#     'ddc9c9', 'dccdce', '1c6330', '68aa63', 'b5c98e', 'e1f0e5', 'a975ba',
#     '6f198c'])




# def samp_arr_img(arr_img):
#   point = ee.Geometry.Point([-21, 0])
#   return arr_img.sample(point).first().get('array')

# # Create a 2D 3x3 array image.
# array_img_2d = ee.Image([1, 2, 3, 4, 5, 6, 7, 8, 9]).toArray().arrayReshape(
#     ee.Image([3, 3]).toArray(),
#     2)
# print('2D 3x3 array image (pixel):', samp_arr_img(array_img_2d).getInfo())
# # [[1, 2, 3],
# #  [4, 5, 6],
# #  [7, 8, 9]]



def Experiments(startCoords):


    # imageData = getImage(startCoords,mosaicWater,'WBM')
    imageData = getRectangleImage(startCoords, landforms, 'constant')
    imageData = segmentationFullSpectrum(imageData)

    output = makeOutputFolder('denoiseTests')
    # a = denoiseAndSave(imageData,5,output)
    # a = denoiseAndSave(imageData,11,output)
    # plt.imshow(imageData)
    # plt.tight_layout()
    # plt.show()
    img = Image.fromarray(imageData)   
    img = img.convert('L')
    fn = newFilename('img',outdir=output)
    img.save(fn)

      # f, axarr = plt.subplots(3,3)
    # axarr[0,0].imshow(imageData)
    # axarr[0,1].imshow(a)
    # axarr[1,1].imshow(a)
    # # plt.imshow(np.array(image))
    # plt.tight_layout()
    # plt.show()  



def openSeg(filename):
    
    denoisedImg = Image.open(filename) 
    return np.array(denoisedImg) 

    # ah = MedianDenoise.denoise(arr,7)
    # n = Image.fromarray(ah) 
    # output = makeOutputFolder('test')
    # fn = newFilename('another',outdir=output)
    # n.save(fn)

if __name__ == '__main__':
    imgSize = 1024
    coords = [-6.155555555555555, 37.031111111111045], [-6.155555555555555, 37.31555555555549], 
    [-5.8711111111111105, 37.31555555555549], [-5.8711111111111105, 37.031111111111045]
    startCoords = [0.6678, 42.9273] # [-6.155555555555555, 37.031111111111045]
    
    #arr = openSeg("output\segmentation-2024-02-12-16-10-54\img.png")
    # "output/segmentation-2024-02-14-00-12-21/denoised-from-typeFix-9.png"
    Experiments(startCoords)
    #hey = GetSegmentedImage(startCoords)

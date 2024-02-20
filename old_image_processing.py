import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.merge import merge
import numpy as np
import argparse
from PIL import Image
from itertools import product
import os, sys, math

plt.style.use('dark_background')

def changeBitDepth(image):
    img = rasterio.open(image)
    print(img.dtypes)
    print(img.bounds)
    data = img.read()
    min= -10
    max= 6500
    normalizedData = (data - min)/(max-min)
    rescaledData = normalizedData * (255 - 0)
    
    rescaledData = rescaledData.astype('uint8')
    newImg = Image.fromarray(rescaledData[0])
    print(rescaledData.shape)
    print(rescaledData)
    newImg.save("srtm16bit-8bit--1000.png")
    

def treatNumpy(image: str):
    img = rasterio.open(image)
    imageData = img.read()
    min= -10
    max= 6500
    newMin = np.cbrt(min)
    newMax = np.cbrt(max)

    bitDepth = 255


    # range scaling explained: https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
    #normalize funct: newX = x - min/ max - min
    #newval = np.sqrt(imageData)
    normalizedData = (np.cbrt(imageData) - newMin)/(newMax-newMin)
    rescaledData = normalizedData * (bitDepth - 0)
    rescaledData = rescaledData.astype('uint8')
    newImg = Image.fromarray(rescaledData[0])
    newImg.save("srtm16bit-8bit-cqrt-10.png")




# given that the SRTM image has a range estimate (according to earth engine) of:
# [-10, 6500], which I will take as [0,6500] just to make it easier (no sqrt of negative nbrs) -- after try with cube root
# TODO: Use ee.reducer.max() and .min() to find the actual value limits of the SRTM image.
# SOLUTIONS: because it cant calculate the max over the whole image, do max/min convolution over it until getting true values
def treat(x):
    min= 0
    max= 6500
    newMin = math.sqrt(min)
    newMax = math.sqrt(max)



    # range scaling explained: https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
    #normalize funct: newX = x - min/ max - min
    newval = math.sqrt(x)
    newX = (newval - newMin)/(newMax-newMin)


def cropToSize(imageDir, clipSize):
    if(os.path.isdir(imageDir)):
        for filename in os.listdir(imageDir):
            f = os.path.join(imageDir, filename)
            # check if file
            if os.path.isfile(f):
                if(f.split(".")[-1] == 'png'):
                    img = Image.open(f)
                    if(img.size[0] > clipSize or img.size[1] > clipSize):
                        y0Clip = img.size[0] - clipSize  # clip excess top rows
                        #x1Clip = img.size[1] # clip excess right columns
                        croppedImg = img.crop((0, y0Clip, clipSize, img.size[0]))
                        newFileName = "1024" + f.split(os.sep)[-1]
                        out = os.path.join("cropped", newFileName)
                        croppedImg.save(out)

                    elif(img.size[0] < clipSize or img.size[1] < clipSize):
                        print("IMAGE SIZE IS LOWER THAN CLIP SIZE.")
                        print(f)
                        return

def tile(imageDir, tileSize):
    if(os.path.isdir(imageDir)):
        for filename in os.listdir(imageDir):
            f = os.path.join(imageDir, filename)
            # check if file
            if os.path.isfile(f):
                if(f.split(".")[-1] == 'png'):
                    img = Image.open(f)
                    w, h = img.size
                    if(h%tileSize != 0 or w%tileSize != 0):
                        print("Tile size must be devisable by the size of the image.")
                        return
                    for i, j in product(range(0, h, tileSize), range(0, w, tileSize)):
                        newFileName = "256_"+ str(i) + "_" + str(j) +"_" + f.split(os.sep)[-1]
                        out = os.path.join("256", newFileName)
                        box = (j, i, j+tileSize, i+tileSize)
                        img.crop(box).save(out)


def sizeCheck(imageDir, expectedSize):
    if(os.path.isdir(imageDir)):
        for filename in os.listdir(imageDir):
            f = os.path.join(imageDir, filename)
            # check if file
            if os.path.isfile(f):
                if(f.split(".")[-1] == 'png'):
                    img = Image.open(f)
                    if(img.size[0] < 1024 or img.size[1] < 1024  ):
                        print(img.size, img)
                elif(f.split(".")[-1] == 'tif' or f.split(".")[-1] == 'tiff'):
                    with rasterio.open(f) as img:
                        bandNbr = [band for band in range(1,img.count + 1)]
                        data = img.read(bandNbr)
                        print(data.shape == expectedSize)
                

def seperateAandRGB(image, rgbDir, aDir):
    f_type = image.split(".")[-1]
    newFileName = image.split(os.sep)[-1] 
    newRGBFileName = newFileName.rstrip(("."+f_type)) + "RGB.png"
    newAFileName = newFileName.rstrip(("."+f_type)) + "A.png"
    img = Image.open(image)
    r, g, b, a = img.split()

    a = a.convert('L')
    rgb = Image.merge('RGB', (r, g, b))
    
    if not os.path.isdir(rgbDir):
        os.mkdir(rgbDir)
    
    if not os.path.isdir(aDir):
        os.mkdir(aDir)

    rgb.save(rgbDir+os.sep+newAFileName)
    a.save(aDir+os.sep+newRGBFileName)
    print("Sucessfully seperated" + str(image))


def satElevationRGBAMerge(fileRGB, fileA):
    img = Image.open(fileRGB)
    a = Image.open(fileA)
    a = a.convert('L')
    
    a.save('a.png')
    img.save('rgb.png')

    r, g, b = img.split()
    result = Image.merge('RGBA', (r, g, b, a))
    #img = Image.new('RGBA',(1025,1025))
    #apperently saving as png is enough to convert from tif
    result.save('newImgAA.png')


def convertPNG(oldImage, newFileName=None, destination=None):
    if(newFileName is None):
        f_type = oldImage.split(".")[-1]
        newFileName = oldImage.split(os.sep)[-1]
        newFileName = newFileName.rstrip(("."+f_type)) + ".png"

    img = Image.open(oldImage)
    channels = img.getbands()
    print(img.mode)
    if(len(channels) == 1):
        print(img.size)
        img = img.convert(mode='L')
    elif(len(channels) == 3):
        img = img.convert(mode='RGB')
    elif(len(channels) == 4):
        img = img.convert(mode='RGBA')
    else: 
        print("Image format not recognized!")
        return None
    
    
    if(destination != None):
        if not os.path.exists(destination):
            os.mkdir(destination)
        newFileName = os.path.join(destination, newFileName)
    img.save(newFileName)
    print("New file created: " + newFileName)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    if (args.separate is not None):       
        # if dir
        if(os.path.isdir(args.separate)):
            for filename in os.listdir(args.separate):
                f = os.path.join(args.separate, filename)
                # check if file
                if os.path.isfile(f):
                    seperateAandRGB(f,"rgb","a")
                        
        # if file
        elif(os.path.isfile(args.separate)):
            if(args.separate.split(".")[-1] != 'png'):
                print("Only PNG is supported!")
            else:
                seperateAandRGB(args.separate,"rgb","a")
        else: 
            print("Use correct notation.")




    if (args.compare is not None):
        if(len(args.compare) == 2):
            if(args.compare[0].split(".")[-1] != 'png' or args.compare[1].split(".")[-1] != 'png'):
                print("Only PNG is supported!")
            else:
                img1 = Image.open(args.compare[0])
                img2 = Image.open(args.compare[1])
                img1 = np.array(img1)
                img2 = np.array(img2)
                if(img1.shape != img2.shape):
                    print("The image sizes don't match:" + str(img1.shape) + "and" + str(img2.shape))
                else:
                    print(img2.shape)
                    if((img1 == img2).all()):
                        print("The images are EQUAL.")
                    else:
                        print("The images are NOT equal.")
        else:
            print("Must specify only 2 files!")

    if (args.name is not None):
        if(args.name.split(".")[-1] == 'png'):
            img = Image.open(args.name)
            print(img.size)
            print(img.getpixel((30,200)))
            print(np.array(img))
            img.show()


#
        elif(args.name.split(".")[-1] == 'tif' or args.name.split(".")[-1] == 'tiff'):
            with rasterio.open(args.name) as img:
                bandNbr = [band for band in range(1,img.count + 1)]
                data = img.read(bandNbr)
                print(data.shape)
                #if grayscale
                if (len(bandNbr) == 1):
                    show(data,cmap='gray',vmin=0, vmax=255)
                    #show(data,cmap='gray') #if min-max are not set it will use the min and max values found in the array to set a pixel range... im assuming
                else:
                    show(data)
                    


    if(args.channelMerge is not None):
        if(len(args.channelMerge) == 2):
            satElevationRGBAMerge(args.channelMerge[0],args.channelMerge[1])
        else:
            print("Must specify only 2 files!")


    if(args.merge is not None):
        if(len(args.merge) == 2):
            print("nothere")
            file0 = rasterio.open(args.merge[0])
            file1 = rasterio.open(args.merge[1])    
            result, transform = merge([file0,file1])
            #show(file0)
            #show(file1)
            # found that the 257th pixel is the same as the first pixel of the next image, so the files merge on the 257th pixel
            #print(file0.read(1)[:,256]==file1.read(1)[:,0])
            plt.show(result)
            print(result.data.shape)
        else:
            print("Must specify only 2 files!")
             




    if(args.toType is not None):
        if(len(args.toType) == 2):            
            # if dir
            if(os.path.isdir(args.toType[0])):
                for filename in os.listdir(args.toType[0]):
                    f = os.path.join(args.toType[0], filename)
                    # check if file
                    if os.path.isfile(f):
                        convertPNG(str(f),destination=args.toType[1])
                        
            # if file
            elif(os.path.isfile(args.toType[0])):
                convertPNG(str(args.toType[0]),str(args.toType[1]))
            else: 
                print("Use correct notation.")
                print("Directory not found.")

        elif(os.path.isfile(args.toType[0])):
            convertPNG(str(args.toType[0]))
        else:
            print("Use correct notation.")

    else:
        #img = Image.open("srtm16bit.tif")
        #img = img.convert(mode='I;16')
        #img.save("test.png")
        
        #img = rasterio.open("srtm16bit.tif")#sys.argv[1])
        #data = img.read()
        newdata = changeBitDepth("bitest.tif")
        #print(np.min(newdata), newdata.dtype)
        #newImg = Image.fromarray(newdata[0])
        #newImg.save("theEnd.png")



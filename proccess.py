


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


def convertPNG(oldImage, newFileName=None, destination=None):
    if(newFileName is None):
        f_type = oldImage.split(".")[-1]
        newFileName = oldImage.split(os.sep)[-1]
        newFileName = newFileName.rstrip(("."+f_type)) + ".png"

    img = Image.open(oldImage)
    channels = img.getbands()
    if(len(channels) == 1):
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
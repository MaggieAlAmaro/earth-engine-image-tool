import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from PIL import Image
import argparse
from itertools import product
import os

#from utils import processDirOrFile

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs, formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")
    openParser = subparsers.add_parser("open", help="check if 2 images are the same")
    openParser.add_argument('name', type=str, help='Name of image or directory to merge with.')
    openParser.add_argument('--plot', type=bool, help='If true plots using matplotlib, if false opens with standard image')

    gridParser = subparsers.add_parser("grid", help="swhow image grid")
    gridParser.add_argument('--logdir', type=str, 
                            help="""Each folder in the logdir contains images to tile OR contains images.
                                    The size will be ImgNbrOfFirstFolder x nbrOfFolders
                            """)

    #return mergeParser, seperateParser
    return parser


def open(img, **kwargs):
    img = Image.open(img)

    print(np.array(img))
    img.show()
    print(img.size)
    
        # with rasterio.open(args.name) as img:
        #     bandNbr = [band for band in range(1,img.count + 1)]
        #     data = img.read(bandNbr)
        #     print(data.shape)
        #     #if grayscale
        #     if (len(bandNbr) == 1):
        #         show(data,cmap='gray',vmin=0, vmax=255)
        #         #show(data,cmap='gray') #if min-max are not set it will use the min and max values found in the array to set a pixel range... im assuming
        #     else:
        #         show(data)

#     if (args.name is not None):
#         if(args.name.split(".")[-1] == 'png'):
#             img = Image.open(args.name)
#             print(img.size)
#             print(img.getpixel((30,200)))
#             print(np.array(img))
#             img.show()


# #
#         elif(args.name.split(".")[-1] == 'tif' or args.name.split(".")[-1] == 'tiff'):
#             with rasterio.open(args.name) as img:
#                 bandNbr = [band for band in range(1,img.count + 1)]
#                 data = img.read(bandNbr)
#                 print(data.shape)
#                 #if grayscale
#                 if (len(bandNbr) == 1):
#                     show(data,cmap='gray',vmin=0, vmax=255)
#                     #show(data,cmap='gray') #if min-max are not set it will use the min and max values found in the array to set a pixel range... im assuming
#                 else:
#                     show(data)
                    



#'00073536\\2023-09-29-10-54-48-e11\\img\\'
def grid(logdir):
    imgs = []
    if(os.path.isdir(logdir)):
        items = os.listdir(logdir)
        if all(os.path.isdir(os.path.join(logdir,i)) for i in items):
            for dir in items:
                pathy = os.path.join(logdir,dir)
                filesInDir = os.listdir(pathy)
                img = [Image.open(os.path.join(pathy,i)) for i in filesInDir]
                imgArr = [np.asarray(a)  for a in img]
                imgs.append(imgArr)

        elif all(os.path.isfile(i) for i in items):
            for i in items:
                img = Image.open(os.path.join(logdir,i))
                imgArr = np.asarray(img)
                imgs.append(imgArr)

    print(imgs[0].shape)

    rows=len(imgs)
    cols = len(imgs[0])


    fig = plt.figure(figsize=(15,15))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
                    )

    for ax, im in zip(grid, imgs):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)

    plt.show()




    # img = Image.open(img)
    # img = np.asarray(img)


    # images = os.listdir('00073536\\2023-09-29-10-54-48-e11\\img\\')

# def image_grid(array, ncols=4):
#     index, height, width, channels = array.shape
#     nrows = index//ncols
    
#     img_grid = (array.reshape(nrows, ncols, height, width, channels)
#               .swapaxes(1,2)
#               .reshape(height*nrows, width*ncols))
    
#     return img_grid

# result = image_grid(np.array(img_arr))
# fig = plt.figure(figsize=(20., 20.))
# plt.imshow(result)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if args.command == 'open':
        open(args.name, plot=args.plot)
        
    if args.command == 'grid':
        grid(args.logdir)

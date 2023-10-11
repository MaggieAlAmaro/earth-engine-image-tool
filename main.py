import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image
import os

from utils import processDirOrFile, newFilename
import analytics, proccess, tests, visualize, rgba


plt.style.use('dark_background')


if __name__ == '__main__':
    print("Main")
    # parser = getParser()
    # args = parser.parse_args()
    # print(args)
    # if args.command == 'blah blah':
    #     processDirOrFile(blah)
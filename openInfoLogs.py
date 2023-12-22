import json, pickle
import datasetSplit

import argparse

def getParser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs, formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")
    openParser = subparsers.add_parser("open", help="Convert to png")   
    openParser.add_argument('pickle_filename', type=str, help='Name of image')  
    openParser.add_argument('-min', type=int, help='Name of image',default= -32768)#-10)

if __name__ == '__main__':
    # parser = getParser()
    # args = parser.parse_args()
    # print(args)
    # if args.command == 'open':
    #     processDirOrFile(blah)
    with open("info.json", "r") as f:
        data = json.loads(f)
        print(data)

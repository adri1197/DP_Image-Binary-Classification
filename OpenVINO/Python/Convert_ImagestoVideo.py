import cv2
import numpy as np
import glob
from argparse import ArgumentParser,SUPPRESS
import sys


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-o", "--output", help="Required. Path to an video file where it will be exported the video", required=True,
                      type=str)
    args.add_argument("-i", "--input", help="Required. Path to a folder with the whole set of images", required=True,
                      type=str)
    args.add_argument("--extension", help="Required. Extension of the images",
                      required=True,
                      type=str)

    return parser
def main():
    args = build_argparser().parse_args()
    img_array = []
    folder_path=args.input+'*.'+args.extension
    size = 0
    for filename in glob.glob(folder_path):
        print(filename+"\n")
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)


    out = cv2.VideoWriter(args.output,cv2.VideoWriter_fourcc(*'MP4V'), 20, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == '__main__':
    sys.exit(main() or 0)
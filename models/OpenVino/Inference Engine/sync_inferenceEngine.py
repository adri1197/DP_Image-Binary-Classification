#Samples/classification_sample
import sys
import os
from argparse import ArgumentParser,SUPPRESS
import cv2
import numpy as np
import logging as log
from time import time
import matplotlib.pyplot as plt
from openvino.inference_engine import IENetwork,IECore

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True,
                      type=str)
    args.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files",
                      required=True,
                      type=str, nargs="+")
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the"
                           " kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to a labels mapping file", default=None, type=str)
    args.add_argument("-nt", "--number_top", help="Optional. Number of top results", default=10, type=int)

    return parser

def main():
    log.basicConfig(format="[%(levelname)s]: %(message)s",level=log.INFO,stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + '.bin'

    #Plugin initialization for specified device and load extensions library
    log.info("Creating Inference Engine")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension,"CPU")
    #Read IR
    log.info('Loading network files:\n\t{}\n\t{}'.format(model_xml,model_bin))
    net = IENetwork(model=model_xml,weights=model_bin)


    #Devices
    if "CPU" in args.device:
        supported_layers = ie.query_network(net,"CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n{}".format(args.device,','.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample´s command line parameters using -l "
                    "or --cpu_extension command line argument")
            
            sys.exit(1)

    assert len(net.inputs.keys()) == 1,"Sample supports only single input topologies"
    assert len(net.outputs) == 1,"Sample supports only single output topologies"

    log.info('Preparing for input blobs')
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = len(args.input)

    #Read and pre-process input images
    n,c,h,w  = net.inputs[input_blob].shape
    print('n:{}\nc:{}\nh:{}\nw:{}'.format(n,c,h,w))
    images = np.ndarray(shape=(n,h,w,c))
    for i in range(n):
        image = cv2.imread(args.input[i],cv2.IMREAD_GRAYSCALE)/255.0
        #plt.imshow(image)
        #plt.show()
        print('Loaded image: ',image.shape)
        if image.shape[:-1] != (h,w):
            log.warning("Image {} is resized from {} to {}".format(args.input[i],image.shape[:-1],(h,w)))
            image = cv2.resize(image,dsize=(h,w))
        image_input = np.array(image).reshape(-1,128,128,1)
        image_input = image_input.transpose((0,3,1,2)) #Change data layout from HWC to CHW
        #images[i] = image_input
        #print('Images[i] | shape: ',images[i].shape)
        print('image_input:',image_input.shape)
    log.info("Batch size is {}.".format(n))

    #Loading model to the plugin
    log.info('Loading model to the plugin')
    exec_net = ie.load_network(network=net,device_name=args.device)

    #Start sync inference
    log.info("Starting inference in synchronous mode")
    res = exec_net.infer(inputs={input_blob: image_input})
    #Processing the output blob
    log.info("Processing output blob")
    res = res[out_blob]
    #log.info("Results:",res)
    categories = ['damaged','undamaged']
    results = res[0][0]
    write_result = 1 if results >= 0.5 else 0 
    print('Image: {} -> Prediction: {} ({})'.format(args.input[0],results,categories[write_result]))
"""     if args.labels:
        with open(args.labels,'r') as f:
            labels_map = [x.split(sep='',maxsplit=1)[-1].strip() for x in f]
    else:
        labels_map = None
 """
"""     classid_str = "classid"
    probability_str = "probability"
    for i,probs in enumerate(res):
        probs = np.squeeze(probs)
        top_ind = np.argsort(probs)[-args.number_top:][::-1]
        print("Image: {}\n".format(args.input[i]))
        print(classid_str,probability_str)
        print("{} {}".format('-' * len(classid_str),'-' * len(probability_str)))
        for id in top_ind:
            det_label = labels_map[id] if labels_map else "{}".format(id)
            label_length = len(det_label)
            space_num_before = (len(classid_str) - label_length) // 2
            space_num_after = len(classid_str) - (space_num_before + label_length) + 2
            space_num_before_prob = (len(probability_str) - len(str(probs[id]))) // 2
            print("{}{}{}{}{:.7f}".format(' ' * space_num_before, det_label,
                                          ' ' * space_num_after, ' ' * space_num_before_prob,
                                          probs[id]))
        print("\n") """





if __name__ == '__main__':
    sys.exit(main() or 0)


#Samples/classification_sample
import sys
import os
from argparse import ArgumentParser,SUPPRESS
import cv2
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IENetwork,IECore
import threading

class InferReqWrap:
    def __init__(self, request, id, num_iter):
        self.id = id
        self.request = request
        self.num_iter = num_iter
        self.cur_iter = 0
        self.cv = threading.Condition()
        self.request.set_completion_callback(self.callback, self.id)

    def callback(self, statusCode, userdata):
        if (userdata != self.id):
            log.error("Request ID {} does not correspond to user data {}".format(self.id, userdata))
        elif statusCode != 0:
            log.error("Request {} failed with status code {}".format(self.id, statusCode))
        self.cur_iter += 1
        log.info("Completed {} Async request execution".format(self.cur_iter))
        if self.cur_iter < self.num_iter:
            # here a user can read output containing inference results and put new input
            # to repeat async request again
            self.request.async_infer(self.input)
        else:
            # continue sample execution after last Asynchronous inference request execution
            self.cv.acquire()
            self.cv.notify()
            self.cv.release()

    def execute(self, mode, input_data):
        if (mode == "async"):
            log.info("Start inference ({} Asynchronous executions)".format(self.num_iter))
            self.input = input_data
            # Start async request for the first time. Wait all repetitions of the async request
            time1=time()
            self.request.async_infer(input_data)
            self.cv.acquire()
            self.cv.wait()
            self.cv.release()
            time2=time()
        elif (mode == "sync"):
            log.info("Start inference ({} Synchronous executions)".format(self.num_iter))
            time1=time()
            for self.cur_iter in range(self.num_iter):
                # here we start inference synchronously and wait for
                # last inference request execution
                self.request.infer(input_data)
                log.info("Completed {} Sync request execution".format(self.cur_iter + 1))
            time2=time()
        else:
            log.error("wrong inference mode is chosen. Please use \"sync\" or \"async\" mode")
            sys.exit(1)
        return time2-time1

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
    args.add_argument("-api", "--inference",
                      help="Optional. Specify the type of inference that you want to apply [sync,async]",
                      default="sync", type=str)

    return parser

def inference(type_inference,exec_net,images,input_blob,out_blob):
    # create one inference request for asynchronous execution
    request_id = 0
    infer_request = exec_net.requests[request_id];
    num_iter = 1
    request_wrap = InferReqWrap(infer_request, request_id, num_iter)

    # Start inference request execution. Wait for last execution being completed
    time = request_wrap.execute(type_inference, {input_blob: images})

    # Processing output blob
    res = infer_request.outputs[out_blob]

    log.info("Processing output blob")
    return res,time


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

    #Loading model to the plugin
    log.info('Loading model to the plugin')
    exec_net = ie.load_network(network=net,device_name=args.device,num_requests=4)

    log.info('Preparing for input blobs')
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    #net.batch_size = len(args.input)

    categories = ['damaged','undamaged']
    #Read and pre-process input images
    #n,c,h,w  = net.inputs[input_blob].shape
    n,h,w,c  = net.inputs[input_blob].shape
    print('n:{}\nc:{}\nh:{}\nw:{}'.format(n,c,h,w))
    #images = np.ndarray(shape=(n,c,h,w))
    images = np.ndarray(shape=(n,h,w,c))

    log.info("Batch size is {}.".format(n))
    ips_list = []

    for name in args.input:
        if os.path.isdir(name):
            #img_input = os.path.join(name,os.listdir(name))
            img_input = os.listdir(name)
        else:
            img_input = [name]

        length_images =  len(img_input)
        if length_images > 1:
            img_input.sort()
        

        #Take the images to preprocess them
        for ind,img in enumerate(img_input):
            #print("[{}] {}".format(ind,img))
            image = cv2.imread(os.path.join(name,img),cv2.IMREAD_GRAYSCALE)

            image_input = np.array(image).reshape(-1,128,128,1)

            #image_input = image_input.transpose((0,3,1,2)) #Change data layout from HWC to CHW
            #image_input = image.transpose((2,0,1)) #Change data layout from HWC to CHW

            images[ind%n] = image_input[0]
            #print("Indice images:{}".format(ind%n))

            if ind%n == n-1 or ind==length_images-1:
            #Start Inference
                type_ie = args.inference.casefold()
                if type_ie == "sync" or type_ie == "async":
                    res,time = inference(type_ie,exec_net,images,input_blob,out_blob)
                else:
                    log.error("wrong inference mode is chosen. Please use \"sync\" or \"async\" mode")
                    sys.exit(1)

                    
                ips=(1/time)*n
                ips_list.append(ips)
                log.info('Inference: {} IPS'.format(ips))

                if ind==length_images-1 and length_images%n !=0:
                    number=length_images%n 
                else:
                    number=n

                    
                for j,results in enumerate(res):
                    write_result = 1 if results[0] >= 0.5 else 0
                    print('Image {}: {} -> Prediction: {} ({})'.format(ind-number+j+2,img_input[ind-number+j+1],results[0],categories[write_result]))
                    if ind == length_images-1 and number-1 <= j:
                        break

    print("Average IPS: {}".format(sum(ips_list) / len(ips_list)))



if __name__ == '__main__':
    sys.exit(main() or 0)


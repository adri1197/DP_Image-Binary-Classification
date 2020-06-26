from __future__ import print_function
import sys
import os
from argparse import ArgumentParser,SUPPRESS
import cv2
import numpy as np
import logging as log
from time import time
import matplotlib.pyplot as plt
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
            self.request.async_infer(input_data)
            self.cv.acquire()
            self.cv.wait()
            self.cv.release()
        elif (mode == "sync"):
            log.info("Start inference ({} Synchronous executions)".format(self.num_iter))
            for self.cur_iter in range(self.num_iter):
                # here we start inference synchronously and wait for
                # last inference request execution
                self.request.infer(input_data)
                log.info("Completed {} Sync request execution".format(self.cur_iter + 1))
        else:
            log.error("wrong inference mode is chosen. Please use \"sync\" or \"async\" mode")
            sys.exit(1)


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True,
                      type=str)
    args.add_argument("-i", "--input", help="Required. Path to a video file",
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
    args.add_argument("-ie", "--inference",
                      help="Optional. Specify the type of inference that you want to apply [sync,async]",
                      default="sync", type=str)
    args.add_argument("--show",
                    help="Optional. Display a window with all the frames which are being proccessed",
                    action='store_true')

    return parser

def getPathVideos(path):
    videos = []
    if os.path.isdir(path):
        for path_file in os.listdir(path):
            videos.append(os.path.join(path,path_file))
    else:
        videos.append(path)
    return videos


def sync_inference(exec_net,images,input_blob,out_blob):
    #Start sync inference
    log.info("Starting inference in synchronous mode")
    time1 = time()
    res = exec_net.infer(inputs={input_blob: images})
    #Processing the output blob
    res = res[out_blob]
    time2 = time()
    log.info("Processing output blob")
    return res,time2-time1


def async_inference(exec_net,images,input_blob,out_blob):
    # create one inference request for asynchronous execution
    request_id = 0
    infer_request = exec_net.requests[request_id];
    num_iter = 1
    request_wrap = InferReqWrap(infer_request, request_id, num_iter)

    # Start inference request execution. Wait for last execution being completed
    time1 = time()
    request_wrap.execute("async", {input_blob: images})

    # Processing output blob
    res = infer_request.outputs[out_blob]
    time2 = time()
    log.info("Processing output blob")
    return res,time2-time1

def main():
    log.basicConfig(format="[%(levelname)s]: %(message)s",level=log.INFO,stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + '.bin'
    categories = ['damaged','undamaged']
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
    #net.batch_size = len(args.input)
    
    #Loading model to the plugin
    log.info('Loading model to the plugin')
    exec_net = ie.load_network(network=net,device_name=args.device,num_requests=4)
    
    n,h,w,c  = net.inputs[input_blob].shape
    print('n:{}\nc:{}\nh:{}\nw:{}'.format(n,c,h,w))
    #images = np.ndarray(shape=(n,c,h,w))
    images = np.ndarray(shape=(n,h,w,c))
    videos_path = getPathVideos(args.input[0])
    for i in range(len(videos_path)):   
        n_frames = 0
        capture = cv2.VideoCapture(videos_path[i])
        length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Video -> {}".format(videos_path[i]))
        print("Frames (in total):{}".format(length))
        print("{}\n".format('-' * len(videos_path[i])))        
        if capture.isOpened() == True:
            while(n_frames < length):
                #Capture frame by frame
                ret,frame = capture.read()
                #print("Nºframe:",n_frames)
                grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                #grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                resize_frame = cv2.resize(grayscale, (128, 128)) 
                if args.show:
                    resize_frame_show = cv2.resize(grayscale, (720, 720)) 
                    cv2.imshow('Earthquake Classification',resize_frame_show)
                image_input = np.array(resize_frame).reshape(-1,128,128,1)
                #print("Images-shape:",n_frames%n)
                images[n_frames%n] = image_input[0]
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                n_frames = n_frames+1
                #INFERENCE STAGE
                if (n_frames % n == 0 and n_frames != 0) or (n != 0 and n_frames == length):

                    if args.inference.casefold() == "sync":
                        res,time = sync_inference(exec_net,images,input_blob,out_blob)
                    elif args.inference.casefold() == "async":
                        res,time = async_inference(exec_net,images,input_blob,out_blob)

                    log.info("Inference Per Second:{}".format((1/time)*n))

                    if n_frames % n != 0:
                        num_show =  n_frames % n
                    else:
                        num_show = n
                    images = np.empty(shape=(num_show,h,w,c))
                    for j in range(num_show):
                        results = res[j][0]
                        write_result = 1 if results >= 0.5 else 0
                        print('Frame {} -> Prediction: [{}] {:.3f}% damaged'.format((n_frames-num_show+1)+j,categories[write_result],100-(results*100)))
        # When everything done, release the capture
        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main() or 0)


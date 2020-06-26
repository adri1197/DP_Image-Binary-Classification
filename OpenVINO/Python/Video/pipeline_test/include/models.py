"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from collections import deque
from itertools import cycle

import cv2
import numpy as np
from openvino.inference_engine import IENetwork
from include.utils.constants import CPU_DEVICE_NAME, MULTI_DEVICE_NAME, GPU_DEVICE_NAME, MYRIAD_DEVICE_NAME, BIN_EXTENSION
from include.utils.utils import parse_devices,parse_nstreams_value_per_device


def center_crop(frame, crop_size):
    img_h, img_w, _ = frame.shape

    x0 = int(round((img_w - crop_size[0]) / 2.))
    y0 = int(round((img_h - crop_size[1]) / 2.))
    x1 = x0 + crop_size[0]
    y1 = y0 + crop_size[1]

    return frame[y0:y1, x0:x1, ...]


def adaptive_resize(frame, dst_size):
    #print(frame.shape)
    h, w = frame.shape
    scale = dst_size / min(h, w)
    ow, oh = int(w * scale), int(h * scale)
    print("W:{} | H:{}".format(ow,oh))
    if ow == w and oh == h:
        return frame
    return cv2.resize(frame, (ow, oh))


def preprocess_frame(frame):
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    resize_frame = cv2.resize(frame, (128,128))
    frame = np.array(resize_frame).reshape(-1,1,128,128)
    #print(frame.shape)
    #frame = adaptive_resize(frame, 128)
    #frame = center_crop(frame, (128, 128))

    #frame = frame.transpose((2, 0, 1))  # HWC -> CHW
    #frame = frame[np.newaxis,..]  # add batch dimension
    return frame


class AsyncWrapper:
    def __init__(self, ie_model, num_requests):
        self.net = ie_model
        self.num_requests = num_requests

        self._result_ready = False
        self._req_ids = cycle(range(num_requests))
        self._result_ids = cycle(range(num_requests))
        self._frames = deque(maxlen=num_requests)

    def infer(self, model_input, frame=None):
        """Schedule current model input to infer, return last result"""
        next_req_id = next(self._req_ids)
        self.net.async_infer(model_input, next_req_id)

        last_frame = self._frames[0] if self._frames else frame

        self._frames.append(frame)
        if next_req_id == self.num_requests - 1:
            self._result_ready = True

        if self._result_ready:
            result_req_id = next(self._result_ids)
            result = self.net.wait_request(result_req_id)
            return result, last_frame
        else:
            return None, None


class IEModel:
    def __init__(self, model_xml, model_bin, ie_core, target_device, num_requests, batch_size=1):
        # Plugin initialization for specified device and load extensions library if specified

        # Read IR
        self.ie_core = ie_core
        self.net = self.ie_core.read_network(model_xml, model_bin)
        self.net.batch_size = batch_size
        assert len(self.net.inputs.keys()) == 1, "One input is expected"
        assert len(self.net.outputs) == 1, "One output is expected"

        print("Loading IR to the plugin...")

        self.exec_net = self.ie_core.load_network(network=self.net, device_name=target_device, num_requests=num_requests)
        self.input_name = next(iter(self.net.inputs))
        self.output_name = next(iter(self.net.outputs))
        self.input_size = self.net.inputs[self.input_name]
        self.output_size = self.exec_net.requests[0].outputs[self.output_name].shape
        self.num_requests = num_requests

    def set_config(self,device,number_streams: int,number_threads: int = None, infer_threads_pinning: int = None):
        devices = parse_devices(device)
        device_number_streams = parse_nstreams_value_per_device(devices, number_streams)
        for device_name in  device_number_streams.keys():
            key = device_name + "_THROUGHPUT_STREAMS"
            supported_config_keys = self.ie_core.get_metric(device_name, 'SUPPORTED_CONFIG_KEYS')
            if key not in supported_config_keys:
                raise Exception("Device " + device_name + " doesn't support config key '" + key + "'! " +
                                "Please specify -nstreams for correct devices in format  <dev1>:<nstreams1>,<dev2>:<nstreams2>");

        for device in devices:
            if device == CPU_DEVICE_NAME:  # CPU supports few special performance-oriented keys
                # limit threading for CPU portion of inference
                if number_threads:
                    self.ie_core.set_config({'CPU_THREADS_NUM': str(number_threads)}, device)

                if MULTI_DEVICE_NAME in device and GPU_DEVICE_NAME in device:
                    self.ie_core.set_config({'CPU_BIND_THREAD': 'NO'}, CPU_DEVICE_NAME)
                else:
                    # pin threads for CPU portion of inference
                    self.ie_core.set_config({'CPU_BIND_THREAD': infer_threads_pinning}, device)

                # for CPU execution, more throughput-oriented execution via streams
                # for pure CPU execution, more throughput-oriented execution via streams
                cpu_throughput = {'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO'}
                if device in device_number_streams.keys():
                    cpu_throughput['CPU_THROUGHPUT_STREAMS'] = str(device_number_streams.get(device))
                self.ie_core.set_config(cpu_throughput, device)
                device_number_streams[device] = self.ie_core.get_config(device, 'CPU_THROUGHPUT_STREAMS')

            elif device == GPU_DEVICE_NAME:
                gpu_throughput = {'GPU_THROUGHPUT_STREAMS': 'GPU_THROUGHPUT_AUTO'}
                if device in device_number_streams.keys():
                    gpu_throughput['GPU_THROUGHPUT_STREAMS'] = str(device_number_streams.get(device))
                self.ie_core.set_config(gpu_throughput, device)
                device_number_streams[device] = self.ie_core.get_config(device, 'GPU_THROUGHPUT_STREAMS')

                if MULTI_DEVICE_NAME in device and CPU_DEVICE_NAME in device:
                    # multi-device execution with the CPU+GPU performs best with GPU trottling hint,
                    # which releases another CPU thread (that is otherwise used by the GPU driver for active polling)
                    self.ie_core.set_config({'CLDNN_PLUGIN_THROTTLE': '1'}, device)

            elif device == MYRIAD_DEVICE_NAME:
                myriad_config = {"VPU_HW_STAGES_OPTIMIZATION": "YES"}
                self.ie_core.set_config(myriad_config,MYRIAD_DEVICE_NAME)
                self.ie_core.set_config({'LOG_LEVEL': 'LOG_INFO'}, MYRIAD_DEVICE_NAME)

    def infer(self, frame):
        input_data = {self.input_name: frame}
        infer_result = self.exec_net.infer(input_data)
        return infer_result[self.output_name]

    def async_infer(self, frame, req_id):
        input_data = {self.input_name: frame}
        self.exec_net.start_async(request_id=req_id, inputs=input_data)
        pass

    def wait_request(self, req_id):
        self.exec_net.requests[req_id].wait()
        return self.exec_net.requests[req_id].outputs[self.output_name]


class ActionRecognitionSequential:
    def __init__(self, encoder, decoder=None):
        self.encoder = encoder
        self.decoder = decoder

    def infer(self, input):
        if self.decoder is not None:
            embeddigns = self.encoder.infer(input[0])
            decoder_input = embeddigns.reshape(1, 16, 512)
            return self.decoder.infer(decoder_input)

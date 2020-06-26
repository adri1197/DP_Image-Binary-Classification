#!/usr/bin/env python
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

from __future__ import print_function

import sys
from argparse import ArgumentParser, SUPPRESS

from openvino.inference_engine import IECore

from include.models import IEModel
from include.result_renderer import ResultRenderer
from include.steps import run_pipeline
from include.utils.constants import XML_EXTENSION,BIN_EXTENSION,MYRIAD_DEVICE_NAME,CPU_DEVICE_NAME
import os


def check_extension(path):
    return path.endswith('.api') or path.endswith('.mp4') or path.endswith('.m4v')


def get_videos(path):
    if os.path.isdir(path):
        videos = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and check_extension(f)]
    else:
        videos = [path]
    return videos



def run(model, videos, no_show, fps=30, labels=None):
    result_presenter = ResultRenderer(no_show=no_show, labels=labels)
    run_pipeline(videos, model, result_presenter.render_frame, fps=fps)


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m","--model",help="Required. Path to model", required=True, type=str)
    args.add_argument("-i", "--input",
                      help="Required. Id of the video capturing device to open (to open default camera just pass 0), "
                           "path to a video or a .txt file with a list of ids or video files (one object per line)",
                      required=True, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. For CPU custom layers, if any. Absolute path to a shared library with the "
                           "kernels implementation.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify a target device to infer on. CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for the device specified. "
                           "Default value is CPU",
                      default="CPU", type=str)
    args.add_argument("--fps", help="Optional. FPS for renderer", default=30, type=int)
    args.add_argument("--no_show", action='store_true', help="Optional. Don't show output")
    args.add_argument('-nstreams', '--number_streams', type=str, required=False, default=None,
                    help='Optional. Number of streams to use for inference on the CPU/GPU in throughput mode '
                        '(for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> '
                        'or just <nstreams>). '
                        'Default value is determined automatically for a device. Please note that although the automatic selection '
                        'usually provides a reasonable performance, it still may be non - optimal for some cases, especially for very small networks. '
                        'See samples README for more details.')
    args.add_argument('-nthreads', '--number_threads', type=int, required=False, default=None,
                      help='Number of threads to use for inference on the CPU '
                           '(including HETERO and MULTI cases).')
    args.add_argument('-pin', '--infer_threads_pinning', type=str, required=False, default='YES', choices=['YES', 'NO', 'NUMA'],
                      help='Optional. Enable  threads->cores (\'YES\' is default value), threads->(NUMA)nodes (\'NUMA\') or completely  disable (\'NO\')' 
                           'CPU threads pinning for CPU-involved inference.')
    return parser


def main():
    args = build_argparser().parse_args()

    videos = get_videos(args.input)

    if not args.input:
        raise ValueError("--input option is expected")

    ie = IECore()

    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")


    model_xml = args.model
    model_bin = args.model.replace(XML_EXTENSION, BIN_EXTENSION)
    labels=["damaged", "undamaged"]

    model = IEModel(model_xml, model_bin, ie, args.device,
                      num_requests=(3 if args.device == 'MYRIAD' else 4))
    
    model.set_config(args.device,args.number_streams,args.number_threads,args.infer_threads_pinning)

    run(model,videos, args.no_show, args.fps, labels)


if __name__ == '__main__':
    sys.exit(main() or 0)
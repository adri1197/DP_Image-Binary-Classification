"""
 Copyright (C) 2018-2020 Intel Corporation

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

import os
import cv2
import numpy as np
from glob import glob

from .constants import IMAGE_EXTENSIONS, BINARY_EXTENSIONS
from .logging import logger


def get_blob_shape(layer, batch_size: int):
    shape = layer.shape.copy()
    layout = layer.layout

    try:
        batch_index = layout.index('N')
    except ValueError:
        batch_index = 1 if layout == 'C' else -1

    if batch_index != -1 and shape[batch_index] != batch_size:
        shape[batch_index] = batch_size

    return shape


def is_image(blob):
    if blob.layout != "NCHW" and blob.layout != 'NHWC':
        return False
    if blob.layout == "NCHW":
        channels = blob.shape[1]
    elif blob.layout == 'NHWC':
        channels = blob.shape[-1]
    #print("Bob Shape: {}".format(blob.shape))
    #print("Layout: {}".format(blob.layout))
    #print("Is_Image Channels: {}".format(channels))
    return channels == 1


def set_inputs(paths_to_input, batch_size, input_info, requests):
  images_path,requests_input_data = get_inputs(paths_to_input, batch_size, input_info, requests)
  for i in range(len(requests)):
    inputs = requests[i].inputs
    for k, v in requests_input_data[i].items():
        if k not in inputs.keys():
            raise Exception("No input with name {} found!".format(k))
        inputs[k][:] = v
  return images_path

def get_inputs(paths_to_input, batch_size, input_info, requests):
    input_image_sizes = {}
    for key in sorted(input_info.keys()):
        if is_image(input_info[key]):
            input_image_sizes[key] = (input_info[key].shape[1], input_info[key].shape[2])
        logger.info("Network input '{}' precision {}, dimensions ({}): {}".format(key,
                                                                                  input_info[key].precision,
                                                                                  input_info[key].layout,
                                                                                  " ".join(str(x) for x in
                                                                                           input_info[key].shape)))

    images_count = len(input_image_sizes.keys())

    image_files = list()
    if paths_to_input:
        image_files = get_files_by_extensions(paths_to_input, IMAGE_EXTENSIONS)
        image_files.sort()

        images_to_be_used = images_count * batch_size * len(requests)
        #print("Images To Be Used:{}".format(images_to_be_used))
        if images_to_be_used > 0 and len(image_files) == 0:
            logger.warn("No supported image inputs found! Please check your file extensions: {}".format(
                ",".join(IMAGE_EXTENSIONS)))
        elif images_to_be_used > len(image_files):
            raise Exception("The batch size is bigger than the images avaliable")
        elif images_to_be_used < len(image_files):
            logger.warn(
                "Some image input files will be ignored: only {} files are required from {}".format(images_to_be_used,
                                                                                                    len(image_files)))

    requests_input_data = []
    for request_id in range(0, len(requests)):
        logger.info("Infer Request {} filling".format(request_id))
        input_data = {}
        keys = list(sorted(input_info.keys()))
        for key in keys:
            if is_image(input_info[key]):
                # input is image
                if len(image_files) > 0:
                    input_data[key] = fill_blob_with_image(image_files, request_id, batch_size, keys.index(key),
                                                           len(keys), input_info[key])


        requests_input_data.append(input_data)

    return image_files[:images_to_be_used],requests_input_data


def get_files_by_extensions(paths_to_input, extensions):
    get_extension = lambda file_path: file_path.split(".")[-1].upper()

    input_files = list()
    for path_to_input in paths_to_input:
        if os.path.isfile(path_to_input):
            files = [os.path.normpath(path_to_input)]
        else:
            path = os.path.join(path_to_input, '*')
            files = glob(path, recursive=True)
        for file in files:
            file_extension = get_extension(file)
            if file_extension in extensions:
                input_files.append(file)

    return input_files

def fill_blob_with_image(image_paths, request_id, batch_size, input_id, input_size, layer):
    shape = layer.shape
    images = np.ndarray(shape)
    image_index = request_id * batch_size * input_size + input_id
    for b in range(batch_size):
        image_index %= len(image_paths)
        image_filename = image_paths[image_index]
        logger.info('Prepare image {}'.format(image_filename))
        image = cv2.imread(image_filename,cv2.IMREAD_GRAYSCALE)

        if layer.layout == 'NCHW':
            new_im_size = (-1,1,128,128)
        elif layer.layout == 'NHWC':
            new_im_size = (-1,128,128,1)
        #print(layer.layout)
        
        if image.shape[:-1] != new_im_size:
            image = np.array(image).reshape(new_im_size)[0]
            
        images[b] = image

        image_index += input_size
    return images

def get_dtype(precision):
    format_map = {
      'FP32' : np.float32,
      'I32'  : np.int32,
      'FP16' : np.float16,
      'I16'  : np.int16,
      'U16'  : np.uint16,
      'I8'   : np.int8,
      'U8'   : np.uint8,
    }
    if precision in format_map.keys():
        return format_map[precision]
    raise Exception("Can't find data type for precision: " + precision)

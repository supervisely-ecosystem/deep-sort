import torch
import cv2

import os

import supervisely_lib as sly

import serve_ann_keeper
import json

import functools
from functools import lru_cache

import serve_globals as g


@sly.timeit
def get_output_classes_and_tags(session_id):
    output_classes_and_tags = g.api.task.send_request(session_id, "get_output_classes_and_tags", data={}, timeout=3)
    sly.logger.info("🟩 Model has been successfully deployed")



@lru_cache(maxsize=10)
def get_frame_np(api, video_id, frame_index):
    img_rgb = api.video.frame.download_np(video_id, frame_index)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


def send_error_data(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = None
        try:
            value = func(*args, **kwargs)
        except Exception as e:
            request_id = kwargs["context"]["request_id"]
            g.my_app.send_response(request_id, data={"error": repr(e)})
        return value

    return wrapper


def get_annotations_from_detector(detector_task_id, context):
    response = g.api.task.send_request(detector_task_id, "process_tracker",
                                       data={'context': context}, timeout=500)

    if response['rc'] == 0:
        return response['ann']
    else:
        return None


def get_objects_count(ann_data):
    objects_ids = []

    for frame_index, frame_data in ann_data.items():
        objects_ids.extend(frame_data['ids'])
    
    return max(objects_ids) + 1


def get_video_shape(video_path):
    zero_image_name = os.listdir(video_path)[0]
    image_shape = cv2.imread(os.path.join(video_path, zero_image_name)).shape

    return tuple([int(image_shape[1]), int(image_shape[0])])



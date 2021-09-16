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


def get_class_names_for_each_object(ann_data):
    objects_labels = {}  # object_id: object_label
    for frame_index, frame_data in ann_data.items():
        for curr_id, label in zip(frame_data['ids'], frame_data['labels']):
            rc = objects_labels.get(curr_id, None)
            if rc is None:
                objects_labels[curr_id] = label

    return objects_labels.values()


def get_video_shape(video_path):
    zero_image_name = os.listdir(video_path)[0]
    image_shape = cv2.imread(os.path.join(video_path, zero_image_name)).shape

    return tuple([int(image_shape[1]), int(image_shape[0])])


def clear_empty_ids(tracker_annotations):
    id_mappings = {}
    last_ordinal_id = 0

    for frame_index, data in tracker_annotations.items():
        data_ids_temp = []
        for current_id in data['ids']:
            new_id = id_mappings.get(current_id, -1)
            if new_id == -1:
                id_mappings[current_id] = last_ordinal_id
                last_ordinal_id += 1
                new_id = id_mappings.get(current_id, -1)
            data_ids_temp.append(new_id)
        data['ids'] = data_ids_temp

    return tracker_annotations


def generate_project_meta_by_classes(selected_classes):
    obj_class_list = []
    for curr_class in selected_classes:
        curr_obj_class = sly.ObjClass(curr_class, sly.Rectangle)
        obj_class_list.append(curr_obj_class)

    objects = sly.ObjClassCollection(obj_class_list)
    return sly.ProjectMeta(obj_classes=objects, project_type=sly.ProjectType.VIDEOS)


def update_project_meta(video_id, selected_classes):
    project_id = g.api.video.get_info_by_id(video_id).project_id
    project_meta = g.api.project.get_meta(project_id)
    project_meta = sly.ProjectMeta.from_json(project_meta)

    tracker_project_meta = generate_project_meta_by_classes(selected_classes)

    project_meta.merge(tracker_project_meta)

    g.api.project.update_meta(project_id, project_meta.to_json())





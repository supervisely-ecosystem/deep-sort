import os
import supervisely_lib as sly
import serve_globals as g

import random

import argparse


def init(data, state):
    # system

    state["device"] = 'cuda:0'
    state["detectorThres"] = 0.6
    state["cosSimilarity"] = 0.6

    state["trackingStarted"] = False

    # stepper
    state["collapsed3"] = True
    state["disabled3"] = True
    data["done3"] = False


def restart(data, state):
    data["done3"] = False


def get_video_for_preview():
    videos_table = g.api.app.get_field(g.task_id, 'data.videosTable')
    selected_videos = g.api.app.get_field(g.task_id, 'state.selectedVideos')

    frames_min = g.api.app.get_field(g.task_id, 'state.framesMin')
    frames_max = g.api.app.get_field(g.task_id, 'state.framesMax')

    random.shuffle(selected_videos)
    video_name = selected_videos[0]

    video_info = [row for row in videos_table if row['name'] == video_name][0]

    start_frame = 0
    end_frame = frames_max[video_info['name']] + 1
    while end_frame > frames_max[video_info['name']]:
        start_frame = random.randint(frames_min[video_info['name']], frames_max[video_info['name']])
        end_frame = start_frame + 4

    return video_info['videoId'], (start_frame, end_frame)


def init_opt(state):
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str,
                        default=state['device'], help='device to process')
    parser.add_argument('--detection_threshold', type=float,
                        default=state['detectorThres'],
                        help='threshold for detector model')
    parser.add_argument('--nms_max_overlap', type=float,
                        default=1.0,
                        help='Non-maxima suppression threshold: Maximum detection overlap.')
    parser.add_argument('--max_cosine_distance', type=float,
                        default=state['cosSimilarity'],
                        help='Gating threshold for cosine distance metric (object appearance).')
    parser.add_argument('--nn_budget', type=int,
                        default=None,
                        help='Maximum size of the appearance descriptors allery. If None, no budget is enforced.')

    parser.add_argument('--thickness', type=int,
                        default=1, help='Thickness of the bounding box strokes')

    parser.add_argument('--info', action='store_true',
                        help='Print debugging info.')

    g.opt = parser.parse_args()



def upload_deep_sort_info(state):
    deep_sort_info = {}

    deep_sort_info["device"] = state["device"]
    deep_sort_info["detectorThres"] = state["detectorThres"]
    deep_sort_info["cosSimilarity"] = state["cosSimilarity"]

    g.api.app.set_field(g.task_id, 'data.deepSortInfo', deep_sort_info)


@g.my_app.callback("apply_parameters")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def apply_parameters(api: sly.Api, task_id, context, state, app_logger):

    init_opt(state)
    upload_deep_sort_info(state)

    g.finish_step(3)


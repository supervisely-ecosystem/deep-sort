import os
import supervisely_lib as sly
import serve_globals as g

import random


def init(data, state):
    # system

    state["confThres"] = 0.4
    state["previewLoading"] = False

    # stepper
    data["videoUrl"] = None

    state["collapsed5"] = True
    state["disabled5"] = True
    data["done5"] = False


def restart(data, state):
    data["done5"] = False


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


@g.my_app.callback("apply_parameters")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def apply_parameters(api: sly.Api, task_id, context, state, app_logger):
    g.finish_step(5)


@g.my_app.callback("generate_annotation_example")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def generate_annotation_example(api: sly.Api, task_id, context, state, app_logger):
    try:
        fields = [
            {"field": "data.videoUrl", "payload": None}
        ]
        api.task.set_fields(task_id, fields)

        video_id, frames_range = get_video_for_preview()
        result = g.api.task.send_request(state['sessionId'], "inference_video_id",
                                         data={'videoId': video_id,
                                               'framesRange': frames_range,
                                               'confThres': state['confThres'],
                                               'isPreview': True})

        fields = [
            {"field": "data.videoUrl", "payload": result['preview_url']},
            {"field": "state.previewLoading", "payload": False},
        ]
        api.task.set_fields(task_id, fields)
    except Exception as ex:
        fields = [
            {"field": "state.previewLoading", "payload": False},
        ]
        api.task.set_fields(task_id, fields)
        raise ex

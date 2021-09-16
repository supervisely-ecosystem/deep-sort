import os

import serve_globals as g
import sly_functions
import supervisely_lib as sly

import ui

from sly_tracker_container import TrainedTrackerContainer


@g.my_app.callback("track")
@sly.timeit
# @sly_functions.send_error_data
def track_in_video_annotation_tool(api: sly.Api, task_id, context, state, app_logger):
    detector_session_id = g.model_info['session_id']

    context['selectedClasses'] = g.selected_classes
    context['detectorThreshold'] = g.opt.detection_threshold

    annotations = sly_functions.get_annotations_from_detector(detector_session_id, context)

    tracker_container = TrainedTrackerContainer(context)
    tracker_container.download_frames()
    tracker_container.dump_annotations(annotations)

    tracker_annotations = tracker_container.track()

    sly_functions.update_project_meta(tracker_container.video_id, g.selected_classes)

    g.api.video.annotation.append(tracker_container.video_id, tracker_annotations)
    tracker_container.update_progress(len(tracker_container.frames_indexes) - 1)


def main():

    data = {}
    state = {}
    data["ownerId"] = g.owner_id
    data["teamId"] = g.team_id

    g.my_app.compile_template(g.serve_source_path)

    ui.init(data, state)

    g.my_app.run(data=data, state=state)


# @TODO: test multiclass
# @TODO: check GPU availability
# @TODO: ping Detector Model
# @TODO: notify bars customize
# @TODO: approve launch format
# @TODO: add confidence tags


if __name__ == "__main__":
    sly.main_wrapper("main", main)

    # track_in_video_annotation_tool({  # for debug
    #     "teamId": 238,
    #     "workspaceId": 343,
    #     "videoId": 1253362,
    #     "objectIds": [236670],
    #     "figureIds": [54200821],
    #     "frameIndex": 430,
    #     "direction": 'forward',
    #     'frames': 100,
    #     'trackId': '5b82a928-0566-4d4d-a8e3-35f5abc736fe',
    #     'figuresIds': [54200821]
    # })

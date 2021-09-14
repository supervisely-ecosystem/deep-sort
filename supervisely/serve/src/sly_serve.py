import os

import serve_globals as g
import sly_functions
import supervisely_lib as sly

from sly_tracker_container import TrainedTrackerContainer


@g.my_app.callback("track")
@sly.timeit
# @sly_functions.send_error_data
# def track_in_video_annotation_tool(api: sly.Api, task_id, context, state, app_logger):
def track_in_video_annotation_tool(context):
    detector_session_id = 9189
    annotations = sly_functions.get_annotations_from_detector(detector_session_id, context)

    tracker_container = TrainedTrackerContainer(context)
    tracker_container.download_frames()
    tracker_container.dump_annotations(annotations)

    annotations = tracker_container.track()

    g.api.video.annotation.append(tracker_container.video_id, annotations)
    tracker_container.update_progress(len(tracker_container.frames_indexes) - 1)


def main():
    sly.logger.info("Script arguments", extra={
        "device": g.device
    })

    # detector_session_id = g.api.app.get_field(g.task_id, 'state.detectorSessionId')
    detector_session_id = 9189

    sly_functions.get_output_classes_and_tags(detector_session_id)

    g.my_app.run()


if __name__ == "__main__":
    # sly.main_wrapper("main", main)

    track_in_video_annotation_tool({  # for debug
        "teamId": 238,
        "workspaceId": 343,
        "videoId": 1253362,
        "objectIds": [236670],
        "figureIds": [54200821],
        "frameIndex": 430,
        "direction": 'forward',
        'frames': 100,
        'trackId': '5b82a928-0566-4d4d-a8e3-35f5abc736fe',
        'figuresIds': [54200821]
    })

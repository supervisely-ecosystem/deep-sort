import argparse
import json

import sly_functions
import serve_globals as g
import os
import cv2

import supervisely_lib as sly

import serve_ann_keeper
import sly_tracker

from functools import partial

from tqdm import tqdm

class TrainedTrackerContainer:
    def __init__(self, context):
        self.frame_index = context["frameIndex"]
        self.frames_count = context["frames"]

        self.data_path = os.path.join(g.my_app.data_dir, 'data', context["trackId"])

        self.frames_path = os.path.join(self.data_path, 'frames')
        self.annotations_path = os.path.join(self.data_path, 'annotations')

        self.track_id = context["trackId"]
        self.video_id = context["videoId"]
        self.video_info = g.api.video.get_info_by_id(self.video_id)
        self.video_fps = round(1 / self.video_info.frames_to_timecodes[1])

        self.direction = context["direction"]

        self.object_ids = context['objectIds']
        # self.class_name = g.api.video.object.get_info_by_id(self.object_ids[0].name)
        self.class_name = 'lemon'

        self.geometries = []
        self.frames_indexes = []

        self.add_frames_indexes()

        self.progress_notify_interval = 1 if round(len(self.frames_indexes) * 0.03) == 0 else \
            round(len(self.frames_indexes) * 0.03)

        self.video_annotator_progress = partial(g.api.video.notify_progress,
                                                track_id=self.track_id,
                                                video_id=self.video_id,
                                                frame_start=self.frames_indexes[0],
                                                frame_end=self.frames_indexes[-1],
                                                total=len(self.frames_indexes) - 1)

        g.logger.info(f'TrackerController Initialized')

    def add_frames_indexes(self):
        total_frames = self.video_info.frames_count
        cur_index = self.frame_index

        while 0 <= cur_index < total_frames and len(self.frames_indexes) < self.frames_count + 1:
            self.frames_indexes.append(cur_index)
            cur_index += (1 if self.direction == 'forward' else -1)

    def update_progress(self, enumerate_frame_index):
        frame_index = self.frames_indexes[enumerate_frame_index]

        if enumerate_frame_index % self.progress_notify_interval == 0:
            need_stop = self.video_annotator_progress(current=enumerate_frame_index)

            if need_stop:
                g.logger.debug('Tracking was stopped', extra={'track_id': self.track_id})
                return -2

        g.logger.info(f'Process frame {enumerate_frame_index} — {frame_index}')
        g.logger.info(f'Tracking completed')

        return 0

    def download_frames(self):
        os.makedirs(self.frames_path, exist_ok=True)
        sly.fs.clean_dir(self.frames_path)

        for index, frame_index in tqdm(enumerate(self.frames_indexes), desc='download frames',
                                       total=len(self.frames_indexes)):
            img_bgr = sly_functions.get_frame_np(g.api, self.video_id, frame_index)
            cv2.imwrite(f"{self.frames_path}/frame{index:06d}.jpg", img_bgr)  # save frame as JPEG file

        video_info = g.api.video.get_info_by_id(self.video_id)
        video_fps = round(1 / video_info.frames_to_timecodes[1])

        video_data = {'id': self.video_id, 'path': self.frames_path,
                      'fps': video_fps, 'origin_path': None}

        return video_data

    def dump_annotations(self, annotations):
        os.makedirs(self.annotations_path, exist_ok=True)
        sly.fs.clean_dir(self.annotations_path)

        sorted_tuples = sorted(annotations.items(), key=lambda item: item[0])
        annotations = {k: v for k, v in sorted_tuples}

        for frame_name, annotation in annotations.items():
            with open(os.path.join(self.annotations_path, f'{frame_name}.json'), 'w') as file:
                json.dump(annotation, file)

    def init_opts(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--device', type=str,
                            default=g.device, help='device to process')

        parser.add_argument('--source_path', type=str,
                            default=self.data_path, help='source_path')

        parser.add_argument('--nms_max_overlap', type=float, default=1.0,
                            help='Non-maxima suppression threshold: Maximum detection overlap.')
        parser.add_argument('--max_cosine_distance', type=float, default=0.4,
                            help='Gating threshold for cosine distance metric (object appearance).')
        parser.add_argument('--nn_budget', type=int, default=None,
                            help='Maximum size of the appearance descriptors allery. If None, no budget is enforced.')

        parser.add_argument('--thickness', type=int,
                            default=1, help='Thickness of the bounding box strokes')

        parser.add_argument('--info', action='store_true',
                            help='Print debugging info.')
        opt = parser.parse_args()

        opt.frame_indexes = self.frames_indexes

        return opt

    def init_annotation_keeper(self, ann_data):
        class_name = self.class_name
        objects_count = sly_functions.get_objects_count(ann_data)
        video_shape = sly_functions.get_video_shape(self.frames_path)
        video_frames_count = self.video_info.frames_count

        ann_keeper = serve_ann_keeper.AnnotationKeeper(video_shape=(video_shape[1], video_shape[0]),
                                                       objects_count=objects_count,
                                                       class_name=class_name,
                                                       video_frames_count=video_frames_count)

        return ann_keeper

    def track(self):
        opt = self.init_opts()
        ann_data = sly_tracker.track(opt)

        ann_keeper = self.init_annotation_keeper(ann_data)
        ann_keeper.add_figures_by_frames(ann_data)

        return ann_keeper.get_annotations()






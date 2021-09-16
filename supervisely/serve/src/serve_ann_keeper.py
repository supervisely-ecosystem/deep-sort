from serve_globals import *

# from supervisely_lib.video_annotation.key_id_map import KeyIdMap

from functools import partial

import sly_functions


class AnnotationKeeper:
    def __init__(self, video_shape, class_names_for_each_object, video_frames_count):
        self.video_frames_count = video_frames_count

        self.video_shape = video_shape

        self.class_names = class_names_for_each_object

        self.project = None
        self.dataset = None
        self.meta = None

        # self.key_id_map = KeyIdMap()
        self.sly_objects_list = []
        self.video_object_list = []

        self.get_sly_objects()
        self.get_video_objects_list()

        self.video_object_collection = sly.VideoObjectCollection(self.video_object_list)

        self.figures = []
        self.frames_list = []
        self.frames_collection = []

    def add_figures_by_frames(self, data):
        for frame_index, frame_data in data.items():
            if len(frame_data['ids']) > 0:
                self.add_figures_by_frame(coords_data=frame_data['coords'],
                                          objects_indexes=frame_data['ids'],
                                          frame_index=frame_index)

    def add_figures_by_frame(self, coords_data, objects_indexes, frame_index):
        temp_figures = []

        for i in range(len(coords_data)):
            figure = sly.VideoFigure(self.video_object_list[objects_indexes[i]],
                                     coords_data[i], frame_index)

            temp_figures.append(figure)

        self.figures.append(temp_figures)

    def get_annotations(self):
        self.get_frames_list()
        self.frames_collection = sly.FrameCollection(self.frames_list)

        video_annotation = sly.VideoAnnotation(self.video_shape, self.video_frames_count,
                                               self.video_object_collection, self.frames_collection)

        return video_annotation

    def get_unique_objects(self, obj_list):
        unique_objects = []
        for obj in obj_list:
            # @TODO: to add different types shapes
            if obj.name not in [temp_object.name for temp_object in unique_objects]:
                unique_objects.append(obj)

        return unique_objects

    def get_sly_objects(self):
        for class_name in self.class_names:
            self.sly_objects_list.append(sly.ObjClass(class_name, sly.Rectangle))

    def get_video_objects_list(self):
        for sly_object in self.sly_objects_list:
            self.video_object_list.append(sly.VideoObject(sly_object))

    def get_frames_list(self):
        for index, figure in enumerate(self.figures):
            self.frames_list.append(sly.Frame(figure[0].frame_index, figure))

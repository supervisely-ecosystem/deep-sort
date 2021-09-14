import clip
import cv2

import torch

import numpy as np

from utils.torch_utils import select_device
from utils.supervisely import convert_annotation

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_clip_detections as gdet

# sly imports
import serve_globals as g
import supervisely_lib as sly
from tqdm import tqdm


def correct_figure(img_size, figure):  # img_size — height, width tuple
    # check figure is within image bounds
    canvas_rect = sly.Rectangle.from_size(img_size)
    if canvas_rect.contains(figure.to_bbox()) is False:
        # crop figure
        figures_after_crop = [cropped_figure for cropped_figure in figure.crop(canvas_rect)]
        if len(figures_after_crop) > 0:
            return figures_after_crop[0]
        else:
            return None
    else:
        return figure


def update_track_data(tracks_data, tracks, frame_index, img_size):
    coordinates_data = []
    track_id_data = []

    for curr_track in tracks:
        if not curr_track.is_confirmed() or curr_track.time_since_update > 1:
            continue
        tl_br = curr_track.to_tlbr()  # (!!!) LTRB
        # class_name = curr_track.class_num
        track_id = curr_track.track_id - 1  # tracks in deepsort started from 1
        # bbox = xyxy

        potential_rectangle = sly.Rectangle(top=int(round(tl_br[1])),
                                            left=int(round(tl_br[0])),
                                            bottom=int(round(tl_br[3])),
                                            right=int(round(tl_br[2])))

        tested_rectangle = correct_figure(img_size, potential_rectangle)
        if tested_rectangle:
            coordinates_data.append(tested_rectangle)
            track_id_data.append(track_id)

    tracks_data[frame_index] = {'coords': coordinates_data,
                                'ids': track_id_data}

    return tracks_data


def track(opt):
    tracks_data = {}

    device = select_device(opt.device)

    nms_max_overlap = opt.nms_max_overlap
    max_cosine_distance = opt.max_cosine_distance
    nn_budget = opt.nn_budget
    frame_index_mapping = opt.frame_indexes

    model_filename = "ViT-B/32"  # initialize deep sort
    model, transform = clip.load(model_filename, device=device)
    encoder = gdet.create_box_encoder(model, transform, batch_size=1, device=device)

    metric = nn_matching.NearestNeighborDistanceMetric(  # calculate cosine distance metric
        "cosine", max_cosine_distance, nn_budget)

    tracker = Tracker(metric)  # initialize tracker

    source_path = opt.source_path

    image_paths = sorted(g.get_files_paths(source_path, ['.png', '.jpg']))
    annotations_path = sorted(g.get_files_paths(source_path, ['.json']))

    frame_index = 0
    for image_path, annotation_path in tqdm(zip(image_paths, annotations_path), desc='deepsort tracking',
                                            total=len(image_paths)):  # @TODO: integrate our ann

        im0 = cv2.imread(image_path)

        pred, classes = convert_annotation(annotation_path)
        pred = [torch.tensor(pred)]

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det) > 0:
                trans_bboxes = det[:, :4].clone()
                bboxes = trans_bboxes[:, :4].cpu()
                confs = det[:, 4]

                # encode yolo detections and feed to tracker
                features = encoder(im0, bboxes)
                detections = [Detection(bbox, conf, class_num, feature) for bbox, conf, class_num, feature in zip(
                    bboxes, confs, classes, features)]

                # run non-maxima supression
                boxs = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                class_nums = np.array([d.class_num for d in detections])
                indices = preprocessing.non_max_suppression(
                    boxs, class_nums, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]

                # Call the tracker
                tracker.predict()
                tracker.update(detections)

                update_track_data(tracks_data=tracks_data,
                                  tracks=tracker.tracks,
                                  frame_index=frame_index_mapping[frame_index],
                                  img_size=im0.shape[:2])

        frame_index += 1

    return tracks_data

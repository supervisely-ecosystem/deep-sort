import argparse
import os

import clip
import cv2

import torch

import numpy as np

from utils.general import xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import select_device
from utils.supervisely import convert_annotation

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_clip_detections as gdet

# sly imports
import sly_globals as g
from tqdm import tqdm


def update_tracks(tracker, frame_count, txt_path, img_path, im0, gn):
    # if len(tracker.tracks):
    #     print("[Tracks]", len(tracker.tracks))

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        xyxy = track.to_tlbr()
        class_num = track.class_num
        bbox = xyxy

        # if opt.info:
        #     print("Tracker ID: {}, Class: {}, BBox Coords (xmin, ymin, xmax, ymax): {}".format(
        #         str(track.track_id), class_num, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        if txt_path:  # Write to file
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                              ) / gn).view(-1).tolist()  # normalized xywh

            with open(txt_path, 'a') as f:
                f.write('frame: {}; track: {}; class: {}; bbox: {};\n'.format(frame_count, track.track_id, class_num,
                                                                              *xywh))

        if img_path is not None:  # Add bbox to image
            label = f'{class_num} #{track.track_id}'
            plot_one_box(xyxy, im0, label=label,
                         color=get_color_for(label), line_thickness=opt.thickness)

            save_image_path = os.path.join(img_path, f'{frame_count:06}.jpg')

            cv2.imwrite(save_image_path, im0)


def get_color_for(class_num):
    colors = [
        "#4892EA",
        "#00EEC3",
        "#FE4EF0",
        "#F4004E",
        "#FA7200",
        "#EEEE17",
        "#90FF00",
        "#78C1D2",
        "#8C29FF"
    ]

    num = hash(class_num)  # may actually be a number or a string
    hex = colors[num % len(colors)]

    # adapted from https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
    rgb = tuple(int(hex.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))

    return rgb


def detect():
    device = select_device(opt.device)

    nms_max_overlap = opt.nms_max_overlap
    max_cosine_distance = opt.max_cosine_distance
    nn_budget = opt.nn_budget

    model_filename = "ViT-B/32"  # initialize deep sort
    model, transform = clip.load(model_filename, device=device)
    encoder = gdet.create_box_encoder(model, transform, batch_size=1, device=device)

    metric = nn_matching.NearestNeighborDistanceMetric(  # calculate cosine distance metric
        "cosine", max_cosine_distance, nn_budget)

    tracker = Tracker(metric)  # initialize tracker

    source_path, output_path = opt.source_path, opt.output_path

    save_txt_path, save_img_path = os.path.join(output_path, 'tracks.txt'), \
                                   os.path.join(output_path, 'frames'),

    os.makedirs(save_img_path, exist_ok=True)

    image_paths = sorted(g.get_files_paths(source_path, ['.png', '.jpg']))
    annotations_path = sorted(g.get_files_paths(source_path, ['.json']))

    frame_count = 0
    for image_path, annotation_path in tqdm(zip(image_paths, annotations_path), total=len(image_paths)):  # @TODO: integrate our ann

        im0 = cv2.imread(image_path)

        pred, classes = convert_annotation(annotation_path)
        pred = [torch.tensor(pred)]

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # @TODO: here need to be OpenCV img

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

                # update tracks
                update_tracks(tracker=tracker,
                              frame_count=frame_count,
                              txt_path=save_txt_path,
                              img_path=save_img_path,
                              im0=im0,
                              gn=gn)

            frame_count += 1

    print(f'Done.')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_path', type=str,
                        default='./test', help='source_path')

    parser.add_argument('--output_path', type=str,
                        default='./output', help='output_path')

    parser.add_argument('--device', type=str,
                        default='cpu', help='source_path')

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
    print(opt)

    detect()

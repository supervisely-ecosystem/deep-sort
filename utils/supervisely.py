import json


def convert_annotation(annotation_path):

    with open(annotation_path, 'r') as sly_ann_data:
        ann_data_json = json.load(sly_ann_data)

    predictions = ann_data_json["objects"]
    formatted_predictions = []
    classes = []

    for prediction in predictions:
        points = prediction['points']['exterior']
        formatted_pred = [points[0][0], points[0][1], points[1][0], points[1][1], prediction['tags'][0]["value"]]


        # convert to width-height from bottom-right
        formatted_pred[2] -= formatted_pred[0]
        formatted_pred[3] -= formatted_pred[1]

        formatted_predictions.append(formatted_pred)
        classes.append(prediction["classTitle"])

    #print(formatted_predictions)

    return formatted_predictions, classes

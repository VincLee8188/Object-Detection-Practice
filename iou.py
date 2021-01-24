import torch


def intersection_over_union(boxes_preds, boxes_labels, box_format='corners'):
    """

    :param boxes_preds: tensor, prediction of bbox (N, 4)
    :param boxes_labels: tensor, ground truth of bbox (N, 4)
    :param box_format: str, midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2).
    :return:
    """
    if box_format == 'midpoint':
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 2:3] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 3:4] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 2:3] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 3:4] + boxes_labels[..., 3:4] / 2
    elif box_format == 'corners':
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    else:
        raise ValueError('Format error')

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1 = (box1_x2 - box2_x1) * (box1_y2 - box1_y1)
    area2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    return intersection / (area1 + area2 - intersection + 1e-6)


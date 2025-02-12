import numpy as np


def nms(boxes, scores, iou_threshold):
    """
    Perform non-maximum suppression on the bounding boxes.

    This function takes in a set of bounding boxes and their corresponding scores,
    and returns a list of indices of the boxes that should be kept.

    The basic idea is to pick the box with the highest score, and then remove all
    the boxes that have IoU with the picked box over the threshold. This is
    repeated until there are no more boxes left.

    :param boxes: A 2D NumPy array of shape (N, 4) where N is the number of boxes.
                  Each row represents a box with its xmin, ymin, xmax, ymax.
    :param scores: A 1D NumPy array of shape (N,) representing the scores of each
                   box.
    :param iou_threshold: A float representing the IoU threshold. Any box with
                          IoU over this threshold will be removed.
    :return: A list of indices of the boxes that should be kept.
    """
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box, boxes):
    """
    Compute the intersection over union (IoU) of the given box with the given
    boxes.

    The IoU is computed as the area of intersection divided by the area of
    union.

    :param box: A 1D NumPy array of shape (4,) representing the bounding box
                with its xmin, ymin, xmax, ymax.
    :param boxes: A 2D NumPy array of shape (N, 4) representing the bounding
                  boxes with their xmin, ymin, xmax, ymax.
    :return: A 1D NumPy array of shape (N,) representing the IoU of the given
             box with the given boxes.
    """
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y
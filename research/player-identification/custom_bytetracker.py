import numpy as np
from scipy.optimize import linear_sum_assignment
import supervision as sv

# Custom function to calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

# Custom tracker logic
class CustomByteTrack(sv.ByteTrack):
    def __init__(self, iou_threshold=0.3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iou_threshold = iou_threshold
        self.previous_detections = {}

    def update_with_custom_logic(self, detections):
        current_detections = detections.xyxy
        current_ids = []

        if self.previous_detections:
            # Calculate IoU matrix
            iou_matrix = np.zeros((len(self.previous_detections), len(current_detections)))
            for i, prev_box in enumerate(self.previous_detections.values()):
                for j, curr_box in enumerate(current_detections):
                    iou_matrix[i, j] = calculate_iou(prev_box, curr_box)

            # Solve assignment problem
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            # Update IDs based on IoU
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    current_ids.append(list(self.previous_detections.keys())[r])
                else:
                    current_ids.append(None)

        # Assign new IDs to unmatched detections
        next_id = max(self.previous_detections.keys(), default=0) + 1
        for i, box in enumerate(current_detections):
            if i >= len(current_ids) or current_ids[i] is None:
                current_ids.append(next_id)
                next_id += 1

        # Update previous detections
        self.previous_detections = {id_: box for id_, box in zip(current_ids, current_detections)}

        # Return updated detections with IDs
        detections.tracker_id = np.array(current_ids)
        return detections

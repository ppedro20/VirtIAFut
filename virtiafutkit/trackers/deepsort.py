import 

class DeepSort:
    """
    Initialize the DeepSort object.

    Parameters:
        max_dist (float, optional): Maximum cosine distance for appearance matching.
        min_confidence (float, optional): Minimum detection confidence for track activation.
        nms_max_overlap (float, optional): Maximum allowed overlap for NMS.
        max_iou_distance (float, optional): Maximum IOU distance for matching.
        max_age (int, optional): Maximum number of missed misses before a track is deleted.
        n_init (int, optional): Number of consecutive detections before track is confirmed.
    """

    def __init__(
        self,
        max_dist: float = 0.2,
        min_confidence: float = 0.3,
        nms_max_overlap: float = 0.7,
        max_iou_distance: float = 0.7,
        max_age: int = 70,
        n_init: int = 3,
    ):
        self.max_dist = max_dist
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.frame_id = 0
        self.tracker = DeepSortTracker(
            max_dist=self.max_dist,
            max_iou_distance=self.max_iou_distance,
            max_age=self.max_age,
            n_init=self.n_init,
        )

    def update_with_detections(self, detections: Detections) -> Detections:
        """
        Updates the tracker with the provided detections and returns the updated
        detection results.

        Args:
            detections (Detections): The detections to pass through the tracker.

        Example:
            ```python
            import supervision as sv
            from ultralytics import YOLO
            import numpy as np

            model = YOLO(<MODEL_PATH>)
            tracker = sv.DeepSort()

            bounding_box_annotator = sv.BoundingBoxAnnotator()
            label_annotator = sv.LabelAnnotator()

            def callback(frame: np.ndarray, index: int) -> np.ndarray:
                results = model(frame)[0]
                detections = sv.Detections.from_ultralytics(results)
                detections = tracker.update_with_detections(detections)

                labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

                annotated_frame = bounding_box_annotator.annotate(
                    scene=frame.copy(), detections=detections)
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame, detections=detections, labels=labels)
                return annotated_frame

            sv.process_video(
                source_path=<SOURCE_VIDEO_PATH>,
                target_path=<TARGET_VIDEO_PATH>,
                callback=callback
            )
            ```
        """

        tracks = self.update_with_tensors(
            tensors=detections2boxes(detections=detections)
        )
        detections = Detections.empty()
        if len(tracks) > 0:
            detections.xyxy = np.array(
                [track.to_tlbr() for track in tracks], dtype=np.float32
            )
            detections.class_id = np.array(
                [int(track.class_id) for track in tracks], dtype=int
            )
            detections.tracker_id = np.array(
                [int(track.track_id) for track in tracks], dtype=int
            )
            detections.confidence = np.array(
                [track.score for track in tracks], dtype=np.float32
            )
        else:
            detections.tracker_id = np.array([], dtype=int)

        return detections

    def update_with_tensors(self, tensors: np.ndarray) -> list:
        """
        Updates the tracker with the provided tensors and returns the updated tracks.

        Parameters:
            tensors: The new tensors to update with.

        Returns:
            list: Updated tracks.
        """
        self.frame_id += 1

        # tensors: [x1, y1, x2, y2, score, class_id, ...]
        bboxes = tensors[:, :4]
        scores = tensors[:, 4]
        class_ids = tensors[:, 5]

        # Filter detections by confidence
        mask = scores > self.min_confidence
        bboxes = bboxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # Optionally, apply NMS here if needed

        features = extract_features(bboxes)  # You need to implement or import this

        detections = [
            DeepSortDetection(bbox, score, class_id, feature)
            for bbox, score, class_id, feature in zip(bboxes, scores, class_ids, features)
        ]

        self.tracker.predict()
        self.tracker.update(detections)

        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            tracks.append(track)

        return tracks
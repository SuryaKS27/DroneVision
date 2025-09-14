# vakt_tracker.py

import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2

def iou_batch(bb_test, bb_gt):
    """
    Computes IoU between two sets of bounding boxes.
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o

class KalmanFilterWrapper:
    """
    A simple Kalman filter wrapper for object tracking.
    Models object motion as a constant velocity process.
    State is [cx, cy, w, h, vx, vy]
    """
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 4)
        # State transition matrix (F)
        self.kf.transitionMatrix = np.array([[1,0,0,0,1,0],
                                            [0,1,0,0,0,1],
                                            [0,0,1,0,0,0],
                                            [0,0,0,1,0,0],
                                            [0,0,0,0,1,0],
                                            [0,0,0,0,0,1]], np.float32)
        # Measurement matrix (H)
        self.kf.measurementMatrix = np.array([[1,0,0,0,0,0],
                                             [0,1,0,0,0,0],
                                             [0,0,1,0,0,0],
                                             [0,0,0,1,0,0]], np.float32)
        # Process noise covariance (Q)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.1
        self.kf.processNoiseCov[4:, 4:] *= 1.0 # Higher noise for velocity
        self.kf.processNoiseCov[2:4, 2:4] *= 0.01 # Lower noise for size
        # Measurement noise covariance (R)
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0

    def init(self, bbox):
        """Initialize the filter with the first detection."""
        center_x = bbox[0] + bbox[2] / 2
        center_y = bbox[1] + bbox[3] / 2
        self.kf.statePost = np.array([center_x, center_y, bbox[2], bbox[3], 0, 0], dtype=np.float32)
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 1.0

    def predict(self):
        """Predict the next state."""
        return self.kf.predict()

    def update(self, bbox):
        """Update the filter with a new measurement."""
        center_x = bbox[0] + bbox[2] / 2
        center_y = bbox[1] + bbox[3] / 2
        measurement = np.array([center_x, center_y, bbox[2], bbox[3]], dtype=np.float32)
        self.kf.correct(measurement)
        
    @property
    def state(self):
        return self.kf.statePost

def get_color_hist(frame, bbox):
    """Computes a 3D color histogram for the region defined by bbox."""
    x1, y1, x2, y2 = map(int, bbox)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros(16*16*16, dtype=np.float32)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # 16 bins for each channel
    hist = cv2.calcHist([hsv_roi], [0, 1, 2], None, [16, 16, 16], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

class VAKTTracker:
    track_id_counter = 0

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3, appearance_lambda=0.8):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.appearance_lambda = appearance_lambda
        self.tracks = []

    def update(self, detections, class_ids, scores, frame):
        # Detections are expected as [x1, y1, x2, y2]
        
        # 1. Predict new locations for existing tracks
        for track in self.tracks:
            track['kalman_filter'].predict()
            track['age'] += 1
            track['time_since_update'] += 1

        # 2. Compute cost matrix
        if len(self.tracks) > 0 and len(detections) > 0:
            # Get predicted bboxes from tracks
            predicted_bboxes = []
            for track in self.tracks:
                state = track['kalman_filter'].state
                predicted_bboxes.append([state[0]-state[2]/2, state[1]-state[3]/2, 
                                         state[0]+state[2]/2, state[1]+state[3]/2])
            predicted_bboxes = np.array(predicted_bboxes)

            # --- Motion Cost (IoU) ---
            iou_matrix = iou_batch(detections, predicted_bboxes)
            motion_cost = 1 - iou_matrix

            # --- Appearance Cost (Color Histogram) ---
            appearance_cost = np.zeros_like(motion_cost)
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    det_hist = get_color_hist(frame, det)
                    # Bhattacharyya distance for comparing histograms
                    dist = cv2.compareHist(track['color_hist'], det_hist, cv2.HISTCMP_BHATTACHARYYA)
                    appearance_cost[j, i] = dist # Note: scipy minimizes, so lower distance is better

            # --- Combined Cost ---
            combined_cost = self.appearance_lambda * appearance_cost + (1 - self.appearance_lambda) * motion_cost
        else:
            combined_cost = np.empty((len(detections), 0))

        # 3. Hungarian algorithm for assignment
        row_ind, col_ind = linear_sum_assignment(combined_cost)

        # 4. Update matched tracks
        matched_indices = []
        for r, c in zip(row_ind, col_ind):
            if combined_cost[r, c] < (1 - self.iou_threshold): # Apply a gating threshold
                track = self.tracks[c]
                det_bbox = detections[r]
                w, h = det_bbox[2] - det_bbox[0], det_bbox[3] - det_bbox[1]
                
                track['kalman_filter'].update(np.array([det_bbox[0], det_bbox[1], w, h]))
                track['bbox'] = det_bbox
                track['hits'] += 1
                track['time_since_update'] = 0
                track['score'] = scores[r]
                track['class_id'] = class_ids[r]
                
                # Slowly update color histogram to adapt to changes
                new_hist = get_color_hist(frame, det_bbox)
                track['color_hist'] = 0.8 * track['color_hist'] + 0.2 * new_hist

                matched_indices.append(r)

        # 5. Create new tracks for unmatched detections
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_indices]
        for i in unmatched_detections:
            det_bbox = detections[i]
            w, h = det_bbox[2] - det_bbox[0], det_bbox[3] - det_bbox[1]
            kf = KalmanFilterWrapper()
            kf.init(np.array([det_bbox[0], det_bbox[1], w, h]))

            new_track = {
                'id': VAKTTracker.track_id_counter,
                'kalman_filter': kf,
                'bbox': det_bbox,
                'class_id': class_ids[i],
                'score': scores[i],
                'hits': 1,
                'age': 1,
                'time_since_update': 0,
                'color_hist': get_color_hist(frame, det_bbox)
            }
            self.tracks.append(new_track)
            VAKTTracker.track_id_counter += 1

        # 6. Clean up old tracks and return confirmed tracks
        self.tracks = [t for t in self.tracks if t['time_since_update'] <= self.max_age]
        
        confirmed_tracks = []
        for track in self.tracks:
            if track['hits'] >= self.min_hits and track['time_since_update'] == 0:
                confirmed_tracks.append({
                    'id': track['id'],
                    'bbox': track['bbox'],
                    'class_id': track['class_id'],
                    'score': track['score']
                })

        return confirmed_tracks
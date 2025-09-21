# tools/deep_vakt_tracker.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque
import cv2

# We will use cosine distance for comparing Re-ID features
def cosine_distance(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)

class KalmanFilterWrapper: # Same as before
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 4)
        self.kf.transitionMatrix = np.array([[1,0,0,0,1,0],[0,1,0,0,0,1],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0]], np.float32)
    def init(self, bbox):
        center_x, center_y, w, h = bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2, bbox[2]-bbox[0], bbox[3]-bbox[1]
        self.kf.statePost = np.array([center_x, center_y, w, h, 0, 0], dtype=np.float32)
    def predict(self):
        state = self.kf.predict()
        return np.array([state[0]-state[2]/2, state[1]-state[3]/2, state[0]+state[2]/2, state[1]+state[3]/2]).reshape(4)
    def update(self, bbox):
        center_x, center_y, w, h = bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2, bbox[2]-bbox[0], bbox[3]-bbox[1]
        self.kf.correct(np.array([center_x, center_y, w, h], dtype=np.float32))

class DeepVAKTTracker:
    track_id_counter = 0
    def __init__(self, max_age=70, min_hits=3, iou_threshold=0.5, max_cosine_distance=0.2):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_cosine_distance = max_cosine_distance
        self.tracks = []

    def update(self, detections, class_ids, scores, embeddings):
        # 1. Predict new locations for existing tracks
        for t in self.tracks:
            t['bbox'] = t['kalman_filter'].predict()
            t['age'] += 1
            t['time_since_update'] += 1

        # 2. Separate tracks into confirmed and unconfirmed/lost
        confirmed_tracks = [t for t in self.tracks if t['hits'] >= self.min_hits and t['time_since_update'] == 1]
        unconfirmed_tracks = [t for t in self.tracks if t not in confirmed_tracks]

        # 3. Associate confirmed tracks with detections using appearance and motion
        if len(confirmed_tracks) > 0 and len(detections) > 0:
            track_features = np.array([t['features'][-1] for t in confirmed_tracks])
            cost_matrix = cosine_distance(embeddings, track_features)
            
            # Gating: remove matches that are too far apart in appearance
            cost_matrix[cost_matrix > self.max_cosine_distance] = 1e+5
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Update matched tracks
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 1e+5:
                    track = confirmed_tracks[c]
                    track['bbox'] = detections[r]
                    track['kalman_filter'].update(detections[r])
                    track['features'].append(embeddings[r])
                    track['hits'] += 1
                    track['time_since_update'] = 0
                    track['matched'] = True
        
        # 4. Handle remaining tracks and detections (simplified IoU matching for leftovers)
        remaining_dets = [i for i, d in enumerate(detections) if i not in [r for r,c in zip(row_ind, col_ind) if cost_matrix[r,c] < 1e+5]]
        remaining_tracks = [t for t in unconfirmed_tracks if not t.get('matched', False)]
        
        if len(remaining_dets) > 0 and len(remaining_tracks) > 0:
            det_boxes = detections[remaining_dets]
            track_boxes = np.array([t['bbox'] for t in remaining_tracks])
            iou_matrix = 1 - self._iou_batch(det_boxes, track_boxes)
            iou_matrix[iou_matrix > (1 - self.iou_threshold)] = 1e+5
            
            row_ind_iou, col_ind_iou = linear_sum_assignment(iou_matrix)
            for r_idx, c_idx in zip(row_ind_iou, col_ind_iou):
                if iou_matrix[r_idx, c_idx] < 1e+5:
                    r = remaining_dets[r_idx]
                    track = remaining_tracks[c_idx]
                    track['bbox'] = detections[r]
                    track['kalman_filter'].update(detections[r])
                    track['features'].append(embeddings[r])
                    track['hits'] += 1
                    track['time_since_update'] = 0
                    track['matched'] = True

        # 5. Create new tracks for unmatched detections
        final_unmatched_dets = [i for i, d in enumerate(detections) if not any(np.array_equal(d, t['bbox']) for t in self.tracks if t.get('matched'))]
        for i in final_unmatched_dets:
            kf = KalmanFilterWrapper()
            kf.init(detections[i])
            new_track = {
                'id': DeepVAKTTracker.track_id_counter, 'kalman_filter': kf, 'bbox': detections[i],
                'class_id': class_ids[i], 'score': scores[i], 'hits': 1, 'age': 1,
                'time_since_update': 0, 'features': deque([embeddings[i]], maxlen=100), 'matched': True
            }
            self.tracks.append(new_track)
            DeepVAKTTracker.track_id_counter += 1

        # 6. Clean up: remove old tracks and reset match status
        self.tracks = [t for t in self.tracks if t['time_since_update'] <= self.max_age]
        
        output_tracks = []
        for track in self.tracks:
            if track['hits'] >= self.min_hits and track['time_since_update'] == 0:
                output_tracks.append({
                    'id': track['id'], 'bbox': track['bbox'],
                    'class_id': track['class_id'], 'score': track['score']
                })
            track['matched'] = False
            
        return output_tracks

    def _iou_batch(self, bb_test, bb_gt):
        bb_gt = np.expand_dims(bb_gt, 0); bb_test = np.expand_dims(bb_test, 1)
        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0]); yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2]); yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0., xx2 - xx1); h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
        return o

# tools/track_from_gt.py (Updated with new color logic)
import os
import cv2
import json
import numpy as np
import sys
import argparse
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

# Import your VAKT tracker
from vakt_tracker import VAKTTracker

# --- Constants and Helpers ---
CLASS_MAP = {
    1: 'person', 2: 'boat', 42: 'surfboard'
    # NOTE: Please verify these IDs against your JSON file's "categories" section.
}
# The random TRACK_COLORS array is no longer needed

def load_annotations(json_path, target_video_filename):
    """
    Loads annotations from a master JSON file, filtering for a specific video.
    """
    print(f"Loading annotations from master file: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'annotations' not in data or 'images' not in data:
        raise ValueError("Annotation file must contain 'images' and 'annotations' keys.")

    print(f"Searching for annotations matching video: {target_video_filename}...")
    
    target_image_ids = set()
    image_id_to_frame_index = {}
    for img in data['images']:
        if img.get('source', {}).get('video') == target_video_filename:
            target_image_ids.add(img['id'])
            image_id_to_frame_index[img['id']] = img['source']['frame_no']

    if not target_image_ids:
        raise ValueError(f"No images found for video '{target_video_filename}' in the JSON file.")

    annotations_by_frame = defaultdict(list)
    for ann in data['annotations']:
        if ann['image_id'] in target_image_ids:
            frame_idx = image_id_to_frame_index[ann['image_id']]
            annotations_by_frame[frame_idx].append(ann)
            
    print(f"Found and loaded annotations for {len(annotations_by_frame)} frames from the target video.")
    return annotations_by_frame

def draw_tracked_boxes(frame, tracks):
    """Draws tracked bounding boxes with IDs on the frame."""
    pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_im)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for track in tracks:
        bbox, track_id, class_id = track['bbox'], track['id'], track['class_id']
        class_name = CLASS_MAP.get(class_id, f'Class-{class_id}')
        
        # --- MODIFIED COLOR LOGIC ---
        if class_name == 'boat':
            color = (0, 0, 255)  # Blue for boats
        else:
            color = (255, 0, 0)  # Red for all other classes
        # --- END OF MODIFIED LOGIC ---

        label = f"ID:{track_id} {class_name}"
        
        draw.rectangle(bbox, outline=color, width=3)
        text_size = font.getbbox(label)
        text_w, text_h = text_size[2] - text_size[0], text_size[3] - text_size[1]
        text_bg = (bbox[0], bbox[1] - text_h - 4, bbox[0] + text_w + 4, bbox[1])
        draw.rectangle(text_bg, fill=color)
        draw.text((bbox[0] + 2, bbox[1] - text_h - 2), label, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

def run_on_video_with_gt(video_path, gt_annotations, out_dir):
    """Performs tracking on a video file using ground truth detections."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(out_dir, f"gt_tracked_{os.path.basename(video_path)}")
    out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    tracker = VAKTTracker(max_age=50, min_hits=3, iou_threshold=0.3, appearance_lambda=0.7)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        det_boxes, det_scores, det_labels = [], [], []
        if frame_count in gt_annotations:
            gt_dets_for_frame = gt_annotations[frame_count]
            for det in gt_dets_for_frame:
                x, y, w, h = det['bbox']
                det_boxes.append([x, y, x + w, y + h])
                det_scores.append(1.0)
                det_labels.append(det['category_id'])

        det_boxes = np.array(det_boxes)
        det_scores = np.array(det_scores)
        det_labels = np.array(det_labels)
        
        tracked_objects = tracker.update(det_boxes, det_labels, det_scores, frame)
        result_frame = draw_tracked_boxes(frame.copy(), tracked_objects)
        out.write(result_frame)
        
        print(f"Processing frame {frame_count}...", end='\r')
        frame_count += 1

    cap.release()
    out.release()
    print(f"\n✅ Saved ground truth tracked video to {out_path}")

def main():
    parser = argparse.ArgumentParser("VAKT Tracking from Ground Truth Annotations")
    parser.add_argument('-i', '--input-video', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('-a', '--annotations', type=str, required=True, help="Path to the master COCO-style JSON annotation file.")
    parser.add_argument('--out-dir', type=str, default="outputs_gt_tracked", help="Directory to save the output video.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    video_filename = os.path.basename(args.input_video)
    gt_annotations = load_annotations(args.annotations, video_filename)
    
    run_on_video_with_gt(args.input_video, gt_annotations, args.out_dir)

if __name__ == '__main__':
    main()

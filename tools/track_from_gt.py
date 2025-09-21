# tools/track_from_gt.py (Corrected for TypeError)
import os
import cv2
import json
import numpy as np
import sys
import argparse
import re
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

# Import your VAKT tracker
from vakt_tracker import VAKTTracker

# --- Constants and Helpers ---
CLASS_MAP = {
    1: 'person', 2: 'boat', 9: 'boat', 42: 'surfboard'
    # NOTE: Please verify against your JSON file's "categories" section.
}

def load_annotations(json_path, target_video_filename):
    """
    Loads annotations from a master JSON file, intelligently filtering for a specific video.
    """
    print(f"Loading annotations from master file: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'annotations' not in data or 'images' not in data:
        raise ValueError("Annotation file must contain 'images' and 'annotations' keys.")

    print(f"Searching for annotations matching video: {target_video_filename}...")
    target_video_basename = os.path.splitext(target_video_filename)[0]
    
    target_image_ids = set()
    image_id_to_frame_index = {}

    for img in data['images']:
        match = False
        if 'source' in img and 'video' in img['source']:
            if os.path.splitext(img['source']['video'])[0] == target_video_basename:
                match = True
                frame_idx = img['source'].get('frame_no', img.get('frame_index'))
        
        if not match and img['file_name'].startswith(target_video_basename):
            match = True
            frame_search = re.search(r'_(\d+)\.(png|jpg|jpeg)$', img['file_name'], re.IGNORECASE)
            if frame_search:
                frame_idx = int(frame_search.group(1))
        
        if match:
            if 'id' not in img: continue
            image_id = img['id']
            target_image_ids.add(image_id)
            if 'frame_idx' not in locals():
                frame_idx = img.get('frame_index', image_id)
            image_id_to_frame_index[image_id] = frame_idx

    if not target_image_ids:
        raise ValueError(f"Could not find any images for video '{target_video_filename}' in the JSON file.")

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
        # --- MODIFIED: Handle both dict and list/tuple format to prevent crash ---
        if isinstance(track, dict):
            # This is the expected format
            bbox = track.get('bbox')
            track_id = track.get('id')
            class_id = track.get('class_id')
        else:
            # This is a fallback for an unexpected list/tuple format
            # It assumes the order is [bbox, id, class_id, ...]
            try:
                bbox = track[0]
                track_id = track[1]
                class_id = track[2]
            except (IndexError, TypeError):
                print(f"Warning: Skipping malformed track object: {track}")
                continue # Skip to the next track if format is wrong
        # --- END OF MODIFICATION ---

        if bbox is None or track_id is None or class_id is None:
            continue # Skip if essential data is missing

        class_name = CLASS_MAP.get(class_id, f'Class-{class_id}')
        
        if class_name == 'boat':
            color = (0, 0, 255)  # Blue for boats
        else:
            color = (255, 0, 0)  # Red for all other classes

        label = f"ID:{track_id} {class_name}"
        draw.rectangle(bbox, outline=color, width=3)
        text_size = font.getbbox(label)
        text_w, text_h = text_size[2] - text_size[0], text_size[3] - text_size[1]
        text_bg = (bbox[0], bbox[1] - text_h - 4, bbox[0] + text_w + 4, bbox[1])
        draw.rectangle(text_bg, fill=color)
        draw.text((bbox[0] + 2, bbox[1] - text_h - 2), label, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)


def run_on_video_with_gt(video_path, gt_annotations, out_dir):
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
            for det in gt_annotations[frame_count]:
                x, y, w, h = det['bbox']
                det_boxes.append([x, y, x + w, y + h])
                det_scores.append(1.0)
                det_labels.append(det['category_id'])

        tracked_objects = tracker.update(np.array(det_boxes), np.array(det_labels), np.array(det_scores), frame)
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

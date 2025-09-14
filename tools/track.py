# # track.py

# import os
# import cv2
# import torch
# import torch.nn as nn
# import torchvision.transforms as T
# from torch.cuda.amp import autocast
# from PIL import Image, ImageDraw, ImageFont
# import numpy as np
# import sys
# import argparse
# import random

# # Make the project modules available
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# from src.core import YAMLConfig

# # Import our new tracker
# from vakt_tracker import VAKTTracker

# # --- Constants and Helpers ---

# # ✅ Assign specific colors for the 3 classes
# CLASS_MAP = {
#     0: 'swimmer',
#     1: 'swimmer with life jacket',
#     2: 'boat'
# }

# CLASS_COLORS = {
#     0: (50, 205, 50),   # swimmer -> LimeGreen
#     1: (255, 69, 0),    # swimmer with life jacket -> OrangeRed
#     2: (30, 144, 255)   # boat -> DodgerBlue
# }

# # Generate a color palette for track IDs
# np.random.seed(42)
# TRACK_COLORS = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)

# def draw_tracked_boxes(frame, tracks):
#     """Draws tracked bounding boxes with IDs on the frame."""
#     pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     draw = ImageDraw.Draw(pil_im)
#     font = ImageFont.load_default()

#     for track in tracks:
#         bbox = track['bbox']
#         track_id = track['id']
#         class_id = track['class_id']
#         score = track['score']
        
#         # Use a consistent color for each track ID
#         color = tuple(TRACK_COLORS[track_id % len(TRACK_COLORS)].tolist())
        
#         # Get class name and draw
#         class_name = CLASS_MAP.get(class_id, f'Class-{class_id}')
#         label = f"ID:{track_id} {class_name} {score:.2f}"
        
#         draw.rectangle(bbox, outline=color, width=3)
#         text_size = font.getbbox(label)
#         text_bg = (bbox[0], bbox[1] - (text_size[3] - text_size[1]) - 4, bbox[0] + (text_size[2] - text_size[0]), bbox[1])
#         draw.rectangle(text_bg, fill=color)
#         draw.text((bbox[0], bbox[1] - 12), label, fill=(255, 255, 255), font=font)

#     return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)


# def run_on_video(video_path, model, transforms, device, out_dir):
#     """Performs detection and tracking on a video file."""
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error opening video file: {video_path}")
#         return

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out_path = os.path.join(out_dir, f"tracked_{os.path.basename(video_path)}")
#     out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
#                           (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
#                            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

#     # Initialize our VAKT tracker
#     tracker = VAKTTracker(max_age=50, min_hits=3, iou_threshold=0.3, appearance_lambda=0.7)
    
#     frame_count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Convert frame for model input
#         im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         w, h = im_pil.size
#         orig_size = torch.tensor([w, h])[None].to(device)
#         im_data = transforms(im_pil)[None].to(device)

#         # Get detections from your RT-DETR model
#         with torch.no_grad(), autocast():
#             labels, boxes, scores = model(im_data, orig_size)
            
#         # Prepare detections for the tracker
#         # Filter detections by confidence threshold
#         thrh = 0.5
#         scr = scores[0]
#         valid_indices = scr > thrh
        
#         det_boxes = boxes[0][valid_indices].cpu().numpy()
#         det_scores = scr[valid_indices].cpu().numpy()
#         det_labels = labels[0][valid_indices].cpu().numpy().astype(int)

#         # Update tracker with new detections
#         tracked_objects = tracker.update(det_boxes, det_labels, det_scores, frame)
        
#         # Draw the results on the frame
#         result_frame = draw_tracked_boxes(frame.copy(), tracked_objects)

#         out.write(result_frame)
#         frame_count += 1
#         print(f"Processing frame {frame_count}...", end='\r')


#     cap.release()
#     out.release()
#     print(f"\n✅ Saved tracked video to {out_path}")


# # ✅ Main function to set up model and start processing
# def main(args):
#     cfg = YAMLConfig(args.config, resume=args.resume)
#     checkpoint = torch.load(args.resume, map_location='cpu')
#     state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    
#     # Load model weights, handling potential mismatches
#     model_state_dict = cfg.model.state_dict()
#     # Filter out unnecessary keys
#     state = {k: v for k, v in state.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
#     cfg.model.load_state_dict(state, strict=False)

#     class Model(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.model = cfg.model.deploy()
#             self.postprocessor = cfg.postprocessor.deploy()

#         def forward(self, images, orig_target_sizes):
#             outputs = self.model(images)
#             outputs = self.postprocessor(outputs, orig_target_sizes)
#             return outputs

#     model = Model().to(args.device).eval()

#     transforms = T.Compose([
#         T.Resize((640, 640)),
#         T.ToTensor(),
#     ])

#     os.makedirs(args.out_dir, exist_ok=True)

#     # Process a single video file or all videos in a folder
#     if os.path.isdir(args.input_path):
#         video_files = [f for f in os.listdir(args.input_path) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
#         for video_file in video_files:
#             file_path = os.path.join(args.input_path, video_file)
#             print(f"\n--- Processing video: {file_path} ---")
#             run_on_video(file_path, model, transforms, args.device, args.out_dir)
#     elif os.path.isfile(args.input_path):
#         print(f"--- Processing video: {args.input_path} ---")
#         run_on_video(args.input_path, model, transforms, args.device, args.out_dir)
#     else:
#         print(f"⚠️ Input path not found: {args.input_path}")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser("RT-DETR with VAKT Tracking")
#     parser.add_argument('-c', '--config', type=str, required=True, help="Path to the model config file.")
#     parser.add_argument('-r', '--resume', type=str, required=True, help="Path to the trained model checkpoint.")
#     parser.add_argument('-i', '--input-path', type=str, required=True, help="Path to a video file or a folder of videos.")
#     parser.add_argument('--out-dir', type=str, default="outputs_tracked", help="Directory to save the output videos.")
#     parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use (e.g., 'cpu', 'cuda').")
#     args = parser.parse_args()
    
#     main(args)

# track.py

import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.cuda.amp import autocast
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import sys
import argparse
import random

# Make the project modules available (adjust the path if needed)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig

# Import our new tracker
from vakt_tracker import VAKTTracker

# --- Constants and Helpers ---

# Class names and colors for visualization
CLASS_MAP = {
    0: 'swimmer',
    1: 'swimmer with life jacket',
    2: 'boat'
}

# Generate a consistent color palette for track IDs
np.random.seed(42)
TRACK_COLORS = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)

def save_results_to_file(tracked_objects, frame_count, file_path):
    """Saves tracking results to a text file in MOT Challenge format."""
    with open(file_path, 'a') as f:
        for track in tracked_objects:
            bbox = track['bbox']
            track_id = track['id']
            # Convert [x1, y1, x2, y2] to [x, y, w, h] for MOT format
            x, y = bbox[0], bbox[1]
            w, h = bbox[2] - x, bbox[3] - y
            # Format: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
            f.write(f"{frame_count},{track_id},{x},{y},{w},{h},{track['score']:.4f},-1,-1,-1\n")


def draw_tracked_boxes(frame, tracks):
    """Draws tracked bounding boxes with IDs on the frame."""
    pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_im)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for track in tracks:
        bbox = track['bbox']
        track_id = track['id']
        class_id = track['class_id']
        score = track['score']
        
        # Use a consistent color for each track ID
        color = tuple(TRACK_COLORS[track_id % len(TRACK_COLORS)].tolist())
        
        class_name = CLASS_MAP.get(class_id, f'Class-{class_id}')
        label = f"ID:{track_id} {class_name} {score:.2f}"
        
        draw.rectangle(bbox, outline=color, width=3)
        
        # Draw a filled rectangle for the text background
        text_size = font.getbbox(label)
        text_w = text_size[2] - text_size[0]
        text_h = text_size[3] - text_size[1]
        text_bg_coords = (bbox[0], bbox[1] - text_h - 4, bbox[0] + text_w + 4, bbox[1])
        draw.rectangle(text_bg_coords, fill=color)
        draw.text((bbox[0] + 2, bbox[1] - text_h - 2), label, fill=(255, 255, 255), font=font)

    # Corrected line
    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)


def run_on_video(video_path, model, transforms, device, out_dir):
    """Performs detection and tracking on a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video_path = os.path.join(out_dir, f"tracked_{os.path.basename(video_path)}")
    out = cv2.VideoWriter(out_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Path for the evaluation results file
    results_file_path = os.path.join(out_dir, "results.txt")
    # Clear previous results file if it exists
    if os.path.exists(results_file_path):
        os.remove(results_file_path)

    # Initialize our VAKT tracker with tuned parameters
    tracker = VAKTTracker(max_age=50, min_hits=3, iou_threshold=0.3, appearance_lambda=0.75)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert frame for model input
        im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(device)
        im_data = transforms(im_pil)[None].to(device)

        # Get detections from your RT-DETR model
        with torch.no_grad(), autocast():
            labels, boxes, scores = model(im_data, orig_size)
            
        # Filter detections by confidence threshold (e.g., 0.5)
        thrh = 0.5
        scr = scores[0]
        valid_indices = scr > thrh
        
        det_boxes = boxes[0][valid_indices].cpu().numpy()
        det_scores = scr[valid_indices].cpu().numpy()
        det_labels = labels[0][valid_indices].cpu().numpy().astype(int)

        # Update tracker with new detections for the current frame
        tracked_objects = tracker.update(det_boxes, det_labels, det_scores, frame)
        
        # Save the tracking results for this frame to the text file
        save_results_to_file(tracked_objects, frame_count, results_file_path)
        
        # Draw the results on the frame
        result_frame = draw_tracked_boxes(frame.copy(), tracked_objects)

        out.write(result_frame)
        print(f"Processing frame {frame_count}... Found {len(tracked_objects)} objects.", end='\r')

    cap.release()
    out.release()
    print(f"\n✅ Saved tracked video to {out_video_path}")
    print(f"✅ Saved evaluation data to {results_file_path}")


def main(args):
    """Main function to set up model and start processing."""
    cfg = YAMLConfig(args.config, resume=args.resume)
    checkpoint = torch.load(args.resume, map_location='cpu')
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    
    cfg.model.load_state_dict(state, strict=False)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device).eval()

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    os.makedirs(args.out_dir, exist_ok=True)

    if os.path.exists(args.input_path):
        run_on_video(args.input_path, model, transforms, args.device, args.out_dir)
    else:
        print(f"⚠️ Input video file not found: {args.input_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("RT-DETR with VAKT Tracking")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the model config file.")
    parser.add_argument('-r', '--resume', type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument('-i', '--input-path', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('--out-dir', type=str, default="outputs_tracked", help="Directory to save the output.")
    parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use (e.g., 'cpu', 'cuda').")
    args = parser.parse_args()
    
    main(args)

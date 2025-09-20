# Version 1 without visualization
# # track_images.py
# import os
# import cv2
# import torch
# import torch.nn as nn
# import torchvision.transforms as T
# from torch.cuda.amp import autocast
# from PIL import Image
# import numpy as np
# import sys
# import argparse
# from tqdm import tqdm

# # Make the project modules available (adjust path if needed)
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# from src.core import YAMLConfig

# # Import your tracker
# from vakt_tracker import VAKTTracker

# def save_results_to_file(tracked_objects, frame_count, file_path):
#     """Saves tracking results to a text file in MOT Challenge format."""
#     with open(file_path, 'a') as f:
#         for track in tracked_objects:
#             bbox = track['bbox']
#             track_id = track['id']
#             # Convert [x1, y1, x2, y2] to [x, y, w, h] for MOT format
#             x, y = bbox[0], bbox[1]
#             w, h = bbox[2] - x, bbox[3] - y
#             # Format: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,-1,-1,-1
#             f.write(f"{frame_count},{track_id},{x},{y},{w},{h},{track['score']:.4f},-1,-1,-1\n")

# def main(args):
#     """Main function to set up model and process an image sequence."""
#     cfg = YAMLConfig(args.config, resume=args.resume)
#     checkpoint = torch.load(args.resume, map_location='cpu')
#     state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
#     cfg.model.load_state_dict(state, strict=False)

#     class Model(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.model = cfg.model.deploy()
#             self.postprocessor = cfg.postprocessor.deploy()

#         def forward(self, images, orig_target_sizes):
#             outputs = self.model(images)
#             return self.postprocessor(outputs, orig_target_sizes)

#     model = Model().to(args.device).eval()

#     transforms = T.Compose([
#         T.Resize((640, 640)),
#         T.ToTensor(),
#     ])

#     os.makedirs(args.out_dir, exist_ok=True)
    
#     # --- Main Processing Loop ---
#     image_files = sorted([f for f in os.listdir(args.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
#     # Path for the evaluation results file
#     results_file_path = os.path.join(args.out_dir, "results.txt")
#     if os.path.exists(results_file_path):
#         os.remove(results_file_path)
        
#     tracker = VAKTTracker(max_age=50, min_hits=3, iou_threshold=0.3, appearance_lambda=0.75)
    
#     print(f"Processing {len(image_files)} images from {args.img_dir}...")
#     for frame_count, img_name in enumerate(tqdm(image_files), 1):
#         img_path = os.path.join(args.img_dir, img_name)
#         frame = cv2.imread(img_path)
#         if frame is None:
#             continue
            
#         im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         w, h = im_pil.size
#         orig_size = torch.tensor([w, h])[None].to(args.device)
#         im_data = transforms(im_pil)[None].to(args.device)

#         with torch.no_grad(), autocast():
#             labels, boxes, scores = model(im_data, orig_size)
            
#         thrh = 0.5
#         scr = scores[0]
#         valid_indices = scr > thrh
        
#         det_boxes = boxes[0][valid_indices].cpu().numpy()
#         det_scores = scr[valid_indices].cpu().numpy()
#         det_labels = labels[0][valid_indices].cpu().numpy().astype(int)

#         tracked_objects = tracker.update(det_boxes, det_labels, det_scores, frame)
        
#         save_results_to_file(tracked_objects, frame_count, results_file_path)

#     print(f"\n✅ Tracking complete. Results saved to {results_file_path}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser("RT-DETR with VAKT Tracking on Image Sequences")
#     parser.add_argument('-c', '--config', type=str, required=True, help="Path to the model config file.")
#     parser.add_argument('-r', '--resume', type=str, required=True, help="Path to the trained model checkpoint.")
#     parser.add_argument('--img-dir', type=str, required=True, help="Path to the folder of input images.")
#     parser.add_argument('--out-dir', type=str, default="outputs_tracked", help="Directory to save the results.txt file.")
#     parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use.")
#     args = parser.parse_args()

#     main(args)

# tools/track_images.py (Updated to save visualized images)
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
# from tqdm import tqdm

# # Make the project modules available (adjust path if needed)
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# from src.core import YAMLConfig

# # Import your tracker
# from vakt_tracker import VAKTTracker

# # --- Helper Functions ---

# CLASS_MAP = {0: 'swimmer', 1: 'swimmer with life jacket', 2: 'boat'}
# np.random.seed(42)
# TRACK_COLORS = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)

# def save_results_to_file(tracked_objects, frame_count, file_path):
#     """Saves tracking results to a text file in MOT Challenge format."""
#     with open(file_path, 'a') as f:
#         for track in tracked_objects:
#             bbox = track['bbox']
#             track_id = track['id']
#             x, y = bbox[0], bbox[1]
#             w, h = bbox[2] - x, bbox[3] - y
#             f.write(f"{frame_count},{track_id},{x},{y},{w},{h},{track['score']:.4f},-1,-1,-1\n")

# # NEW: Function to draw the bounding boxes on the image
# def draw_tracked_boxes(frame, tracks):
#     """Draws tracked bounding boxes with IDs on the frame."""
#     pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     draw = ImageDraw.Draw(pil_im)
#     try:
#         font = ImageFont.truetype("arial.ttf", 15)
#     except IOError:
#         font = ImageFont.load_default()

#     for track in tracks:
#         bbox = track['bbox']
#         track_id = track['id']
#         class_id = track['class_id']
#         score = track['score']
        
#         color = tuple(TRACK_COLORS[track_id % len(TRACK_COLORS)].tolist())
#         class_name = CLASS_MAP.get(class_id, f'Class-{class_id}')
#         label = f"ID:{track_id} {class_name} {score:.2f}"
        
#         draw.rectangle(bbox, outline=color, width=3)
#         text_size = font.getbbox(label)
#         text_w = text_size[2] - text_size[0]
#         text_h = text_size[3] - text_size[1]
#         text_bg_coords = (bbox[0], bbox[1] - text_h - 4, bbox[0] + text_w + 4, bbox[1])
#         draw.rectangle(text_bg_coords, fill=color)
#         draw.text((bbox[0] + 2, bbox[1] - text_h - 2), label, fill=(255, 255, 255), font=font)

#     return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)


# def main(args):
#     """Main function to set up model and process an image sequence."""
#     cfg = YAMLConfig(args.config, resume=args.resume)
#     checkpoint = torch.load(args.resume, map_location='cpu')
#     state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
#     cfg.model.load_state_dict(state, strict=False)

#     class Model(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.model = cfg.model.deploy()
#             self.postprocessor = cfg.postprocessor.deploy()

#         def forward(self, images, orig_target_sizes):
#             outputs = self.model(images)
#             return self.postprocessor(outputs, orig_target_sizes)

#     model = Model().to(args.device).eval()

#     transforms = T.Compose([
#         T.Resize((640, 640)),
#         T.ToTensor(),
#     ])

#     os.makedirs(args.out_dir, exist_ok=True)
#     # NEW: Create a sub-directory for the visualized images
#     visualized_output_dir = os.path.join(args.out_dir, "visualized")
#     os.makedirs(visualized_output_dir, exist_ok=True)
    
#     image_files = sorted([f for f in os.listdir(args.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
#     results_file_path = os.path.join(args.out_dir, "results.txt")
#     if os.path.exists(results_file_path):
#         os.remove(results_file_path)
        
#     tracker = VAKTTracker(max_age=50, min_hits=3, iou_threshold=0.3, appearance_lambda=0.75)
    
#     print(f"Processing {len(image_files)} images from {args.img_dir}...")
#     for frame_count, img_name in enumerate(tqdm(image_files), 1):
#         img_path = os.path.join(args.img_dir, img_name)
#         frame = cv2.imread(img_path)
#         if frame is None:
#             continue
            
#         im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         w, h = im_pil.size
#         orig_size = torch.tensor([w, h])[None].to(args.device)
#         im_data = transforms(im_pil)[None].to(args.device)

#         with torch.no_grad(), autocast():
#             labels, boxes, scores = model(im_data, orig_size)
            
#         thrh = 0.5
#         scr = scores[0]
#         valid_indices = scr > thrh
        
#         det_boxes = boxes[0][valid_indices].cpu().numpy()
#         det_scores = scr[valid_indices].cpu().numpy()
#         det_labels = labels[0][valid_indices].cpu().numpy().astype(int)

#         tracked_objects = tracker.update(det_boxes, det_labels, det_scores, frame)
        
#         # 1. Save results for evaluation
#         save_results_to_file(tracked_objects, frame_count, results_file_path)
        
#         # 2. NEW: Draw boxes and save the visualized image
#         result_frame = draw_tracked_boxes(frame.copy(), tracked_objects)
#         save_path = os.path.join(visualized_output_dir, img_name)
#         cv2.imwrite(save_path, result_frame)

#     print(f"\n✅ Tracking complete.")
#     print(f"  -> Evaluation data saved to: {results_file_path}")
#     print(f"  -> Visualized images saved to: {visualized_output_dir}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser("RT-DETR with VAKT Tracking on Image Sequences")
#     parser.add_argument('-c', '--config', type=str, required=True, help="Path to the model config file.")
#     parser.add_argument('-r', '--resume', type=str, required=True, help="Path to the trained model checkpoint.")
#     parser.add_argument('--img-dir', type=str, required=True, help="Path to the folder of input images.")
#     parser.add_argument('--out-dir', type=str, default="outputs_tracked", help="Directory to save the results.")
#     parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use.")
#     args = parser.parse_args()
#     main(args)


# Version 3 - Intermediate Results

# tools/track_images.py (Updated with Debug Mode)
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
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig
from vakt_tracker import VAKTTracker

# --- Helper Functions ---
CLASS_MAP = {0: 'swimmer', 1: 'swimmer with life jacket', 2: 'boat'}
np.random.seed(42)
TRACK_COLORS = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)

def save_results_to_file(tracked_objects, frame_count, file_path):
    with open(file_path, 'a') as f:
        for track in tracked_objects:
            bbox = track['bbox']
            track_id = track['id']
            x, y = bbox[0], bbox[1]
            w, h = bbox[2] - x, bbox[3] - y
            f.write(f"{frame_count},{track_id},{x},{y},{w},{h},{track['score']:.4f},-1,-1,-1\n")

def draw_tracked_boxes(frame, tracks):
    pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_im)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for track in tracks:
        bbox, track_id, class_id, score = track['bbox'], track['id'], track['class_id'], track['score']
        color = tuple(TRACK_COLORS[track_id % len(TRACK_COLORS)].tolist())
        class_name = CLASS_MAP.get(class_id, f'Class-{class_id}')
        label = f"ID:{track_id} {class_name} {score:.2f}"
        
        draw.rectangle(bbox, outline=color, width=3)
        text_size = font.getbbox(label)
        text_w, text_h = text_size[2] - text_size[0], text_size[3] - text_size[1]
        text_bg_coords = (bbox[0], bbox[1] - text_h - 4, bbox[0] + text_w + 4, bbox[1])
        draw.rectangle(text_bg_coords, fill=color)
        draw.text((bbox[0] + 2, bbox[1] - text_h - 2), label, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB_BGR)

# <-- NEW: Function to draw intermediate results -->
def draw_debug_frame(frame, raw_detections, confirmed_tracks, debug_info):
    # 1. Draw Raw Detections (Gray)
    for box in raw_detections:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)

    # 2. Draw Kalman Predictions (Yellow)
    if 'predicted_boxes' in debug_info:
        for i, box in enumerate(debug_info['predicted_boxes']):
            x1, y1, x2, y2 = map(int, box)
            track_id = debug_info['predicted_ids'][i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"Pred ID:{track_id}", (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 3. Draw Final Confirmed Tracks (Colorful, on top)
    frame_with_tracks = draw_tracked_boxes(frame, confirmed_tracks)
    return frame_with_tracks

def main(args):
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
            return self.postprocessor(outputs, orig_target_sizes)

    model = Model().to(args.device).eval()
    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])

    os.makedirs(args.out_dir, exist_ok=True)
    
    # <-- MODIFIED: Create different output dirs for normal vs debug -->
    vis_folder_name = "visualized_debug" if args.debug else "visualized"
    visualized_output_dir = os.path.join(args.out_dir, vis_folder_name)
    os.makedirs(visualized_output_dir, exist_ok=True)
    
    image_files = sorted([f for f in os.listdir(args.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    results_file_path = os.path.join(args.out_dir, "results.txt")
    if os.path.exists(results_file_path):
        os.remove(results_file_path)
        
    tracker = VAKTTracker(max_age=50, min_hits=3, iou_threshold=0.3, appearance_lambda=0.75)
    
    print(f"Processing {len(image_files)} images from {args.img_dir}...")
    for frame_count, img_name in enumerate(tqdm(image_files), 1):
        img_path = os.path.join(args.img_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
            
        im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)
        im_data = transforms(im_pil)[None].to(args.device)

        with torch.no_grad(), autocast():
            labels, boxes, scores = model(im_data, orig_size)
            
        thrh = 0.5
        scr = scores[0]
        valid_indices = scr > thrh
        
        det_boxes = boxes[0][valid_indices].cpu().numpy()
        det_scores = scr[valid_indices].cpu().numpy()
        det_labels = labels[0][valid_indices].cpu().numpy().astype(int)

        # <-- MODIFIED: Get both return values from the tracker
        tracked_objects, debug_info = tracker.update(det_boxes, det_labels, det_scores, frame)
        
        save_results_to_file(tracked_objects, frame_count, results_file_path)
        
        # <-- MODIFIED: Choose drawing function based on debug flag
        if args.debug:
            result_frame = draw_debug_frame(frame.copy(), det_boxes, tracked_objects, debug_info)
        else:
            result_frame = draw_tracked_boxes(frame.copy(), tracked_objects)
            
        save_path = os.path.join(visualized_output_dir, img_name)
        cv2.imwrite(save_path, result_frame)

    print(f"\n✅ Tracking complete.")
    print(f"  -> Evaluation data saved to: {results_file_path}")
    print(f"  -> Visualized images saved to: {visualized_output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("RT-DETR with VAKT Tracking on Image Sequences")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the model config file.")
    parser.add_argument('-r', '--resume', type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument('--img-dir', type=str, required=True, help="Path to the folder of input images.")
    parser.add_argument('--out-dir', type=str, default="outputs_tracked", help="Directory to save the results.")
    parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use.")
    # <-- NEW: Add the debug flag -->
    parser.add_argument('--debug', action='store_true', help="Enable debug mode to visualize intermediate tracking steps.")
    args = parser.parse_args()
    main(args)

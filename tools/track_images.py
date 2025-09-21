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

# # tools/track_images.py (Updated to save visualized images)
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
# tools/track_images.py

# # Version 3 with heat matrices
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
# import matplotlib.pyplot as plt

# # Make the project modules available (adjust path if needed)
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# from src.core import YAMLConfig
# from vakt_tracker import VAKTTracker

# # --- Constants and Helpers ---
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

# def draw_tracked_boxes(frame, tracks):
#     """Draws final tracked bounding boxes with IDs on the frame."""
#     pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     draw = ImageDraw.Draw(pil_im)
#     try:
#         font = ImageFont.truetype("arial.ttf", 15)
#     except IOError:
#         font = ImageFont.load_default()

#     for track in tracks:
#         bbox, track_id, class_id, score = track['bbox'], track['id'], track['class_id'], track['score']
#         color = tuple(TRACK_COLORS[track_id % len(TRACK_COLORS)].tolist())
#         class_name = CLASS_MAP.get(class_id, f'Class-{class_id}')
#         label = f"ID:{track_id} {class_name} {score:.2f}"

#         draw.rectangle(bbox, outline=color, width=3)
#         text_size = font.getbbox(label)
#         text_w, text_h = text_size[2] - text_size[0], text_size[3] - text_size[1]
#         text_bg_coords = (bbox[0], bbox[1] - text_h - 4, bbox[0] + text_w + 4, bbox[1])
#         draw.rectangle(text_bg_coords, fill=color)
#         draw.text((bbox[0] + 2, bbox[1] - text_h - 2), label, fill=(255, 255, 255), font=font)

#     # Corrected line for the AttributeError
#     return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

# def draw_debug_frame(frame, raw_detections, confirmed_tracks, debug_info):
#     """Draws a detailed debug view of the tracking process."""
#     # 1. Draw Raw Detections (Gray)
#     for box in raw_detections:
#         x1, y1, x2, y2 = map(int, box)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)

#     # 2. Draw Kalman Predictions (Yellow)
#     if 'predicted_boxes' in debug_info:
#         for i, box in enumerate(debug_info['predicted_boxes']):
#             x1, y1, x2, y2 = map(int, box)
#             track_id = debug_info['predicted_ids'][i]
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
#             cv2.putText(frame, f"Pred ID:{track_id}", (x1, y1 - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

#     # 3. Draw Final Confirmed Tracks (Colorful, on top)
#     frame_with_tracks = draw_tracked_boxes(frame, confirmed_tracks)
#     return frame_with_tracks

# def visualize_cost_matrices(debug_info, frame_count, save_dir):
#     """Generates and saves a heatmap of the cost matrices."""
#     motion_cost = debug_info.get('motion_cost')
#     appearance_cost = debug_info.get('appearance_cost')
#     combined_cost = debug_info.get('combined_cost')

#     if combined_cost is None or combined_cost.size == 0:
#         return

#     num_detections, num_tracks = combined_cost.shape
#     track_ids = debug_info.get('predicted_ids', list(range(num_tracks)))

#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#     fig.suptitle(f'Frame {frame_count}: Cost Matrix Analysis (Darker = Better Match)', fontsize=16)

#     im1 = axes[0].imshow(motion_cost, cmap='viridis_r', aspect='auto')
#     axes[0].set_title('1. Motion Cost (1 - IoU)')
#     fig.colorbar(im1, ax=axes[0])

#     im2 = axes[1].imshow(appearance_cost, cmap='viridis_r', aspect='auto')
#     axes[1].set_title('2. Appearance Cost (Color Similarity)')
#     fig.colorbar(im2, ax=axes[1])

#     im3 = axes[2].imshow(combined_cost, cmap='viridis_r', aspect='auto')
#     axes[2].set_title('3. Final Combined Cost')
#     fig.colorbar(im3, ax=axes[2])

#     for ax in axes:
#         ax.set_ylabel('New Detections (Index)')
#         ax.set_xlabel('Existing Tracks (ID)')
#         ax.set_xticks(np.arange(num_tracks))
#         ax.set_xticklabels(track_ids)
#         if num_detections > 0:
#             ax.set_yticks(np.arange(num_detections))
#             ax.set_yticklabels(np.arange(num_detections))

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     save_path = os.path.join(save_dir, f'frame_{frame_count:04d}_matrices.png')
#     plt.savefig(save_path)
#     plt.close(fig)

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
#     transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])

#     os.makedirs(args.out_dir, exist_ok=True)

#     vis_folder_name = "visualized_debug" if args.debug else "visualized"
#     visualized_output_dir = os.path.join(args.out_dir, vis_folder_name)
#     os.makedirs(visualized_output_dir, exist_ok=True)

#     matrices_output_dir = None
#     if args.save_matrices:
#         matrices_output_dir = os.path.join(args.out_dir, "cost_matrices")
#         os.makedirs(matrices_output_dir, exist_ok=True)

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

#         tracked_objects, debug_info = tracker.update(det_boxes, det_labels, det_scores, frame)

#         save_results_to_file(tracked_objects, frame_count, results_file_path)

#         if args.save_matrices:
#             visualize_cost_matrices(debug_info, frame_count, matrices_output_dir)

#         if args.debug:
#             result_frame = draw_debug_frame(frame.copy(), det_boxes, tracked_objects, debug_info)
#         else:
#             result_frame = draw_tracked_boxes(frame.copy(), tracked_objects)
#         save_path = os.path.join(visualized_output_dir, img_name)
#         cv2.imwrite(save_path, result_frame)

#     print(f"\n✅ Tracking complete.")
#     print(f"  -> Evaluation data saved to: {results_file_path}")
#     print(f"  -> Visualized images saved to: {visualized_output_dir}")
#     if args.save_matrices:
#         print(f"  -> Cost matrix plots saved to: {matrices_output_dir}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser("RT-DETR with VAKT Tracking on Image Sequences")
#     parser.add_argument('-c', '--config', type=str, required=True, help="Path to the model config file.")
#     parser.add_argument('-r', '--resume', type=str, required=True, help="Path to the trained model checkpoint.")
#     parser.add_argument('--img-dir', type=str, required=True, help="Path to the folder of input images.")
#     parser.add_argument('--out-dir', type=str, default="outputs_tracked", help="Directory to save the results.")
#     parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use.")
#     parser.add_argument('--debug', action='store_true', help="Enable debug mode to visualize intermediate tracking steps.")
#     parser.add_argument('--save-matrices', action='store_true', help="Save heatmaps of the cost matrices for analysis.")
#     args = parser.parse_args()
#     main(args)

# version 4 with heatmaps of actual images
# tools/track_images.py (Updated with Image Heatmaps)
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
# import matplotlib.pyplot as plt

# # Make the project modules available (adjust path if needed)
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# from src.core import YAMLConfig
# from vakt_tracker import VAKTTracker # Ensure this is the latest version that returns debug_info

# # --- Constants and Helpers ---
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

# def draw_tracked_boxes(frame, tracks):
#     """Draws final tracked bounding boxes with IDs on the frame."""
#     pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     draw = ImageDraw.Draw(pil_im)
#     try:
#         font = ImageFont.truetype("arial.ttf", 15)
#     except IOError:
#         font = ImageFont.load_default()

#     for track in tracks:
#         bbox, track_id, class_id, score = track['bbox'], track['id'], track['class_id'], track['score']
#         color = tuple(TRACK_COLORS[track_id % len(TRACK_COLORS)].tolist())
#         class_name = CLASS_MAP.get(class_id, f'Class-{class_id}')
#         label = f"ID:{track_id} {class_name} {score:.2f}"

#         draw.rectangle(bbox, outline=color, width=3)
#         text_size = font.getbbox(label)
#         text_w, text_h = text_size[2] - text_size[0], text_size[3] - text_size[1]
#         text_bg_coords = (bbox[0], bbox[1] - text_h - 4, bbox[0] + text_w + 4, bbox[1])
#         draw.rectangle(text_bg_coords, fill=color)
#         draw.text((bbox[0] + 2, bbox[1] - text_h - 2), label, fill=(255, 255, 255), font=font)

#     return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

# def draw_debug_frame(frame, raw_detections, confirmed_tracks, debug_info):
#     """Draws a detailed debug view of the tracking process (detections, predictions, final tracks)."""
#     # 1. Draw Raw Detections (Gray)
#     for box in raw_detections:
#         x1, y1, x2, y2 = map(int, box)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)

#     # 2. Draw Kalman Predictions (Yellow)
#     if 'predicted_boxes' in debug_info:
#         for i, box in enumerate(debug_info['predicted_boxes']):
#             x1, y1, x2, y2 = map(int, box)
#             track_id = debug_info['predicted_ids'][i]
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
#             cv2.putText(frame, f"Pred ID:{track_id}", (x1, y1 - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

#     # 3. Draw Final Confirmed Tracks (Colorful, on top)
#     frame_with_tracks = draw_tracked_boxes(frame, confirmed_tracks)
#     return frame_with_tracks

# def visualize_cost_matrices(debug_info, frame_count, save_dir):
#     """Generates and saves a heatmap of the numerical cost matrices."""
#     motion_cost = debug_info.get('motion_cost')
#     appearance_cost = debug_info.get('appearance_cost')
#     combined_cost = debug_info.get('combined_cost')

#     if combined_cost is None or combined_cost.size == 0:
#         return

#     num_detections, num_tracks = combined_cost.shape
#     track_ids = debug_info.get('predicted_ids', list(range(num_tracks)))

#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#     fig.suptitle(f'Frame {frame_count}: Cost Matrix Analysis (Darker = Better Match)', fontsize=16)

#     im1 = axes[0].imshow(motion_cost, cmap='viridis_r', aspect='auto', vmin=0, vmax=1) # Ensure consistent colormap range
#     axes[0].set_title('1. Motion Cost (1 - IoU)')
#     fig.colorbar(im1, ax=axes[0])

#     im2 = axes[1].imshow(appearance_cost, cmap='viridis_r', aspect='auto', vmin=0, vmax=1) # Ensure consistent colormap range
#     axes[1].set_title('2. Appearance Cost (Color Similarity)')
#     fig.colorbar(im2, ax=axes[1])

#     im3 = axes[2].imshow(combined_cost, cmap='viridis_r', aspect='auto', vmin=0, vmax=1) # Ensure consistent colormap range
#     axes[2].set_title('3. Final Combined Cost')
#     fig.colorbar(im3, ax=axes[2])

#     for ax in axes:
#         ax.set_ylabel('New Detections (Index)')
#         ax.set_xlabel('Existing Tracks (ID)')
#         ax.set_xticks(np.arange(num_tracks))
#         ax.set_xticklabels(track_ids)
#         if num_detections > 0:
#             ax.set_yticks(np.arange(num_detections))
#             ax.set_yticklabels(np.arange(num_detections))

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     save_path = os.path.join(save_dir, f'frame_{frame_count:04d}_matrices.png')
#     plt.savefig(save_path)
#     plt.close(fig)


# # --- NEW FUNCTION: To generate image-based heatmaps ---
# def generate_image_heatmaps(original_frame, det_boxes, debug_info, frame_count, save_dir):
#     """
#     Generates and saves image-based heatmaps visualizing motion predictions and appearance.
#     """
#     h, w, _ = original_frame.shape
#     base_alpha = 0.4 # Transparency of the heatmap overlay

#     # --- 1. Motion Prediction Heatmap (based on Kalman Filter's state) ---
#     motion_heatmap = np.zeros((h, w), dtype=np.float32)
#     if 'predicted_boxes' in debug_info and debug_info['predicted_boxes'].size > 0:
#         for box in debug_info['predicted_boxes']:
#             # Convert Kalman state [cx, cy, w, h] to [x1, y1, x2, y2]
#             cx, cy, bw, bh = box[0]+(box[2]-box[0])/2, box[1]+(box[3]-box[1])/2, box[2]-box[0], box[3]-box[1]
#             x1, y1, x2, y2 = int(cx - bw/2), int(cy - bh/2), int(cx + bw/2), int(cy + bh/2)

#             # Create a localized Gaussian blur / "hotspot" for each prediction
#             # This is a simplified representation of where KF "expects" object
#             heatmap_area = motion_heatmap[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
#             if heatmap_area.size > 0:
#                 cv2.rectangle(motion_heatmap, (x1, y1), (x2, y2), 1.0, -1) # Fill rectangle for simplicity
        
#         # Apply some blur to make it look more like a heatmap
#         motion_heatmap = cv2.GaussianBlur(motion_heatmap, (55, 55), 0)
#         motion_heatmap = np.clip(motion_heatmap, 0, 1) # Normalize to 0-1

#     # --- 2. Combined Detections (Motion + Appearance combined influence) Heatmap ---
#     combined_det_heatmap = np.zeros((h, w), dtype=np.float32)
#     if 'combined_cost' in debug_info and debug_info['combined_cost'].size > 0:
#         cost_matrix = debug_info['combined_cost']
#         # For each detection, find the minimum cost to any track
#         min_costs_per_detection = np.min(cost_matrix, axis=1) # lower is better
#         # Normalize costs so that 0 (best match) is high intensity, 1 (worst) is low
#         # Or, just show the detections themselves
        
#         for i, det_box in enumerate(det_boxes):
#             x1, y1, x2, y2 = map(int, det_box)
#             intensity = 1.0 - min_costs_per_detection[i] if len(min_costs_per_detection) > i else 1.0 # Invert cost for intensity
            
#             # Draw a filled rectangle with intensity based on how well it matched
#             heatmap_area = combined_det_heatmap[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
#             if heatmap_area.size > 0:
#                 cv2.rectangle(combined_det_heatmap, (x1, y1), (x2, y2), intensity, -1)
        
#         combined_det_heatmap = cv2.GaussianBlur(combined_det_heatmap, (45, 45), 0)
#         combined_det_heatmap = np.clip(combined_det_heatmap, 0, 1) # Normalize to 0-1


#     # --- Overlay heatmaps on the original image ---
#     output_frame_motion = original_frame.copy()
#     output_frame_combined_det = original_frame.copy()

#     # Apply colormap to heatmaps
#     motion_heatmap_colored = cv2.applyColorMap(np.uint8(255 * motion_heatmap), cv2.COLORMAP_JET)
#     combined_det_heatmap_colored = cv2.applyColorMap(np.uint8(255 * combined_det_heatmap), cv2.COLORMAP_HOT)

#     # Blend with original image
#     cv2.addWeighted(motion_heatmap_colored, base_alpha, output_frame_motion, 1 - base_alpha, 0, output_frame_motion)
#     cv2.addWeighted(combined_det_heatmap_colored, base_alpha, output_frame_combined_det, 1 - base_alpha, 0, output_frame_combined_det)

#     # --- Combine into a single visualization for saving ---
#     # Put original frame, motion heatmap frame, and combined detection heatmap frame side-by-side
#     # Pad if necessary to ensure consistent height for concatenation
#     pad_h = max(output_frame_motion.shape[0], output_frame_combined_det.shape[0])
    
#     # Text labels
#     cv2.putText(original_frame, "Original Frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#     cv2.putText(output_frame_motion, "Motion Prediction Heatmap", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
#     cv2.putText(output_frame_combined_det, "Combined Detections Match Heatmap", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

#     combined_visualization = np.concatenate([original_frame, output_frame_motion, output_frame_combined_det], axis=1)

#     # Save the combined visualization
#     save_path = os.path.join(save_dir, f'frame_{frame_count:04d}_image_heatmaps.png')
#     cv2.imwrite(save_path, combined_visualization)


# # --- Main execution loop (same as before, but with heatmap call) ---
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
#     transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])

#     os.makedirs(args.out_dir, exist_ok=True)

#     vis_folder_name = "visualized_debug" if args.debug else "visualized"
#     visualized_output_dir = os.path.join(args.out_dir, vis_folder_name)
#     os.makedirs(visualized_output_dir, exist_ok=True)

#     matrices_output_dir = None
#     if args.save_matrices:
#         matrices_output_dir = os.path.join(args.out_dir, "cost_matrices")
#         os.makedirs(matrices_output_dir, exist_ok=True)
    
#     # New: Image heatmaps directory
#     image_heatmaps_output_dir = None
#     if args.visualize_heatmaps:
#         image_heatmaps_output_dir = os.path.join(args.out_dir, "image_heatmaps")
#         os.makedirs(image_heatmaps_output_dir, exist_ok=True)


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
#         original_frame_for_heatmap = frame.copy() # Keep a clean copy for heatmap generation

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

#         tracked_objects, debug_info = tracker.update(det_boxes, det_labels, det_scores, frame)

#         save_results_to_file(tracked_objects, frame_count, results_file_path)

#         if args.save_matrices:
#             visualize_cost_matrices(debug_info, frame_count, matrices_output_dir)
        
#         # New: Generate and save image heatmaps
#         if args.visualize_heatmaps:
#             generate_image_heatmaps(original_frame_for_heatmap, det_boxes, debug_info, frame_count, image_heatmaps_output_dir)


#         if args.debug:
#             result_frame = draw_debug_frame(frame.copy(), det_boxes, tracked_objects, debug_info)
#         else:
#             result_frame = draw_tracked_boxes(frame.copy(), tracked_objects)
#         save_path = os.path.join(visualized_output_dir, img_name)
#         cv2.imwrite(save_path, result_frame)

#     print(f"\n✅ Tracking complete.")
#     print(f"  -> Evaluation data saved to: {results_file_path}")
#     print(f"  -> Visualized images saved to: {visualized_output_dir}")
#     if args.save_matrices:
#         print(f"  -> Cost matrix plots saved to: {matrices_output_dir}")
#     if args.visualize_heatmaps:
#         print(f"  -> Image heatmaps saved to: {image_heatmaps_output_dir}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser("RT-DETR with VAKT Tracking on Image Sequences")
#     parser.add_argument('-c', '--config', type=str, required=True, help="Path to the model config file.")
#     parser.add_argument('-r', '--resume', type=str, required=True, help="Path to the trained model checkpoint.")
#     parser.add_argument('--img-dir', type=str, required=True, help="Path to the folder of input images.")
#     parser.add_argument('--out-dir', type=str, default="outputs_tracked", help="Directory to save the results.")
#     parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use.")
#     parser.add_argument('--debug', action='store_true', help="Enable debug mode to visualize intermediate tracking steps.")
#     parser.add_argument('--save-matrices', action='store_true', help="Save heatmaps of the cost matrices for analysis.")
#     parser.add_argument('--visualize-heatmaps', action='store_true', help="Generate and save image-based heatmaps for motion and appearance.") # NEW FLAG
#     args = parser.parse_args()
#     main(args)

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
import matplotlib.pyplot as plt

# Make the project modules available (adjust path if needed)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig
from vakt_tracker import VAKTTracker # Ensure this is the latest version that returns debug_info

# --- Constants and Helpers ---
CLASS_MAP = {0: 'swimmer', 1: 'swimmer with life jacket', 2: 'boat'}
np.random.seed(42)
TRACK_COLORS = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)

def save_results_to_file(tracked_objects, frame_count, file_path):
    """Saves tracking results to a text file in MOT Challenge format."""
    with open(file_path, 'a') as f:
        for track in tracked_objects:
            bbox = track['bbox']
            track_id = track['id']
            x, y = bbox[0], bbox[1]
            w, h = bbox[2] - x, bbox[3] - y
            f.write(f"{frame_count},{track_id},{x},{y},{w},{h},{track['score']:.4f},-1,-1,-1\n")

def draw_tracked_boxes(frame, tracks):
    """Draws final tracked bounding boxes with IDs on the frame."""
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

    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

def draw_debug_frame(frame, raw_detections, confirmed_tracks, debug_info):
    """Draws a detailed debug view of the tracking process (detections, predictions, final tracks)."""
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

def visualize_cost_matrices(debug_info, frame_count, save_dir):
    """Generates and saves a heatmap of the numerical cost matrices."""
    motion_cost = debug_info.get('motion_cost')
    appearance_cost = debug_info.get('appearance_cost')
    combined_cost = debug_info.get('combined_cost')

    if combined_cost is None or combined_cost.size == 0:
        return

    num_detections, num_tracks = combined_cost.shape
    track_ids = debug_info.get('predicted_ids', list(range(num_tracks)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Frame {frame_count}: Cost Matrix Analysis (Darker = Better Match)', fontsize=16)

    im1 = axes[0].imshow(motion_cost, cmap='viridis_r', aspect='auto', vmin=0, vmax=1)
    axes[0].set_title('1. Motion Cost (1 - IoU)')
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(appearance_cost, cmap='viridis_r', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title('2. Appearance Cost (Color Similarity)')
    fig.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(combined_cost, cmap='viridis_r', aspect='auto', vmin=0, vmax=1)
    axes[2].set_title('3. Final Combined Cost')
    fig.colorbar(im3, ax=axes[2])

    for ax in axes:
        ax.set_ylabel('New Detections (Index)')
        ax.set_xlabel('Existing Tracks (ID)')
        ax.set_xticks(np.arange(num_tracks))
        ax.set_xticklabels(track_ids)
        if num_detections > 0:
            ax.set_yticks(np.arange(num_detections))
            ax.set_yticklabels(np.arange(num_detections))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, f'frame_{frame_count:04d}_matrices.png')
    plt.savefig(save_path)
    plt.close(fig)


def generate_image_heatmaps(original_frame, det_boxes, debug_info, frame_count, save_dir):
    """
    Generates and saves image-based heatmaps visualizing motion predictions and appearance.
    """
    h, w, _ = original_frame.shape
    base_alpha = 0.4 # Transparency of the heatmap overlay

    # --- 1. Motion Prediction Heatmap (based on Kalman Filter's state) ---
    motion_heatmap = np.zeros((h, w), dtype=np.float32)
    if 'predicted_boxes' in debug_info and debug_info['predicted_boxes'].size > 0:
        for box in debug_info['predicted_boxes']:
            cx, cy, bw, bh = box[0]+(box[2]-box[0])/2, box[1]+(box[3]-box[1])/2, box[2]-box[0], box[3]-box[1]
            x1, y1, x2, y2 = int(cx - bw/2), int(cy - bh/2), int(cx + bw/2), int(cy + bh/2)

            heatmap_area = motion_heatmap[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if heatmap_area.size > 0:
                cv2.rectangle(motion_heatmap, (x1, y1), (x2, y2), 1.0, -1)
        
        motion_heatmap = cv2.GaussianBlur(motion_heatmap, (55, 55), 0)
        motion_heatmap = np.clip(motion_heatmap, 0, 1)

    # --- 2. Combined Detections (Motion + Appearance combined influence) Heatmap ---
    combined_det_heatmap = np.zeros((h, w), dtype=np.float32)
    if 'combined_cost' in debug_info and debug_info['combined_cost'].size > 0:
        cost_matrix = debug_info['combined_cost']
        min_costs_per_detection = np.min(cost_matrix, axis=1)
        
        for i, det_box in enumerate(det_boxes):
            x1, y1, x2, y2 = map(int, det_box)
            intensity = 1.0 - min_costs_per_detection[i] if len(min_costs_per_detection) > i else 1.0
            
            heatmap_area = combined_det_heatmap[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if heatmap_area.size > 0:
                cv2.rectangle(combined_det_heatmap, (x1, y1), (x2, y2), intensity, -1)
        
        combined_det_heatmap = cv2.GaussianBlur(combined_det_heatmap, (45, 45), 0)
        combined_det_heatmap = np.clip(combined_det_heatmap, 0, 1)

    # --- Overlay heatmaps on the original image ---
    output_frame_motion = original_frame.copy()
    output_frame_combined_det = original_frame.copy()

    motion_heatmap_colored = cv2.applyColorMap(np.uint8(255 * motion_heatmap), cv2.COLORMAP_JET)
    combined_det_heatmap_colored = cv2.applyColorMap(np.uint8(255 * combined_det_heatmap), cv2.COLORMAP_HOT)

    cv2.addWeighted(motion_heatmap_colored, base_alpha, output_frame_motion, 1 - base_alpha, 0, output_frame_motion)
    cv2.addWeighted(combined_det_heatmap_colored, base_alpha, output_frame_combined_det, 1 - base_alpha, 0, output_frame_combined_det)

    # --- Combine into a single visualization for saving ---
    cv2.putText(original_frame, "Original Frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(output_frame_motion, "Motion Prediction Heatmap", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output_frame_combined_det, "Combined Detections Match Heatmap", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

    combined_visualization = np.concatenate([original_frame, output_frame_motion, output_frame_combined_det], axis=1)

    save_path = os.path.join(save_dir, f'frame_{frame_count:04d}_image_heatmaps.png')
    cv2.imwrite(save_path, combined_visualization)


# --- Main execution loop ---
def main(args):
    """Main function to set up model and process an image sequence."""
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

    # --- REMOVED: Directory creation for visualized output images ---
    # vis_folder_name = "visualized_debug" if args.debug else "visualized"
    # visualized_output_dir = os.path.join(args.out_dir, vis_folder_name)
    # os.makedirs(visualized_output_dir, exist_ok=True)

    matrices_output_dir = None
    if args.save_matrices:
        matrices_output_dir = os.path.join(args.out_dir, "cost_matrices")
        os.makedirs(matrices_output_dir, exist_ok=True)
    
    image_heatmaps_output_dir = None
    if args.visualize_heatmaps:
        image_heatmaps_output_dir = os.path.join(args.out_dir, "image_heatmaps")
        os.makedirs(image_heatmaps_output_dir, exist_ok=True)

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
        original_frame_for_heatmap = frame.copy()

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

        tracked_objects, debug_info = tracker.update(det_boxes, det_labels, det_scores, frame)

        save_results_to_file(tracked_objects, frame_count, results_file_path)

        if args.save_matrices:
            visualize_cost_matrices(debug_info, frame_count, matrices_output_dir)
        
        if args.visualize_heatmaps:
            generate_image_heatmaps(original_frame_for_heatmap, det_boxes, debug_info, frame_count, image_heatmaps_output_dir)

        # --- REMOVED: Drawing and saving of the final visualized frame ---
        # if args.debug:
        #     result_frame = draw_debug_frame(frame.copy(), det_boxes, tracked_objects, debug_info)
        # else:
        #     result_frame = draw_tracked_boxes(frame.copy(), tracked_objects)
        # save_path = os.path.join(visualized_output_dir, img_name)
        # cv2.imwrite(save_path, result_frame)

    print(f"\n✅ Tracking complete.")
    print(f"  -> Evaluation data saved to: {results_file_path}")
    # --- REMOVED: Print statement for the visualized images directory ---
    # print(f"  -> Visualized images saved to: {visualized_output_dir}")
    if args.save_matrices:
        print(f"  -> Cost matrix plots saved to: {matrices_output_dir}")
    if args.visualize_heatmaps:
        print(f"  -> Image heatmaps saved to: {image_heatmaps_output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("RT-DETR with VAKT Tracking on Image Sequences")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the model config file.")
    parser.add_argument('-r', '--resume', type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument('--img-dir', type=str, required=True, help="Path to the folder of input images.")
    parser.add_argument('--out-dir', type=str, default="outputs_tracked", help="Directory to save the results.")
    parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode to visualize intermediate tracking steps.")
    parser.add_argument('--save-matrices', action='store_true', help="Save heatmaps of the cost matrices for analysis.")
    parser.add_argument('--visualize-heatmaps', action='store_true', help="Generate and save image-based heatmaps for motion and appearance.")
    args = parser.parse_args()
    main(args)

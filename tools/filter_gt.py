# # tools/filter_gt.py
# import argparse
# import os

# def main():
#     parser = argparse.ArgumentParser(description='Filter a MOT ground truth file by frame range.')
#     parser.add_argument('--input_gt', type=str, required=True, help='Path to the full gt.txt file.')
#     parser.add_argument('--output_gt', type=str, required=True, help='Path to save the filtered subset gt.txt file.')
#     parser.add_argument('--start_frame', type=int, required=True, help='The first frame number to include.')
#     parser.add_argument('--end_frame', type=int, required=True, help='The last frame number to include.')
#     args = parser.parse_args()

#     # Ensure output directory exists
#     os.makedirs(os.path.dirname(args.output_gt), exist_ok=True)
    
#     with open(args.input_gt, 'r') as f_in, open(args.output_gt, 'w') as f_out:
#         for line in f_in:
#             parts = line.strip().split(',')
#             try:
#                 frame_id = int(parts[0])
#             except (ValueError, IndexError):
#                 continue # Skip malformed lines
            
#             if args.start_frame <= frame_id <= args.end_frame:
#                 # Re-number the frame ID to start from 1 for the subset
#                 new_frame_id = frame_id - args.start_frame + 1
#                 parts[0] = str(new_frame_id)
#                 f_out.write(','.join(parts) + '\n')
                
#     print(f"✅ Filtered frames {args.start_frame}-{args.end_frame}. New GT saved to {args.output_gt}")

# if __name__ == '__main__':
#     main()

# tools/create_subset_gt.py
import json
import os
import argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='Create a correct GT file for a subset of images from a COCO JSON.')
    parser.add_argument('--coco_json', type=str, required=True, help='Path to the main COCO annotations file with track IDs.')
    parser.add_argument('--img_dir', type=str, required=True, help='Path to the subfolder with a continuous sequence of images.')
    parser.add_argument('--output_gt', type=str, required=True, help='Path to save the new, correct gt.txt file.')
    args = parser.parse_args()

    print(f"Loading main annotations from {args.coco_json}...")
    with open(args.coco_json, 'r') as f:
        data = json.load(f)

    # Create a mapping from filename to image_id
    filename_to_id = {img['file_name']: img['id'] for img in data['images']}
    
    # Create a mapping from image_id to its annotations
    annotations_by_id = defaultdict(list)
    for ann in data['annotations']:
        annotations_by_id[ann['image_id']].append(ann)

    # Get the sorted list of image files from the subset directory
    subset_files = sorted([f for f in os.listdir(args.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(f"Found {len(subset_files)} images in {args.img_dir}. Creating matched GT file...")
    
    os.makedirs(os.path.dirname(args.output_gt), exist_ok=True)
    
    with open(args.output_gt, 'w') as f_out:
        # Loop through the image files in the subset folder, assigning new frame numbers
        for new_frame_id, filename in enumerate(subset_files, 1):
            if filename not in filename_to_id:
                print(f"Warning: {filename} not found in main JSON file. Skipping.")
                continue
            
            original_image_id = filename_to_id[filename]
            
            # Find all annotations for this image and write them with the new frame ID
            for ann in annotations_by_id[original_image_id]:
                track_id = ann['track_id']
                x, y, w, h = ann['bbox']
                line = f"{new_frame_id},{track_id},{x},{y},{w},{h},1,-1,-1,-1\n"
                f_out.write(line)

    print(f"✅ Successfully created a matched ground truth file at {args.output_gt}")

if __name__ == '__main__':
    main()

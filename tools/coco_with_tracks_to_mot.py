# coco_with_tracks_to_mot.py
import json
import argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(
        description='Convert COCO-style JSON with track IDs to MOT format.'
    )
    parser.add_argument('--coco_json', type=str, required=True, 
                        help='Path to the COCO annotations file with track IDs.')
    parser.add_argument('--output_gt', type=str, required=True, 
                        help='Path to save the output gt.txt file.')
    args = parser.parse_args()

    print(f"Loading annotations from {args.coco_json}...")
    with open(args.coco_json, 'r') as f:
        data = json.load(f)

    # --- Important Note ---
    # Check your JSON file for the exact key used for the track ID.
    # Common keys are 'track_id', 'instance_id', or sometimes just 'id'.
    # I am using 'track_id' here. If yours is different, change the key below.
    TRACK_ID_KEY = 'track_id' 

    if 'annotations' not in data or not data['annotations']:
        print("Error: No 'annotations' found in the JSON file.")
        return
        
    # Verify that the track ID key exists in the first annotation
    if TRACK_ID_KEY not in data['annotations'][0]:
        print(f"Error: The key '{TRACK_ID_KEY}' was not found in the annotations.")
        print("Please check your JSON file and update the TRACK_ID_KEY variable in this script.")
        return

    # Group annotations by frame (image_id) to sort them later
    grouped_annotations = defaultdict(list)
    for ann in data['annotations']:
        frame_id = ann['image_id']
        track_id = ann[TRACK_ID_KEY]
        x, y, w, h = ann['bbox']
        
        # Format: <frame_id>,<track_id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,1,-1,-1,-1
        line = f"{frame_id},{track_id},{x},{y},{w},{h},1,-1,-1,-1\n"
        grouped_annotations[frame_id].append(line)

    print(f"Writing to {args.output_gt}...")
    with open(args.output_gt, 'w') as f_out:
        # Sort by frame number and write to file
        for frame_id in sorted(grouped_annotations.keys()):
            for line in grouped_annotations[frame_id]:
                f_out.write(line)
                
    print(f"✅ Successfully converted annotations and saved to {args.output_gt}")

if __name__ == '__main__':
    main()
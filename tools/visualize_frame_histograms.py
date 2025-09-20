# tools/visualize_frame_histograms.py
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
from tqdm import tqdm

# The helper function 'generate_frame_visualization' remains exactly the same.
def generate_frame_visualization(frame, frame_data, save_path):
    num_tracks = len(frame_data)
    if num_tracks == 0: return
    fig = plt.figure(figsize=(20, 5 * num_tracks))
    gs = GridSpec(num_tracks, 2, width_ratios=[3, 2])
    ax_main = fig.add_subplot(gs[:, 0])
    frame_with_boxes = frame.copy()
    for _, track in frame_data.iterrows():
        track_id = int(track['id'])
        x, y, w, h = map(int, [track['x'], track['y'], track['w'], track['h']])
        # Use a consistent color map for track IDs
        color = plt.cm.get_cmap('hsv', 256)(track_id % 256)[:3]
        color = tuple([int(c*255) for c in color]) # Convert to BGR for OpenCV
        cv2.rectangle(frame_with_boxes, (x, y), (x+w, y+h), color, 3)
        cv2.putText(frame_with_boxes, f"ID:{track_id}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    ax_main.imshow(cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB))
    ax_main.set_title(f"Frame {int(frame_data.iloc[0]['frame'])}")
    ax_main.axis('off')

    for i, (_, track) in enumerate(frame_data.iterrows()):
        track_id = int(track['id'])
        x, y, w, h = map(int, [track['x'], track['y'], track['w'], track['h']])
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0: continue
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv_roi], [1], None, [256], [0, 256])
        ax_hist = fig.add_subplot(gs[i, 1])
        ax_hist.plot(hist_h, color='r', label='Hue')
        ax_hist.plot(hist_s, color='g', label='Saturation')
        ax_hist.set_title(f'Histograms for Track ID: {track_id}')
        ax_hist.legend(); ax_hist.set_xlim([0, 256])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def main(args):
    """
    Main function to read tracking results and generate frame-by-frame histogram visualizations.
    """
    try:
        results_df = pd.read_csv(args.results_file, header=None,
                                 names=['frame', 'id', 'x', 'y', 'w', 'h', 'score', 'c1', 'c2', 'c3'])
    except FileNotFoundError:
        print(f"Error: Results file not found at '{args.results_file}'")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    
    # --- MODIFIED "ONE-BY-ONE" LOGIC ---

    # 1. Get a sorted list of unique frame numbers that have tracking data.
    unique_frames = sorted(results_df['frame'].unique())

    # 2. Get a sorted list of all image files in the directory.
    all_files = sorted(os.listdir(args.img_dir))
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 3. IMPORTANT: Warn the user if the counts do not match.
    if len(image_files) != len(unique_frames):
        print("="*60)
        print("!! WARNING: POTENTIAL DATA MISMATCH !!")
        print(f"Found {len(image_files)} image files but tracking data for {len(unique_frames)} unique frames.")
        print("The visualization may be incorrect as the data could be misaligned.")
        print("Processing the minimum number of available pairs.")
        print("="*60)

    # 4. Process the two lists together, one by one.
    num_pairs_to_process = min(len(image_files), len(unique_frames))
    print(f"Processing {num_pairs_to_process} pairs of images and frame data sequentially...")

    for image_name, frame_number in tqdm(zip(image_files, unique_frames), total=num_pairs_to_process):
        # Filter the dataframe for the current frame number from the list
        frame_data = results_df[results_df['frame'] == frame_number]
        
        # Load the corresponding image from the list
        image_path = os.path.join(args.img_dir, image_name)
        frame_image = cv2.imread(image_path)
        
        if frame_image is None:
            print(f"Warning: Failed to load image '{image_path}'. Skipping.")
            continue

        # Use the frame number for the output filename for clarity
        save_path = os.path.join(args.out_dir, f'frame_{frame_number:04d}_analysis.png')
        generate_frame_visualization(frame_image, frame_data, save_path)
        
    print(f"\n✅ Analysis complete. Visualizations saved in: '{args.out_dir}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze and visualize color histograms for all tracks on a per-frame basis.")
    parser.add_argument('--results-file', type=str, required=True, help="Path to the results.txt file from the tracker.")
    parser.add_argument('--img-dir', type=str, required=True, help="Path to the directory of original image frames.")
    parser.add_argument('--out-dir', type=str, default="outputs_frame_analysis", help="Directory to save the analysis images.")
    args = parser.parse_args()
    main(args)

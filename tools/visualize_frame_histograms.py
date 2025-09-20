# tools/visualize_frame_histograms.py
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
from tqdm import tqdm

# --- Drawing and Plotting Helper ---

def generate_frame_visualization(frame, frame_data, save_path):
    """
    Creates a single composite image showing the main frame with bounding boxes
    and the color histograms for every object tracked in that frame.
    """
    num_tracks = len(frame_data)
    if num_tracks == 0:
        return

    # Create a figure with a flexible grid
    fig = plt.figure(figsize=(20, 5 * num_tracks))
    gs = GridSpec(num_tracks, 2, width_ratios=[3, 2]) # Main image is wider

    # --- Plot the main image with all bounding boxes ---
    ax_main = fig.add_subplot(gs[:, 0])
    
    # Draw boxes on a copy of the frame
    frame_with_boxes = frame.copy()
    for _, track in frame_data.iterrows():
        track_id = int(track['id'])
        x, y, w, h = map(int, [track['x'], track['y'], track['w'], track['h']])
        color = plt.cm.get_cmap('hsv', 256)(track_id % 256)[:3] # Use matplotlib colormap for consistency
        color = tuple([int(c*255) for c in color]) # Convert to BGR for OpenCV
        cv2.rectangle(frame_with_boxes, (x, y), (x+w, y+h), color, 3)
        cv2.putText(frame_with_boxes, f"ID:{track_id}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    ax_main.imshow(cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB))
    ax_main.set_title(f"Frame {int(frame_data.iloc[0]['frame'])}")
    ax_main.axis('off')

    # --- Plot histograms for each track in the frame ---
    for i, (_, track) in enumerate(frame_data.iterrows()):
        track_id = int(track['id'])
        x, y, w, h = map(int, [track['x'], track['y'], track['w'], track['h']])

        # Crop ROI
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0: continue
        
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms
        hist_h = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv_roi], [1], None, [256], [0, 256])
        
        # Create subplot for this track's histograms
        ax_hist = fig.add_subplot(gs[i, 1])
        ax_hist.plot(hist_h, color='r', label='Hue')
        ax_hist.plot(hist_s, color='g', label='Saturation')
        ax_hist.set_title(f'Histograms for Track ID: {track_id}')
        ax_hist.legend()
        ax_hist.set_xlim([0, 256])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def main(args):
    """
    Main function to read tracking results and generate frame-by-frame histogram visualizations.
    """
    # Load the tracking results file
    try:
        results_df = pd.read_csv(args.results_file, header=None,
                                 names=['frame', 'id', 'x', 'y', 'w', 'h', 'score', 'c1', 'c2', 'c3'])
    except FileNotFoundError:
        print(f"Error: Results file not found at '{args.results_file}'")
        return
        
    # Create the main output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Group results by frame
    grouped = results_df.groupby('frame')
    
    print(f"Found {len(grouped)} frames with tracking data to analyze...")

    for frame_number, frame_data in tqdm(grouped):
        # Find the corresponding image file
        image_name_candidates = [f"{frame_number:04d}.jpg", f"{frame_number:04d}.png"]
        
        image_path = None
        for name in image_name_candidates:
            path = os.path.join(args.img_dir, name)
            if os.path.exists(path):
                image_path = path
                break
        
        if not image_path:
            print(f"Warning: Could not find image for frame {frame_number}. Skipping.")
            continue
            
        frame_image = cv2.imread(image_path)
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

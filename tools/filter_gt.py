# tools/filter_gt.py
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Filter a MOT ground truth file by frame range.')
    parser.add_argument('--input_gt', type=str, required=True, help='Path to the full gt.txt file.')
    parser.add_argument('--output_gt', type=str, required=True, help='Path to save the filtered subset gt.txt file.')
    parser.add_argument('--start_frame', type=int, required=True, help='The first frame number to include.')
    parser.add_argument('--end_frame', type=int, required=True, help='The last frame number to include.')
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_gt), exist_ok=True)
    
    with open(args.input_gt, 'r') as f_in, open(args.output_gt, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split(',')
            try:
                frame_id = int(parts[0])
            except (ValueError, IndexError):
                continue # Skip malformed lines
            
            if args.start_frame <= frame_id <= args.end_frame:
                # Re-number the frame ID to start from 1 for the subset
                new_frame_id = frame_id - args.start_frame + 1
                parts[0] = str(new_frame_id)
                f_out.write(','.join(parts) + '\n')
                
    print(f"✅ Filtered frames {args.start_frame}-{args.end_frame}. New GT saved to {args.output_gt}")

if __name__ == '__main__':
    main()

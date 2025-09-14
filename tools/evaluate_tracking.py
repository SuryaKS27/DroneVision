# evaluate_tracking.py
import motmetrics as mm
import argparse
import os

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a tracking result using MOT Challenge metrics.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--gt', type=str, required=True, help='Path to the ground truth file (e.g., gt/gt.txt).')
    parser.add_argument('--results', type=str, required=True, help='Path to the tracker results file (e.g., results.txt).')
    args = parser.parse_args()

    if not os.path.exists(args.gt):
        print(f"Error: Ground truth file not found at {args.gt}")
        return
    if not os.path.exists(args.results):
        print(f"Error: Results file not found at {args.results}")
        return

    # Load data using the mot15-2D format parser
    # Format: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,-1,-1,-1
    gt = mm.io.loadtxt(args.gt, fmt='mot15-2D')
    ts = mm.io.loadtxt(args.results, fmt='mot15-2D')

    # Create an accumulator that will be updated frame by frame
    acc = mm.MOTAccumulator(auto_id=True)

    # Get the list of frames that have annotations in either ground truth or tracker output
    frames = gt.index.get_level_values('FrameId').union(ts.index.get_level_values('FrameId')).unique()
    
    for frame_id in sorted(frames):
        # Get ground truth and tracker detections for the current frame
        gt_dets = gt[gt.index.get_level_values('FrameId') == frame_id]
        ts_dets = ts[ts.index.get_level_values('FrameId') == frame_id]
        
        # Extract object IDs and bounding boxes
        gt_ids = gt_dets.index.get_level_values('Id').to_numpy()
        ts_ids = ts_dets.index.get_level_values('Id').to_numpy()
        
        gt_boxes = gt_dets[['X', 'Y', 'Width', 'Height']].to_numpy()
        ts_boxes = ts_dets[['X', 'Y', 'Width', 'Height']].to_numpy()
        
        # Calculate the IoU distance matrix between ground truth and tracker detections
        # A max_iou of 0.5 means any pair with IoU < 0.5 is considered a mismatch
        distances = mm.distances.iou_matrix(gt_boxes, ts_boxes, max_iou=0.5)
        
        # Update the accumulator with the results for this frame
        acc.update(gt_ids, ts_ids, distances)

    # Create a metrics host to compute and display the results
    mh = mm.metrics.create()

    # Compute and display the metrics
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='overall')
    
    # Use standard MOT Challenge metric names for display
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)

if __name__ == '__main__':
    main()
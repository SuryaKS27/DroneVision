# # evaluate.py
# import os
# import argparse
# import trackeval

# def main():
#     parser = argparse.ArgumentParser(description='Evaluate tracking results using TrackEval.')
#     parser.add_argument('--gt_path', type=str, required=True, help='Path to the ground truth directory.')
#     parser.add_argument('--results_path', type=str, required=True, help='Path to the tracker results directory.')
#     parser.add_argument('--out_dir', type=str, default='eval_output', help='Directory to save evaluation results.')
#     args = parser.parse_args()
    
#     # --- Configuration for TrackEval ---
#     eval_config = trackeval.Evaluator.get_default_eval_config()
#     # You can disable default metrics if you only want HOTA, etc.
#     # eval_config['METRICS'] = {'HOTA', 'CLEAR', 'Identity'}
    
#     dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
#     dataset_config['GT_FOLDER'] = args.gt_path
#     dataset_config['TRACKERS_FOLDER'] = args.results_path
#     dataset_config['TRACKERS_TO_EVAL'] = ['']  # Evaluate the tracker in the root of TRACKERS_FOLDER
#     dataset_config['SPLIT_TO_EVAL'] = 'train' # The default split for MOTChallenge
#     dataset_config['BENCHMARK'] = 'MOT17' # Use a standard benchmark for metric names
    
#     # By default, TrackEval expects gt.txt inside a sequence folder, e.g., GT_FOLDER/SEQ_NAME/gt/gt.txt
#     # We will tell it our gt.txt is directly in the GT_FOLDER
#     dataset_config['GT_LOC_FORMAT'] = '{gt_folder}/gt.txt'

#     # --- Run Evaluation ---
#     evaluator = trackeval.Evaluator(eval_config)
#     dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    
#     print("Running evaluation...")
#     output_res, output_msg = evaluator.evaluate(dataset_list,-1)
    
#     # The metrics are in a nested dictionary, let's print them
#     metrics = list(output_res['MotChallenge2DBox']['']['']['COMBINED_SEQ'].keys())
    
#     print("\n--- Evaluation Summary ---")
#     for metric in metrics:
#         val = output_res['MotChallenge2DBox']['']['']['COMBINED_SEQ'][metric]
#         print(f"{metric:<15}: {val:.3f}")
        
#     print("\n✅ Evaluation complete.")

# if __name__ == '__main__':
#     main()

# tools/evaluate.py (with visualization)
# tools/evaluate.py (Corrected for custom datasets)
# tools/evaluate.py (Corrected to handle custom datasets without seqinfo.ini)
# tools/evaluate_simple.py
import motmetrics as mm
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Evaluate tracking results using py-motmetrics.')
    parser.add_argument('--gt_file', type=str, required=True, help='Path to the ground truth .txt file.')
    parser.add-argument('--results_file', type=str, required=True, help='Path to the tracker results .txt file.')
    args = parser.parse_args()

    # Load data from files
    gt = mm.io.loadtxt(args.gt_file, fmt='mot15-2D')
    ts = mm.io.loadtxt(args.results_file, fmt='mot15-2D')

    # Create an accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    # Get a list of all frames
    frames = gt.index.get_level_values('FrameId').union(ts.index.get_level_values('FrameId')).unique()
    
    # Process each frame
    for frame_id in frames:
        gt_dets = gt[gt.index.get_level_values('FrameId') == frame_id]
        ts_dets = ts[ts.index.get_level_values('FrameId') == frame_id]
        
        # Calculate the distance (IoU) between ground truth and tracker detections
        distances = mm.distances.iou_matrix(
            gt_dets[['X', 'Y', 'Width', 'Height']].values,
            ts_dets[['X', 'Y', 'Width', 'Height']].values,
            max_iou=0.5
        )
        
        # Update the accumulator with the results for this frame
        acc.update(
            gt_dets.index.get_level_values('Id').values,
            ts_dets.index.get_level_values('Id').values,
            distances
        )

    # Create a metrics host and compute the summary
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='overall')

    # Print the summary
    print(mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    ))

if __name__ == '__main__':
    main()

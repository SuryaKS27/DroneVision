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
# tools/evaluate_simple.py (with visualizations)
import motmetrics as mm
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    parser = argparse.ArgumentParser(description='Evaluate and visualize tracking results with py-motmetrics.')
    parser.add_argument('--gt_file', type=str, required=True, help='Path to the ground truth .txt file.')
    parser.add_argument('--results_file', type=str, required=True, help='Path to the tracker results .txt file.')
    parser.add_argument('--out_dir', type=str, default='evaluation/final_report', help='Directory to save the plots.')
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    gt = mm.io.loadtxt(args.gt_file, fmt='mot15-2D')
    ts = mm.io.loadtxt(args.results_file, fmt='mot15-2D')

    # Create accumulator
    acc = mm.MOTAccumulator(auto_id=True)
    frames = gt.index.get_level_values('FrameId').union(ts.index.get_level_values('FrameId')).unique()
    
    for frame_id in frames:
        gt_dets = gt[gt.index.get_level_values('FrameId') == frame_id]
        ts_dets = ts[ts.index.get_level_values('FrameId') == frame_id]
        distances = mm.distances.iou_matrix(
            gt_dets[['X', 'Y', 'Width', 'Height']].values,
            ts_dets[['X', 'Y', 'Width', 'Height']].values,
            max_iou=0.5
        )
        acc.update(
            gt_dets.index.get_level_values('Id').values,
            ts_dets.index.get_level_values('Id').values,
            distances
        )

    # --- 1. Print the Detailed Text Summary ---
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='overall')
    print("--- Detailed Evaluation Summary ---")
    print(mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    ))

    # --- 2. Generate and Save Visualizations ---
    print("\nGenerating metrics plots...")
    
    # Extract overall summary results for plotting
    overall_summary = summary.loc['overall']

    # Plot 1: Main Performance Metrics (Bar Chart)
    metrics_to_plot = {
        'MOTA': overall_summary['mota'],
        'IDF1': overall_summary['idf1'],
        'Precision': overall_summary['precision'],
        'Recall': overall_summary['recall'],
        'MOTP': overall_summary['motp']
    }
    names = list(metrics_to_plot.keys())
    values = [v * 100 for v in metrics_to_plot.values()] # As percentage

    plt.figure(figsize=(10, 6))
    sns.barplot(x=values, y=names, palette='coolwarm', orient='h')
    for index, value in enumerate(values):
        plt.text(value, index, f' {value:.2f}%', va='center')
    plt.title('Main Performance Metrics', fontsize=16)
    plt.xlabel('Score (%)')
    plt.xlim(0, 105)
    
    plot1_path = os.path.join(args.out_dir, 'summary_metrics.png')
    plt.savefig(plot1_path, bbox_inches='tight')
    print(f"  -> Summary bar chart saved to: {plot1_path}")

    # Plot 2: Error Breakdown (Pie Chart)
    error_counts = {
        'False Positives': overall_summary['num_false_positives'],
        'Misses': overall_summary['num_misses'],
        'ID Switches': overall_summary['num_switches']
    }
    # Filter out zero-value errors to avoid cluttering the pie chart
    errors_to_plot = {k: v for k, v in error_counts.items() if v > 0}
    
    if errors_to_plot:
        plt.figure(figsize=(8, 8))
        plt.pie(errors_to_plot.values(), labels=errors_to_plot.keys(), autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Reds_r'))
        plt.title('Breakdown of Tracking Errors', fontsize=16)
        
        plot2_path = os.path.join(args.out_dir, 'error_breakdown.png')
        plt.savefig(plot2_path, bbox_inches='tight')
        print(f"  -> Error breakdown pie chart saved to: {plot2_path}")

    print("✅ Visualization complete.")

if __name__ == '__main__':
    main()

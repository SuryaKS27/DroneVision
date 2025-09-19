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
import os
import argparse
import trackeval
import matplotlib.pyplot as plt
import seaborn as sns

def get_seq_length_from_gt(gt_file_path):
    """Reads a gt.txt file to determine the number of frames."""
    if not os.path.exists(gt_file_path):
        raise FileNotFoundError(f"Ground truth file not found at: {gt_file_path}")
    
    last_frame = 0
    with open(gt_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            try:
                frame_id = int(parts[0])
                if frame_id > last_frame:
                    last_frame = frame_id
            except (ValueError, IndexError):
                continue
    return last_frame

def main():
    parser = argparse.ArgumentParser(description='Evaluate and visualize tracking results.')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to the directory containing the gt.txt file.')
    parser.add_argument('--results_path', type=str, required=True, help='Path to the directory containing the results.txt file.')
    parser.add_argument('--out_dir', type=str, default='evaluation/final_report', help='Directory to save evaluation results and plot.')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # --- Configuration for TrackEval ---
    eval_config = trackeval.Evaluator.get_default_eval_config()
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    
    # --- KEY CHANGES FOR CUSTOM DATASET ---
    # 1. Manually determine sequence length from the ground truth file.
    gt_file = os.path.join(args.gt_path, 'gt.txt')
    seq_length = get_seq_length_from_gt(gt_file)
    
    # 2. Set BENCHMARK to 'Custom'.
    dataset_config['BENCHMARK'] = 'Custom'
    
    # 3. Provide the sequence name AND its length. This prevents the need for seqinfo.ini.
    dataset_config['SEQ_INFO'] = {'drone-sequence': {'seqLength': seq_length}}
    
    # 4. Set paths and tell the evaluator where to find the files directly.
    dataset_config['GT_FOLDER'] = args.gt_path
    dataset_config['TRACKERS_FOLDER'] = args.results_path
    dataset_config['TRACKERS_TO_EVAL'] = [''] 
    dataset_config['SPLIT_TO_EVAL'] = 'train'
    dataset_config['GT_LOC_FORMAT'] = '{gt_folder}/gt.txt'

    # --- Run Evaluation ---
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    
    print("Running evaluation...")
    output_res, output_msg = evaluator.evaluate(dataset_list)
    
    results_dict = output_res['MotChallenge2DBox']['']['drone-sequence']
    
    mh = trackeval.metrics.create_metrics(eval_config['METRICS'])
    summary = mh.compute_unique_metrics(results_dict)
    strsummary = trackeval.io.render_summary(summary, formatters=mh.get_display_name_formatters())
    print("\n--- Detailed Evaluation Summary ---")
    print(strsummary)
    
    # --- Visualize the Key Metrics ---
    print("\nGenerating metrics plot...")
    
    metrics_to_plot = {
        'HOTA': summary['HOTA']['HOTA'],
        'MOTA': summary['CLEAR']['MOTA'],
        'IDF1': summary['Identity']['IDF1'],
        'AssA': summary['HOTA']['AssA'],
        'DetA': summary['HOTA']['DetA'],
        'MOTP': summary['CLEAR']['MOTP'],
    }
    
    names = list(metrics_to_plot.keys())
    values = [v * 100 for v in metrics_to_plot.values()]

    plt.figure(figsize=(12, 7))
    sns.barplot(x=values, y=names, palette='viridis', orient='h')
    
    for index, value in enumerate(values):
        plt.text(value, index, f' {value:.2f}%', va='center', fontsize=10)
        
    plt.title('Tracking Performance Metrics Summary', fontsize=16)
    plt.xlabel('Score (%)', fontsize=12)
    plt.xlim(0, 105)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plot_path = os.path.join(args.out_dir, 'evaluation_summary.png')
    plt.savefig(plot_path, bbox_inches='tight')
    
    print(f"✅ Evaluation complete. Plot saved to: {plot_path}")

if __name__ == '__main__':
    main()
